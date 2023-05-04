# from vlnce_baselines.dagger_trainer import DaggerTrainer
from vlnce_baselines.config.default import get_config
from vlnce_baselines.models.waypoint_policy import WaypointPolicy
from vlnce_baselines.models.cma_policy import CMA_Policy
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders import resnet_encoders
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.common.env_utils import construct_envs
from habitat_baselines.utils.common import CategoricalNet

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.utils import extract_instruction_tokens
from habitat_baselines.common.obs_transformers import (
    CenterCropper,
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
import gym
import gzip
import json
from gym import spaces
from importlib import import_module
import torch
import torchvision
import cv2
import time
import os
import datetime

import sys
import numpy as np
import rospy
sys.path.append('/home/elle/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/')
from interbotix_xs_modules.locobot import InterbotixLocobotCreate3XS
import pyrealsense2 as rs
from sensor_msgs.msg import Image
import einops
import math
from habitat.core.simulator import ActionSpaceConfiguration
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from embeddings import BERTProcessor
from PIL import Image as PILIMAGE
# import matplotlib as plt

locobot = InterbotixLocobotCreate3XS(robot_model="locobot_base")
locobot.camera.pan_tilt_go_home()



def do_action(action, locobot):
    if locobot == None:
        return
    if action == 0:
        print("stop")
    elif action == 1:
        locobot.base.move(0.25, 0, 1.0)
    #turn right
    elif action == 2:
        locobot.base.move(0.1, -math.pi / 6.0, 1.3)
    #turn left
    elif action == 3:
        locobot.base.move(0.1, math.pi / 6.0, 1.3)
    elif action == 4:
        locobot.camera.tilt( -math.pi / 6.0)
        time.sleep(1)
    elif action == 5:
        locobot.camera.tilt(math.pi / 6.0)
    time.sleep(1)
    return get_observation(locobot)


input_text = "Stay to the left of the towel then take a right toward the fridge."

vocab_size = 2504
batch_size= 1
seq_length = len(input_text.split())
observation = {
    "instruction" : gym.spaces.Box(low=0, high=100, shape=(vocab_size, seq_length)),
    "depth" : gym.spaces.Box(low=0, high=1, shape=(256, 256, 1)), # [BATCH, HEIGHT, WIDTH, CHANNEL] #480 originally 
    "rgb" : gym.spaces.Box(low=0, high=256, shape=(256, 256, 3))#imgs: must have pixel values ranging from 0-255. Assumes a size of [Bx3xHxW] # color frame shape og (480, 640, 3)
}
# input_text = "go down the middle of the hallway. At the chairs and the table, turn right. Go into the crevice in between the two couches and face the window. Then turn around and walk towards the elevator. Face the elevator. At the elevator doors, turn left down the hallway. Once you see a wall, turn left. Stop when you see the black wheeled desk chair at the end of the hallway."

processor = BERTProcessor()
feats = processor.get_instruction_embeddings(input_text)
image_crop = CenterCropper(256, tuple(("rgb")))
depth_crop = CenterCropper(224, tuple(("depth")))

now = datetime.datetime.now().strftime("%I:%M:%S")
folder_path = "./logs/" + now
os.mkdir(folder_path)
print("done setting up")
def get_observation(locobot):
    observations = {}
    instruction = einops.repeat(feats, 'm a -> k m a', k=batch_size)

    observations["rxr_instruction"] = torch.Tensor(instruction)
    color_image = None
    depth_image = None
    if locobot != None:
        color_image, depth_image = locobot.base.get_img()

        #print("color, depth size", color_image.shape, depth_image.shape)
        color_image = torch.Tensor(color_image)
        depth_image = torch.Tensor(depth_image)
        depth_image = depth_image[:, 112:-112] / 255.0

        color_image = color_image[:, 80:-80]
        
        color_image = torch.nn.functional.interpolate(color_image.transpose(0,-1).unsqueeze(0), size=224).transpose(1,-1).squeeze()
        depth_image = torch.nn.functional.interpolate(depth_image.transpose(0,-1).unsqueeze(0), size=256).transpose(1,-1).squeeze()

        nans = torch.isnan(depth_image)
        zeros = torch.zeros_like(depth_image)
        depth_image = torch.where(nans, zeros, depth_image)
        observations["depth"] = depth_image.unsqueeze(0)
        observations["rgb"] = color_image.unsqueeze(0)
    else:
        color_image = torch.load("./saved_images/rgb.pt")
        depth_image = torch.load("./saved_images/depth.pt")
        print(depth_image.size())
        print(color_image.size())
        depth_image = depth_image[0, 112:368,192:448] / 255.0
        color_image = color_image[0, 80:-80]
        torch.nn.functional.interpolate(color_image.float(), size=224)
        observations["depth"] = depth_image.unsqueeze(0)
        observations["rgb"] = color_image.unsqueeze(0)
    print("depth size", observations["depth"].size())
    print("rgb size", observations["rgb"].size())

    im = PILIMAGE.fromarray(observations["rgb"].squeeze().numpy().astype(np.uint8))
    now = datetime.datetime.now().strftime("%I:%M:%S")
    im.save(folder_path +"/" + now + ".png", "PNG")
    return observations

observation_space = spaces.Dict(observation)

#set up log files
now = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
logfile = open("./logs/log " + now, "w")
logfile.write("instruction: " + input_text + "\n")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#distance isn't continuous, but offset is. can try the other config files/actions later
config = get_config(BASE_DIR + "/VLN-CE/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml")
sim_config = get_config(BASE_DIR + "/VLN-CE/habitat_extensions/config/rxr_vlnce_english_task.yaml").SIMULATOR
step_config = get_config(BASE_DIR + "/VLN-CE/habitat_extensions/config/vlnce_task.yaml").SIMULATOR
sim_config.update(step_config)
'''action space '''
actions = HabitatSimV1ActionSpaceConfiguration(sim_config)
action_space = spaces.Discrete(6)
action_config = actions.get()
print(config.MODEL.DEPTH_ENCODER)

policy = CMA_Policy(observation_space, action_space, config.MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------from dagger trainer file----------

rnn_states = torch.zeros(
    1, #verify num_envs is 1 in the sim
    policy.net.num_recurrent_layers,
    config.MODEL.STATE_ENCODER.hidden_size,
    device=device,
)
prev_actions = torch.zeros(
    1,
    1,
    device=device,
    dtype=torch.long,
)
not_done_masks = torch.zeros(
    1, 1, dtype=torch.uint8, device=device
)

print("rnn states shape", rnn_states.shape)

counter = 0
observation = get_observation(locobot)
actions, rnn_states = policy.act(observation, rnn_states, prev_actions, not_done_masks)
print("actions:", actions)
max_actions = 250
while(counter < max_actions):
    actions, rnn_states = policy.act(observation, rnn_states, prev_actions, not_done_masks)
    print("actions:", actions[0])
    logfile.write("Action performed: " + str(actions[0]) + "\n")
    counter += 1
    prev_actions = actions.unsqueeze(0)
    observation = do_action(actions[0], locobot)
locobot.camera.pan_tilt_go_home()
logfile.close()