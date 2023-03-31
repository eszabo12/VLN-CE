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

import sys
import numpy as np
import rospy
sys.path.append('/home/elle/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/')
from interbotix_xs_modules.locobot import InterbotixLocobotCreate3XS
import pyrealsense2 as rs
from sensor_msgs.msg import Image
import einops
import time
import math
from habitat.core.simulator import ActionSpaceConfiguration
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from embeddings import BERTProcessor


locobot = InterbotixLocobotCreate3XS(robot_model="locobot_base")


def do_action(action, locobot):
    if locobot == None:
        return
    if action == 0:
        print("stop")
        return None
    elif action == 1:
        locobot.base.move(0.25, 0, 1.0)
    elif action == 2:
        locobot.base.move(0.25, math.pi / 12.0, 1)
    elif action == 3:
        locobot.base.move(0.25, math.pi / 12.0, 1)
    elif action == 4:
        locobot.camera.tilt(0.8)
        time.sleep(1)
    elif action == 5:
        locobot.camera.tilt(-0.3)
        time.sleep(1)
    return get_observation(locobot)


seq_length = 50
vocab_size = 2504
batch_size= 5
observation = {
    "instruction" : gym.spaces.Box(low=0, high=100, shape=(vocab_size, seq_length)),
    "depth" : gym.spaces.Box(low=0, high=1, shape=(256, 256, 1)), # [BATCH, HEIGHT, WIDTH, CHANNEL] #480 originally 
    "rgb" : gym.spaces.Box(low=0, high=256, shape=(256, 256, 3))#imgs: must have pixel values ranging from 0-255. Assumes a size of [Bx3xHxW] # color frame shape og (480, 640, 3)
}

input_text = "Go down the hallway, turning left at the end, then turn back around and go back down the same hallway. stop when you reach the couches."
processor = BERTProcessor()
feats = processor.get_instruction_embeddings(input_text)

print("done setting up")
def get_observation(locobot):
    observations = {}
    instruction = einops.repeat(feats, 'm a -> k m a', k=batch_size)

    observations["rxr_instruction"] = torch.Tensor(instruction)
    color_image = None
    depth_image = None
    if locobot != None:
        color_image, depth_image = locobot.base.get_img()
        color_image = torch.Tensor(einops.repeat(color_image, 'm n l -> k m n l', k=batch_size)).long()
        print("rgb size", color_image.size())
        depth_image = torch.Tensor(einops.repeat(depth_image, 'm n l-> k m n l', k=batch_size))/ 255.0
        observations["depth"] = depth_image[:, 112:-112]
        print("depth size", observations["depth"].size())
        observations["rgb"] = color_image
    # torch.save(color_image, "./saved_images/rgb.pt")
    # torch.save(depth_image, "./saved_images/depth.pt")
    else:
        color_image = torch.load("./saved_images/rgb.pt")
        depth_image = torch.load("./saved_images/depth.pt")
        depth_image = depth_image[:, 128:-128, 192:416] / 255.0
        color_image = color_image[:, 96:352, 128:384]
        print(color_image.size())
        observations["depth"] = depth_image
        observations["rgb"] = color_image
    print("depth size", observations["depth"].size())
    print("rgb size", observations["rgb"].size())
    # observations["rgb_history"] = color_image
    # observations["depth_history"] = depth_image
    # observations["angle_features"] = torch.zeros(10)
    return observations


observation_space = spaces.Dict(observation)



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#distance isn't continuous, but offset is. can try the other config files/actions later
config = get_config(BASE_DIR + "/VLN-CE/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml")
print(config.MODEL)
#add the config.SIMULATOR attributes
sim_config = get_config(BASE_DIR + "/VLN-CE/habitat_extensions/config/rxr_vlnce_english_task.yaml").SIMULATOR
# step size config- uses v0
step_config = get_config(BASE_DIR + "/VLN-CE/habitat_extensions/config/vlnce_task.yaml").SIMULATOR
print(step_config)
# /home/elle/Repos/research/VLN-CE/habitat_extensions/config/vlnce_task.yaml
sim_config.update(step_config)
'''action space '''
actions = HabitatSimV1ActionSpaceConfiguration(sim_config)
action_space = spaces.Discrete(6)
action_config = actions.get()


policy = CMA_Policy(observation_space, action_space, config.MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------from dagger trainer file----------

# envs = construct_envs(config, get_env_class(config.ENV_NAME))
expert_uuid = config.IL.DAGGER.expert_policy_sensor_uuid

rnn_states = torch.zeros(
    batch_size,
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
print(config.INFERENCE.CKPT_PATH)
value, actions, action_elements, modes, variances, action_log_probs, rnn_states_out, pano_stop_distribution = policy.act(observation, rnn_states, prev_actions, not_done_masks)
# observation = do_action(actions[0], locobot)
print("actions:", actions.size(), actions)
# max_actions = 10
# while(observation != None and counter < max_actions):
#     actions, rnn_states = policy.act(observation, rnn_states, prev_actions, masks)
#     observation = do_action(actions[0], locobot)
#     print("actions:", actions.size(), actions[0])
#     counter += 1
# locobot.camera.pan_tilt_go_home()

# apply_obs_transforms_batch