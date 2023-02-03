from vlnce_baselines.dagger_trainer import DaggerTrainer
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

import sys
import numpy as np
sys.path.append('/home/elle/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/')
from interbotix_xs_modules.locobot import InterbotixLocobotCreate3XS
import pyrealsense2 as rs
from sensor_msgs.msg import Image
import einops
import time
import math

def do_action(action, locobot):
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


# data = []
# with gzip.open('data/datasets/R2R_VLNCE_v1-3/test/test.json.gz','r') as fin:    
#     i=0    
#     for line in fin:
#         data.append(line)
#         i +=1
#         if i > 40:
#             break
# # json_str = json.dumps(data) + "\n"               # 2. string (i.e. JSON)
# with open("ex_data.json", 'w') as fout:       # 4. fewer bytes (i.e. gzip)
#     fout.write(data)
# """ "instruction_text": "Turn right through the large doorway into the living room. Walk straight past the couches on the left. Turn right into the kitchen and pause by the oven. ", "instruction_tokens": [2494, 1968, 2418, 2389, 1336, 766, 1264, 2389, 1404, 1994, 15, 2584, 2288, 1728, 2389, 595, 1613, 2389, 1360, 15, 2494, 1968, 1264, 2389, 1306, 119, 1741, 404, 2389, 1667, 15
# ]
# """

seq_length = 50
vocab_size = 2504
batch_size= 5
observation = {
    "instruction" : gym.spaces.Box(low=0, high=100, shape=(vocab_size, seq_length)),
    "depth" : gym.spaces.Box(low=0, high=1, shape=(256, 256, 1)), # [BATCH, HEIGHT, WIDTH, CHANNEL] #480 originally 
    "rgb" : gym.spaces.Box(low=0, high=256, shape=(256, 256, 3))#imgs: must have pixel values ranging from 0-255. Assumes a size of [Bx3xHxW] # color frame shape og (480, 640, 3)

}

# instruction_text = "Turn right through the large doorway into the living room. Walk straight past the couches on the left. Turn right into the kitchen and pause by the oven. "
# instruction_text = VocabFromText([instruction_text])
# "Walk down the hallway
# instruction = torch.Tensor([2584, 780, 2389, 1126, 108, 15] + [0]*(50-6)).long() #ADDED pad token where token exceeded vocab size
# instruction = torch.Tensor([2494, 159, 119, 15]+ [0]*(50-4)).long() #turn around
print("instruction length", instruction.size())
instruction = "Turn right"
instruction = extract_instruction_tokens(instruction)
def get_observation(locobot):
    observations = {}

    instruction = einops.repeat(instruction, 'm -> k m', k=batch_size)

    observations["instruction"] = instruction
    color_image, depth_image = locobot.base.get_img()
    color_image = torch.Tensor(einops.repeat(color_image, 'm n l -> k m n l', k=batch_size)).long()
    print("rgb size", color_image.size())
    depth_image = torch.Tensor(einops.repeat(depth_image, 'm n l-> k m n l', k=batch_size))/ 255.0
    observations["depth"] = depth_image[:, 112:-112]
    print("depth size", observations["depth"].size())
    observations["rgb"] = color_image
    # torch.save(color_image, "./saved_images/rgb.pt")
    # torch.save(depth_image, "./saved_images/depth.pt")
    observations["depth"] = torch.load("./saved_images/depth.pt")[:, 112:-112, 192:-192]
    observations["rgb"] = torch.load("./saved_images/rgb.pt")[:, 112:-112, 192:-192]
    print("depth size", observations["depth"].size())
    print("rgb size", observations["rgb"].size())

    observations["rgb_history"] = color_image
    observations["depth_history"] = depth_image
    observations["angle_features"] = torch.zeros(10)
    return observations

# from VLN-CE/habitat_extensions/config/vlnce_waypoint_task.yaml
#   RGB_SENSOR:
#     WIDTH: 224
#     HEIGHT: 224
#     HFOV: 90
#     TYPE: HabitatSimRGBSensor
#   DEPTH_SENSOR:
#     WIDTH: 256
#     HEIGHT: 256
locobot = InterbotixLocobotCreate3XS(robot_model="locobot_base")
observation_space = spaces.Dict(observation)
#distance isn't continuous, but offset is. can try the other config files/actions later
config = get_config("/home/elle/elle_ws/VLN-CE/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml")
policy = CMA_Policy(observation_space, action_space, config.MODEL)
device = torch.get_device()
# ---------from dagger trainer file----------

envs = construct_envs(config, get_env_class(config.ENV_NAME))
expert_uuid = config.IL.DAGGER.expert_policy_sensor_uuid

rnn_states = torch.zeros(
    envs.num_envs,
    policy.net.num_recurrent_layers,
    config.MODEL.STATE_ENCODER.hidden_size,
    device=device,
)
prev_actions = torch.zeros(
    envs.num_envs,
    1,
    device=device,
    dtype=torch.long,
)
not_done_masks = torch.zeros(
    envs.num_envs, 1, dtype=torch.uint8, device=device
)

observations = envs.reset()
observations = extract_instruction_tokens(
    observations, config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
)
batch = batch_obs(observations, device)
batch = apply_obs_transforms_batch(batch, obs_transforms)



#   POSSIBLE_ACTIONS: ['STOP', 'GO_TOWARD_POINT']
actions = {}
actions["r"] = spaces.Box(0.0, 215.0, (1,), np.float64)
actions["theta"] = spaces.Box(0.0, math.pi*2, (1,), np.float64)
first_space = spaces.Dict(actions)
stop_space = spaces.Box(0, 0)
full_dict = {"STOP:": stop_space, "GO_TOWARD_POINT":first_space}
action_space = spaces.Dict(full_dict)
print("observation space:", observation_space)

observations = get_observation(locobot)
prev_actions = torch.zeros(batch_size, 512)
masks = torch.ones(896, 512).bool()
rnn_states = torch.zeros(
    1, # num envs
    batch_size, #num layers
    config.MODEL.STATE_ENCODER.hidden_size,
)

print("rnn states shape", rnn_states.shape)

# actions, rnn_states = policy.act(observations, rnn_states, prev_actions, masks)
# print("actions:", actions.size(), actions)
#   POSSIBLE_ACTIONS: ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"]
# 30 degrees move left and right, .25 step size, 30 degrees
counter = 0
observation = get_observation(locobot)
value, actions, action_elements, modes, variances, action_log_probs, rnn_states_out, pano_stop_distribution = policy.act(observation, rnn_states, prev_actions, masks)
# observation = do_action(actions[0], locobot)
print("actions:", actions.size(), actions)
# max_actions = 10
# while(observation != None and counter < max_actions):
#     actions, rnn_states = policy.act(observation, rnn_states, prev_actions, masks)
#     observation = do_action(actions[0], locobot)
#     print("actions:", actions.size(), actions[0])
#     counter += 1
# locobot.camera.pan_tilt_go_home()

#apply_obs_transforms_batch