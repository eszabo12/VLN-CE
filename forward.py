# sys.path.append('')

from vlnce_baselines.config.default import get_config
from vlnce_baselines.models.seq2seq_policy import Seq2SeqNet
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders import resnet_encoders
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
import gym
import gzip
from gym import spaces
from importlib import import_module
import torch

import sys
import numpy as np
sys.path.append('/home/elle/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/')
from interbotix_xs_modules.locobot import InterbotixLocobotCreate3XS
import pyrealsense2 as rs
from sensor_msgs.msg import Image
import einops

with gzip.open('data/datasets/R2R_VLNCE_v1-3/test/test.json.gz','r') as fin:        
    for line in fin:        
        print('got line', line[0])
        print("line size,", len(line))
        break
""" "instruction_text": "Turn right through the large doorway into the living room. Walk straight past the couches on the left. Turn right into the kitchen and pause by the oven. ", "instruction_tokens": [2494, 1968, 2418, 2389, 1336, 766, 1264, 2389, 1404, 1994, 15, 2584, 2288, 1728, 2389, 595, 1613, 2389, 1360, 15, 2494, 1968, 1264, 2389, 1306, 119, 1741, 404, 2389, 1667, 15
]
"""
seq_length = 50
vocab_size = 2504
batch_size= 5 # cuz that's what it says in the yaml
observation = {
    "instruction" : gym.spaces.Box(low=0, high=100, shape=(vocab_size, seq_length)),
    "depth" : gym.spaces.Box(low=0, high=100, shape=(256, 640, 1)), # [BATCH, HEIGHT, WIDTH, CHANNEL] #480 originally
    "rgb" : gym.spaces.Box(low=0, high=256, shape=(480, 640, 3))#imgs: must have pixel values ranging from 0-255. Assumes a size of [Bx3xHxW]
# color frame shape (480, 640, 3)
}


def get_observation(locobot):
    observations = {}
    #instruction tokens
    instruction = torch.Tensor([2494, 1968, 2418, 2389, 1336, 766, 1264, 2389, 1404, 1994, 15, 2584, 2288, 1728, 2389, 595, 1613, 2389, 1360, 15, 2494, 1968, 1264, 2389, 1306, 119, 1741, 404, 2389, 1667, 15
    ])
    instruction = einops.repeat(instruction, 'm-> k m', k=batch_size)
    observations["instruction"] = instruction
    
    color_image, depth_image = locobot.base.get_img()
    color_image = einops.repeat(color_image, 'm n l -> k m n l', k=batch_size)
                # observations: [BATCH, HEIGHT, WIDTH, CHANNEL]

    depth_image = einops.repeat(depth_image, 'm n l-> k m n l', k=batch_size)
    observations["depth"] = depth_image
    observations["rgb"] = color_image
    return observations



locobot = InterbotixLocobotCreate3XS(robot_model="locobot_base")
observation_space = spaces.Dict(observation)
model_config = get_config("vlnce_baselines/config/r2r_baselines/seq2seq.yaml").MODEL
#use previous action is false by default

num_actions = 2 
model = Seq2SeqNet(observation_space, model_config, num_actions)

observations = get_observation(locobot)
prev_actions = []
rnn_states = []
masks = []

x, rnn_states_out = model.forward(observations, rnn_states, prev_actions, masks)
print("forward pass compelte")
