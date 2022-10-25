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
import torchvision

import sys
import numpy as np
sys.path.append('/home/elle/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/')
from interbotix_xs_modules.locobot import InterbotixLocobotCreate3XS
import pyrealsense2 as rs
from sensor_msgs.msg import Image
import einops

# with gzip.open('data/datasets/R2R_VLNCE_v1-3/test/test.json.gz','r') as fin:        
#     for line in fin:        
#         print('got line', line[0])
#         print("line size,", len(line))
#         break
# """ "instruction_text": "Turn right through the large doorway into the living room. Walk straight past the couches on the left. Turn right into the kitchen and pause by the oven. ", "instruction_tokens": [2494, 1968, 2418, 2389, 1336, 766, 1264, 2389, 1404, 1994, 15, 2584, 2288, 1728, 2389, 595, 1613, 2389, 1360, 15, 2494, 1968, 1264, 2389, 1306, 119, 1741, 404, 2389, 1667, 15
# ]
# """
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
    # instruction = torch.Tensor([2494, 1968, 2418, 2389, 1336, 766, 1264, 2389, 1404, 1994, 15, 2584, 2288, 1728, 2389, 595, 1613, 2389, 1360, 15, 2494, 1968, 1264, 2389, 1306, 119, 1741, 404, 2389, 1667, 15
    # ] + [0]*(50-31)).long()
    instruction = torch.Tensor([2494, 1968, 2418, 2389, 1336, 766, 1264, 2389, 1404, 1994, 15, 0, 2288, 1728, 2389, 595, 1613, 2389, 1360, 15, 2494, 1968, 1264, 2389, 1306, 119, 1741, 404, 2389, 1667, 15
    ] + [0]*(50-31)).long() #ADDED pad token where token exceeded vocab size
    # instruction = torch.nn.functional.one_hot(instruction.squeeze(), num_classes=2504)
    # print("instruction size,", instruction.size())

    instruction = einops.repeat(instruction, 'm -> k m', k=batch_size)


    observations["instruction"] = instruction
    
    color_image, depth_image = locobot.base.get_img()
    color_image = torch.Tensor(einops.repeat(color_image, 'm n l -> k m n l', k=batch_size)).long()
                # observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
    print("rgb size", color_image.size())

    depth_image = torch.Tensor(einops.repeat(depth_image, 'm n l-> k m n l', k=batch_size))/ 255.0
    print("depth size", depth_image.size())
    observations["depth"] = torchvision.transforms.Resize((256, 256))(
        depth_image[:, 80:-80].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    observations["rgb"] = color_image

    return observations


locobot = InterbotixLocobotCreate3XS(robot_model="locobot_base")
observation_space = spaces.Dict(observation)
model_config = get_config("vlnce_baselines/config/r2r_baselines/seq2seq.yaml").MODEL
#use previous action is false by default

num_actions = 2 
model = Seq2SeqNet(observation_space, model_config, num_actions)

observations = get_observation(locobot)
    # N = rnn_states.size(1)
    # T = x.size(0) // N
prev_actions = torch.zeros(1) # it doesn't even matter what this is because it doesn't get checked unless check_prev_actions is true
rnn_states = torch.zeros(2, 512, 1) # i made the first dimension one more than x's first dimension to avoid weird sliving
masks = torch.ones(896, 512)
# import pdb; pdb.set_trace()
print("model output size", model_config.STATE_ENCODER.hidden_size)
x, rnn_state_out = model.forward(observations, rnn_states, prev_actions, masks)
print("forward pass complete", x.size(), x.type())