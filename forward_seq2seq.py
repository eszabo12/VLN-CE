
from vlnce_baselines.config.default import get_config
from vlnce_baselines.models.seq2seq_policy import Seq2SeqNet, Seq2SeqPolicy
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders import resnet_encoders
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from habitat_baselines.utils.common import CategoricalNet

import gym
import gzip
from gym import spaces
from importlib import import_module
import torch
import torchvision
import cv2

import argparse
import sys
import numpy as np
sys.path.append('/home/elle/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/')
from interbotix_xs_modules.locobot import InterbotixLocobotCreate3XS
import pyrealsense2 as rs
from sensor_msgs.msg import Image
import einops
import time
import math


#the interbotix camera move function ensures that the movement isn't outside of the range of its limits.
def do_action(action, locobot, load_image):
    if action == 0:
        print("stop")
        return None
    elif action == 1:
        print("move forward")
        locobot.base.move(0.25, 0, 0.5)
    elif action == 2:
        print("turn right")
        locobot.base.move(0, math.pi / 12.0, 0.5)
    elif action == 3:
        print("turn left")
        locobot.base.move(0, -1 * math.pi / 12.0, 0.5)
    elif action == 4:
        print("look up")
        locobot.camera.tilt(0.3)
        time.sleep(1)
    elif action == 5:
        print("look down")
        locobot.camera.tilt(-0.3)
        time.sleep(1)
    
    return get_observation(locobot, load_image)


# with gzip.open('data/datasets/R2R_VLNCE_v1-3/test/test.json.gz','r') as fin:        
#     for line in fin:        
#         print('got line', line[0])
#         print("line size,", len(line))
#         break
# """ "instruction_text": "Turn right through the large doorway into the living room. Walk straight past the couches on the left. Turn right into the kitchen and pause by the oven. ", "instruction_tokens": [2494, 1968, 2418, 2389, 1336, 766, 1264, 2389, 1404, 1994, 15, 2584, 2288, 1728, 2389, 595, 1613, 2389, 1360, 15, 2494, 1968, 1264, 2389, 1306, 119, 1741, 404, 2389, 1667, 15
# ]

seq_length = 50
vocab_size = 2504
batch_size= 5 # cuz that's what it says in the yaml
observation = {
    "instruction" : gym.spaces.Box(low=0, high=100, shape=(vocab_size, seq_length)),
    "depth" : gym.spaces.Box(low=0, high=100, shape=(256, 640, 1)), # [BATCH, HEIGHT, WIDTH, CHANNEL] #480 originally
    "rgb" : gym.spaces.Box(low=0, high=256, shape=(480, 640, 3))#imgs: must have pixel values ranging from 0-255. Assumes a size of [Bx3xHxW]
# color frame shape (480, 640, 3)
}
# instruction = torch.Tensor([2494, 780, 2389, 1126, 108, 15] + [0]*(50-6)).long() #ADDED pad token where token exceeded vocab size
instruction = torch.Tensor([2494, 780, 2389, 1126, 108, 15, 178, 2389, 589, 2494, 1968, 15] + [0]*(seq_length-12)).long() #ADDED pad token where token exceeded vocab size
#walk down the hallway, at the corner turn right.
# 178, 2389, 589, 2494, 1968
print("instruction length", instruction.size())
instruction = einops.repeat(instruction, 'm -> k m', k=batch_size)

def get_observation(locobot, load_image):
    observations = {}
    observations["instruction"] = instruction
    color_image = depth_image = []
    if not load_image:
        color_image, depth_image = locobot.base.get_img()
        color_image = torch.Tensor(einops.repeat(color_image, 'm n l -> k m n l', k=batch_size)).long()
                # observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        # print("rgb size", color_image.size())

        depth_image = torch.Tensor(einops.repeat(depth_image, 'm n l-> k m n l', k=batch_size))/ 255.0
        # print("depth size", depth_image.size())
        observations["depth"] = torchvision.transforms.Resize((256, 256))(
            depth_image[:, 80:-80].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        observations["rgb"] = torchvision.transforms.Resize((256, 256))(
            color_image[:, 80:-80].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # print("observation, depth sizes", observations["rgb"].size(), observations["depth"].size())
        torch.save(observations["depth"], "saved_images/depth.pt")
        torch.save(observations["rgb"], "saved_images/rgb.png")
    else:
        color_image = torch.load("saved_images/rgb.pt")
        depth_image = torch.load("saved_images/depth.pt")
        observations["depth"] = depth_image
        observations["rgb"] = color_image
    # cv2.imshow(observations["rgb"], "color" )
    # cv2.waitkey(0)
    # cv2.destroyAllWindows()
    
    return observations

locobot = InterbotixLocobotCreate3XS(robot_model="locobot_base")
observation_space = spaces.Dict(observation)
config = get_config("vlnce_baselines/config/r2r_baselines/seq2seq.yaml")

num_actions = 6
action_space = spaces.Discrete(num_actions)
prev_actions = torch.zeros(6) # it doesn't even matter what this is because it doesn't get checked unless check_prev_actions is true
# rnn_states = torch.zeros(
#     envs.num_envs,
#     self.policy.net.num_recurrent_layers,
#     self.config.MODEL.STATE_ENCODER.hidden_size,
#     device=self.device,
# )

masks = torch.ones(896, 512).bool()
# import pdb; pdb.set_trace()

policy = Seq2SeqPolicy(observation_space, action_space, config.MODEL)
rnn_states = torch.zeros(
    1, # num envs
    batch_size, #num layers
    config.MODEL.STATE_ENCODER.hidden_size,
)

print("rnn states shape", rnn_states.shape)
policy.eval()

# actions, rnn_states = policy.act(observations, rnn_states, prev_actions, masks)
# print("actions:", actions.size(), actions)
#   POSSIBLE_ACTIONS: ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"]
# 30 degrees move left and right, .25 step size, 30 degrees


if __name__ == '__main__':
    # locobot.base.move(0.5, 0, 30)
    # observation = do_action(2, locobot, True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_image', action='store_true')
    args = parser.parse_args()
    load_image = args.load_image
    counter = 0
    locobot.camera.pan_tilt_go_home()
    observation = get_observation(locobot, load_image)
    actions, rnn_states = policy.act(observation, rnn_states, prev_actions, masks)
    observation = do_action(actions[0], locobot, load_image)
    # print("actions:", actions.size(), actions)
    max_actions = 25
    while(observation != None and counter < max_actions):
        actions, rnn_states = policy.act(observation, rnn_states, prev_actions, masks)
        observation = do_action(actions[0], locobot, load_image)
        # print("actions:", actions.size(), actions[0])
        counter += 1
    locobot.camera.pan_tilt_go_home()