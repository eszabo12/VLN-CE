# sys.path.append('')

from vlnce_baselines.config.default import get_config
from vlnce_baselines.models.seq2seq_policy import Seq2SeqNet
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders import resnet_encoders
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
import gym
from gym import spaces
from importlib import import_module

import sys
sys.path.append('/home/elle/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/')
from interbotix_xs_modules.locobot import InterbotixLocobotCreate3XS
import pyrealsense2 as rs
from sensor_msgs.msg import Image

observation = {
    "instruction" : gym.spaces.Box(low=0, high=100, shape=(2,)),
    "depth" : gym.spaces.Box(low=0, high=100, shape=(2,)), # [BATCH, HEIGHT, WIDTH, CHANNEL]
    "rgb" : gym.spaces.Box(low=0, high=100, shape=(2,))#imgs: must have pixel values ranging from 0-255. Assumes a size of [Bx3xHxW]
}

locobot = InterbotixLocobotCreate3XS(robot_model="locobot_base")
observation_space = spaces.Dict(observation)
model_config = get_config("vlnce_baselines/config/r2r_baselines/seq2seq.yaml")
#use previous action is false by default

num_actions = 2
model = Seq2SeqNet(observation_space, model_config, num_actions)
prev_actions = []
x, rnn_states_out = model.forward(self, observations, rnn_states, prev_actions, masks)
