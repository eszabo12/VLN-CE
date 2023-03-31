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
# import rospy
sys.path.append('/home/elle/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/')
from interbotix_xs_modules.locobot import InterbotixLocobotCreate3XS
# import pyrealsense2 as rs
# from sensor_msgs.msg import Image
import einops
import time
import math
from habitat.core.simulator import ActionSpaceConfiguration
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
# from embeddings import BERTProcessor

locobot = InterbotixLocobotCreate3XS(robot_model="locobot_base")

def do_action(action):
    if locobot == None:
        return
    if action == 0:
        print("stop")
        return None
    elif action == 1:
        locobot.base.move(0.25, 0, 1.0)
        # time.sleep(1)
    elif action == 2:
        locobot.base.move(0.25, math.pi / 12.0, 1)
        time.sleep(1)
    elif action == 3:
        locobot.base.move(0.25, -math.pi / 12.0, 1)
        time.sleep(1)
    elif action == 4:
        locobot.camera.tilt(0.8)
        time.sleep(1)
    elif action == 5:
        locobot.camera.tilt(-0.3)
        time.sleep(1)

action = -1
while(action != 0):
    print("\noptions: \n0: End sequence \n1: Go forward \n2: Turn Left \n3: Turn right \n4: Look down \n5: Look up")
    action = int(input())
    if not action in range(6):
        print("Action not in range. Try again")
        continue
    do_action(action)