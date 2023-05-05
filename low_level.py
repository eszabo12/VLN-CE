import habitat
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# What is the Embodied AI task?
config = habitat.get_config(BASE_DIR + "/VLN-CE/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml")
# Are we in sim or reality?
# if args.use_simulation: # Use Habitat-Sim
# config.SIMULATOR.TYPE = "Habitat-Sim-v0"
# else: # Use LoCoBot via PyRobot
config.SIMULATOR.TYPE = "PyRobot-Locobot-v0"
# Create environment (sim or real doesn’t matter)
env = habitat.Env(config)
observations = env.reset()
# Which model are we testing?
model = torch.load("/home/elle/Repos/research/VLN-CE/data/checkpoints/rxr_cma_en/ckpt.6.pth")
# Let’s act!
while not env.episode_over:
    action = model(observations)
    observations = env.step(action)