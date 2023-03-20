import torch as th

import sys
import random
import os
import matplotlib.pyplot as plt
from gym_unity.envs import UnityToGymWrapper
from animalai.envs.environment import AnimalAIEnvironment


class AnimalAIEnvironmentLoader:

    def __init__(self, config_file=None, random_config=True, config_file_name=''):
        if config_file is not None:
            self.config_file = config_file
        else:
            self.config_folder = "../generated_envs/"
            config_files = os.listdir(self.config_folder)
            if random_config:
                config_random = random.randint(0, len(config_files))
                self.config_file = (
                        self.config_folder + config_files[config_random]
                )
            else:
                self.config_file = (self.config_folder + config_file_name)

    def get_animalai_env(self):
        aai_env = AnimalAIEnvironment(
            seed=123,
            file_name="../env/AnimalAI",
            arenas_configurations=self.config_file,
            play=False,
            base_port=5000,
            inference=False,
            useCamera=True,
            resolution=256,
            useRayCasts=False,
            # raysPerSide=1,
            # rayMaxDegrees = 30,
        )
        env = UnityToGymWrapper(aai_env, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True)
        return env
