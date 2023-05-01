import torch as th

import sys
import random
import os
import matplotlib.pyplot as plt
from gym_unity.envs import UnityToGymWrapper

from animalai_env.animalai.envs.environment import AnimalAIEnvironment


class AnimalAIEnvironmentLoader:

    def __init__(self, random_config=True, config_file_name='', is_server=False):
        self.is_server = is_server
        if config_file_name is not None:
            if is_server:
                self.config_folder = "generated_envs/"
            else:
                self.config_folder = "../generated_envs/"
            self.config_file = (self.config_folder + config_file_name)
        else:
            if is_server:
                self.config_folder = "generated_envs/"
            else:
                self.config_folder = "../generated_envs/"
            config_files = os.listdir(self.config_folder)
            if random_config:
                config_random = random.randint(0, len(config_files))
                self.config_file = (
                        self.config_folder + config_files[config_random]
                )
                print(self.config_file)
            else:
                self.config_file = (self.config_folder + config_file_name)

    def get_animalai_env(self):
        if self.is_server:
            file_name = 'env/AnimalAI'
        else:
            file_name = '../env/AnimalAI'
        if IS_SERVER:
            aai_env = animai.envs.AnimalAIEnvironment(
                seed=123,
                file_name=file_name,
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
        else:
            aai_env = AnimalAIEnvironment(
                seed=123,
                file_name=file_name,
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
