import random
import wandb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
import torch.nn.functional as F
import torchvision.transforms as T
IS_SERVER = True
if not IS_SERVER:
    from baseline_models.logger import Logger
    from baseline_models.conv_dqn import DQN
    from baseline_models.animalai_loader import AnimalAIEnvironmentLoader
else:
    from logger import Logger
    from conv_dqn import DQN
    from animalai_loader import AnimalAIEnvironmentLoader


envs_to_run = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class TrainModel(object):
    ACTIONS_TO_USE = [1, 2, 3, 6]

    def __init__(self, model, env, memory=(True, 1000), writer=None, params={}):
        self.model_to_train = model
        self.env = env
        self.use_memory = memory
        self.memory=None
        self.writer = writer
        self.params = params

    def run_train(self, target_net, policy_net, memory, params, optimizer, writer, max_timesteps=10000):
        episode_durations = []
        num_episodes = 10000
        steps_done = 0
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            obs = self.env.reset()
            state = self.process_frames(obs)
            current_screen = state
            rew_ep = 0
            loss_ep = 0
            timestep = 0
            episode_frames = []
            for t in count():
                # Select and perform an action
                timestep += 1
                action, steps_done = self.select_action(state, params, policy_net, self.env.action_space.n, steps_done)

                returned_state, reward, done, _ = self.env.step(action.item())
                self.writer.log({"Action taken": action.item()})
                reward = torch.tensor([reward], device=device)

                rew_ep += reward.item()
                # Observe new state
                last_screen = current_screen
                current_screen = self.process_frames(returned_state)
                # if t % 2000:
                #     episode_frames.append(wandb.Image(returned_state))
                if not done:
                    next_state = current_screen
                else:
                    next_state = None

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                loss_ep = self.optimize_model(policy_net, target_net, params, memory, optimizer, loss_ep, writer)

                if done or t == max_timesteps:
                    print(t)
                    episode_durations.append(t + 1)
                    self.writer.log({"Reward episode": rew_ep, "Episode duration": t + 1, "Train loss": loss_ep / (t + 1)})
                    print(loss_ep / (t + 1))
                    # episode_frames_wandb = make_grid(episode_frames)
                    # images = wandb.Image(episode_frames_wandb, caption=f'Episode {i_episode} states')
                    #self.writer.log({'states': episode_frames})

                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % params['target_update'] == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if i_episode % 1000 == 0 and i_episode != 0:
                self.evaluate(target_net, writer, i_episode)
        return

    def init_model(self, actions=4):
        obs = self.env.reset()
        init_screen = self.process_frames(obs)
        _, _, screen_height, screen_width = init_screen.shape
        if actions == 0:
            n_actions = self.env.action_space.n
        else:
            n_actions = actions

        policy_net = self.model_to_train(init_screen.squeeze(0).shape, n_actions).to(device)
        target_net = self.model_to_train(init_screen.squeeze(0).shape, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.RMSprop(policy_net.parameters())
        if self.use_memory[0] is not None:
            self.memory = ReplayMemory(self.use_memory[1])
            self.run_train(target_net, policy_net, self.memory, self.params, optimizer, self.writer)
        return

    def process_frames(self,obs):
        resize = T.Compose([T.ToPILImage(),
                            T.Resize(128, interpolation=Image.CUBIC),
                            T.ToTensor()])
        screen = obs.transpose((2, 0, 1))
        _, screen_height, screen_width = screen.shape
        screen = torch.tensor(screen)
        return resize(screen).unsqueeze(0).to(device)

    def select_action(self, state, params, policy_net, n_actions, steps_done):
        sample = random.random()
        eps_threshold = 0.8
        # eps_threshold = params['eps_end'] + (params['eps_start'] - params['eps_end']) * \
        #     math.exp(-1. * steps_done / params['eps_decay'])
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                #Exploiting
                return torch.tensor([[self.ACTIONS_TO_USE[policy_net(state).max(1)[1].view(1, 1)]]], device=device, dtype=torch.long), steps_done
        else:
            return torch.tensor([[random.choice(self.ACTIONS_TO_USE)]], device=device, dtype=torch.long), steps_done

    def optimize_model(self, policy_net, target_net, params, memory, optimizer, loss_ep, writer):
        if len(memory) < params['batch_size']:
            return loss_ep
        transitions = memory.sample(params['batch_size'])
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # Map action batch to 4 actions
        action_batch_new = torch.tensor([self.ACTIONS_TO_USE.index(a.item()) for a in action_batch], dtype=torch.int64, device=device)
        action_batch_new = action_batch_new.view(action_batch_new.shape[0], 1)
        state_action_values = policy_net(state_batch).gather(1, action_batch_new)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(params['batch_size'], device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(device)
        writer.log({'Batch reward': reward_batch.sum().output_nr})
        expected_state_action_values = (next_state_values * params['gamma']) + reward_batch

        # Compute Huber loss
        #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        loss_ep = loss_ep + loss.item()

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        return loss_ep

    def evaluate(self, target_net, writer, i_episode, max_timesteps=1000):
        with torch.no_grad():
            # Initialize the environment and state
            obs = self.env.reset()
            #last_screen = self.process_frames()
            current_screen = self.process_frames(obs)
            state = current_screen
            rew_ep = 0
            for t in count():
                action = self.ACTIONS_TO_USE[target_net(state).max(1)[1].view(1, 1)]
                screen, reward, done, _ = self.env.step(action)
                reward = torch.tensor([reward], device=device)
                rew_ep += reward.item()
                state = self.process_frames(screen)
                if done or t==max_timesteps:
                    writer.add_scalar('Reward ep test', rew_ep, i_episode)
                    writer.log({"Reward episode test": rew_ep})
                    break
        return


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def process_frames_a(state):
    resize = T.Compose([T.ToPILImage(),
                        T.Grayscale(),
                        T.ToTensor()])
    screen = torch.tensor(state).T
    return resize(screen).to(device)


def main():
    params = {
        'batch_size': 64,
        'gamma': 0.99,
        'eps_start': 0.9,
        'eps_end':0.02,
        'eps_decay': .999985,
        'target_update': 1000
    }

    env_loader = AnimalAIEnvironmentLoader(
        random_config=False,
        config_file_name="config_multiple_209.yml",
        is_server=IS_SERVER)
    env = env_loader.get_animalai_env()

    wandb_logger = Logger("baseline_dqn", project='rl_loop')
    logger = wandb_logger.get_logger()
    trainer = TrainModel(DQN,
                         env, (True, 1000),
                         logger, params)
    trainer.init_model()


    # run_lin_dqn(env, params, logger)
    # #run_conv_dqn(env, params, writer)

    return

if __name__ == '__main__':
    main()