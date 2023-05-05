import random
import wandb
import sys
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable

IS_SERVER = True
from baseline_models.logger import Logger
from segment_anything_objects.dqn_sam import DQN
from baseline_models.animalai_loader import AnimalAIEnvironmentLoader
from segment_anything_objects.sam_utils import SegmentAnythingObjectExtractor

envs_to_run = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class TrainModel(object):
    ACTIONS_TO_USE = [1, 2, 3, 6]

    def __init__(self, model, env, memory=(True, 100), writer=None, params={}):
        self.model_to_train = model
        self.env = env
        self.use_memory = memory
        self.memory=None
        self.writer = writer
        self.params = params
        self.object_extractor = SegmentAnythingObjectExtractor()

    def train(self, target_net, policy_net, memory, params, optimizer, writer, max_timesteps=2000):
        episode_durations = []
        num_episodes = 3000
        steps_done = 0
        counter = 0
        smallest_loss = 99999
        loss = 0
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            obs = self.env.reset()
            #state = self.process_frames(obs)
            state = self.object_extractor.extract_objects(obs)
            rew_ep = 0
            loss_ep = 0
            losses = []

            for t in count():
                # Select and perform an action
                action, steps_done = self.select_action(state, params, policy_net, self.env.action_space.n, steps_done)

                screen, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)

                rew_ep += reward.item()
                #current_screen = self.process_frames(screen)
                current_screen = self.object_extractor.extract_objects(screen)
                if not done:
                    next_state = current_screen
                else:
                    next_state = None

                # Store the transition in memory
                memory.push(state, action, reward, next_state, done)

                # Move to the next state
                prev_state = state
                state = next_state

                # Perform one step of the optimization (on the target network)
                if len(memory) > 32:
                    loss_ep = self.compute_td_loss(policy_net, memory, params, optimizer)
                    losses.append(loss_ep.item())
                    self.writer.log(
                        {"Loss/iter": loss_ep, "Episode": counter})
                    counter +=1

                if done or t == max_timesteps:
                    episode_durations.append(t + 1)
                    print(f'---End of episode-- {i_episode}')
                    self.writer.log(
                        {"Reward/episode": rew_ep, "Episode": i_episode})
                    print(rew_ep)


                    loss = np.sum(losses)
                    self.writer.log(
                        {"Loss/episode": loss/(t+1), "Episode": i_episode})

                    break
            if i_episode % 20 == 0 and i_episode != 0 and loss < smallest_loss:
                PATH = f"model_with_obj_{i_episode}.ckp"

                torch.save({
                    'episode': i_episode,
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, PATH)
                smallest_loss = loss

        return

    def compute_td_loss(self, model, replay_buffer, params, optimizer, batch_size=32):

        state, action, reward, next_state, done, non_final_mask = replay_buffer.sample_td(batch_size)
        # state = self.object_extractor.extract_objects(state)
        # next_state = self.object_extractor.extract_objects(next_state)

        state = state.to(device)
        next_state = next_state.to(device)
        action = action.to(device)

        target = action.squeeze(1)
        values = torch.tensor(self.ACTIONS_TO_USE).to(device)
        t_size = target.numel()
        v_size = values.numel()
        t_expand = target.unsqueeze(1).expand(t_size, v_size)
        v_expand = values.unsqueeze(0).expand(t_size, v_size)
        result_actions = (t_expand - v_expand == 0).nonzero()[:, 1].unsqueeze(1)

        reward = reward.to(device)
        done = done.to(device)

        q_values = model(state)
        #q_values = torch.tensor([[self.ACTIONS_TO_USE[model(state).max(1)[1].view(1, 1)]]])
        next_q_values = torch.zeros(batch_size, device=device)
        next_q_values[non_final_mask] = model(next_state).max(1)[0]

        q_value = q_values.gather(1, result_actions).squeeze(1)

        expected_q_value = reward + params['gamma'] * next_q_values * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def init_model(self, actions=4, checkpoint_file=""):
        obs = self.env.reset()
        init_screen = self.object_extractor.extract_objects(obs)
        # init_screen = self.process_frames(obs)
        # _, _, screen_height, screen_width = init_screen.shape
        if actions == 0:
            n_actions = self.env.action_space.n
        else:
            n_actions = actions
        #objects_in_init_screen = self.object_extractor.extract_objects(init_screen.squeeze(0))
        policy_net = self.model_to_train(init_screen.shape, n_actions).to(device)
        optimizer = optim.RMSprop(policy_net.parameters())
        optimizer = optim.Adam(policy_net.parameters(), lr=0.00001)
        if checkpoint_file != "":
            print(f"Trainning from checkpoint {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file)
            policy_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        target_net = self.model_to_train(init_screen.shape, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        if self.use_memory[0] is not None:
            self.memory = ReplayMemory(self.use_memory[1])
            self.train(target_net, policy_net, self.memory, self.params, optimizer, self.writer)
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
            #current_screen = self.process_frames(obs)
            state = self.object_extractor.extract_objects(obs)
            rew_ep = 0
            for t in count():
                action = self.ACTIONS_TO_USE[target_net(state).max(1)[1].view(1, 1)]
                screen, reward, done, _ = self.env.step(action)
                reward = torch.tensor([reward], device=device)
                rew_ep += reward.item()
                #state = self.process_frames(screen)
                state = self.object_extractor.extract_objects(screen)
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

    def sample_td(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
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
        return state_batch, action_batch, reward_batch, non_final_next_states, torch.tensor(batch.done,
                                                                                            dtype=torch.int64), non_final_mask

    def __len__(self):
        return len(self.memory)


def process_frames_a(state):
    resize = T.Compose([T.ToPILImage(),
                        T.Grayscale(),
                        T.ToTensor()])
    screen = torch.tensor(state).T
    return resize(screen).to(device)


def main():
    args = sys.argv[1:]
    checkpoint_file = ""
    if args[0] == '-checkpoint':
        checkpoint_file = args[1]
    params = {
        'batch_size': 10,
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

    wandb_logger = Logger(f"{checkpoint_file}samobjects_dqn_no_target_64image", project='rl_loop')
    logger = wandb_logger.get_logger()
    trainer = TrainModel(DQN,
                         env, (True, 1000),
                         logger, params)
    trainer.init_model(checkpoint_file=checkpoint_file)


    return

if __name__ == '__main__':
    main()