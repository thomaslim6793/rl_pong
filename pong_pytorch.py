import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

class PolicyModel(nn.Module):
    def __init__(self, D=80*80, H=200, seed=12288743):
        super(PolicyModel, self).__init__()
        torch.manual_seed(seed)
        # Define the dimensions
        self.D = D
        self.H = H

        # Define the layers
        self.fc1 = nn.Linear(D, H)
        self.fc2 = nn.Linear(H, 1)

        # Initialization 
        nn.init.normal_(self.fc1.weight, std=0.01 / D**0.5)
        nn.init.normal_(self.fc2.weight, std=0.01 / H**0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
class ReinforcementLearningPong:

    def __init__(self, device, policy_model, env, **kwargs):
        self.device = device
        self.policy_model = policy_model
        self.env = env
        
        self.progress_show_interval = kwargs.get('progress_show_interval', 100)
        self.save_video = kwargs.get('save_video', True)
        self.video_path = kwargs.get('video_path', './videos')
        self.video_save_interval = kwargs.get('video_save_interval', 10)

    def frame_preprocessing(self, observation_frame):
        observation_frame = observation_frame[35:195]
        observation_frame = observation_frame[::2, ::2, 0]
        observation_frame[observation_frame == 144] = 0  # Erase the background (type 1).
        observation_frame[observation_frame == 109] = 0  # Erase the background (type 2).
        observation_frame[observation_frame != 0] = 1  # Set the items (rackets, ball) to 1.
        return observation_frame.astype(float)
    
    # Take the difference of the current and previous frames as input
    # To capture the change in position of the ball and rackets, i.e.
    # the input is the motion of the objects
    def update_input(self, prev_x, cur_x, D):
        if prev_x is not None:
            x = cur_x - prev_x
            x = torch.tensor(x).float().to(self.device)
        else:
            x = torch.zeros(D).to(self.device)
        return x
    
    # e.g. rewards = [1, 0, 0, 1] will become [1, 0.99^2 , 0.99,1]
    def discount_rewards(self, rewards, gamma):
        discounted_rewards = torch.zeros_like(rewards)
        cur_discount_sum = 0
        # From the last reward to the first...
        for t in reversed(range(0, rewards.size(0))):
            # Reset the reward sum if current reward is not zero
            # because in Pong a positive or negative reward indicates the end of match
            # such that the next action is completely independent of the previous ones
            if rewards[t] != 0:
                cur_discount_sum = 0
            cur_discount_sum = rewards[t] + gamma * cur_discount_sum
            discounted_rewards[t] = cur_discount_sum
        return discounted_rewards
    
    # Use REINFORCE to define the loss function which is
    # : -1 * SUM over all t (discounted_reward at time t * log(probability of action at time t))
    # The probability of action at time t is defines as: action_t * action_proba_t + (1 - action_t) * (1 - action_proba_t)
    def loss_REINFORCE(self, actions, action_probas, discounted_rewards):
        log_policy = actions * action_probas + (1 - actions) * (1 - action_probas)
        log_policy = torch.log(log_policy)
        loss = -torch.sum(log_policy * discounted_rewards)
        return loss
    
    def run_reinforcement_learning(self, render=False, **kwargs):
        max_episodes = kwargs.get('max_episodes', 8000)
        grad_accum_interval = kwargs.get('grad_accum_interval', 3)
        learning_rate = kwargs.get('learning_rate', 1e-4)
        decay_rate = kwargs.get('decay_rate', 0.99)
        gamma = kwargs.get('gamma', 0.99)

        optimizer = optim.RMSprop(self.policy_model.parameters(), lr=learning_rate, alpha=decay_rate)
        optimizer.zero_grad()  # Ensure gradients are initialized to zero

        episode_number = 0
        grad_counter = 0  # To keep track of accumulated gradients

        os.makedirs(self.video_path, exist_ok=True)
        env_recorder = VideoRecorder(self.env, path=os.path.join(self.video_path, f'Pong_ep_{episode_number}.mp4'), enabled=True)
        reward_sum = 0
        running_reward = None

        xs, actions, action_probas, rewards = [], [], [], []
        observation = env_recorder.env.reset()[0]
        prev_x = None
        while episode_number <= max_episodes:
            if render:
                env_recorder.env.render()

            # Get the current state input
            cur_x = self.frame_preprocessing(observation).ravel()
            x = self.update_input(prev_x, cur_x, self.policy_model.D)
            xs.append(x)
            prev_x = cur_x
            # Forward pass to get action probability
            action_proba = self.policy_model(x).squeeze()
            # Sample the action given this probability
            action = 2 if torch.rand(1).item() < action_proba else 3
            # Perform the action and get the next state, reward, and termination status
            observation, reward, terminated, truncated, info = env_recorder.env.step(action)

            if self.save_video and episode_number % self.video_save_interval == 0:
                env_recorder.capture_frame()
            reward_sum += reward

            # Accumulate the states, log probabilities, and rewards
            actions.append(1 if action == 2 else 0)
            action_probas.append(action_proba) 
            rewards.append(reward)

            if terminated:
                xs = torch.stack(xs)
                actions = torch.tensor(actions).float().to(device)
                action_probas = torch.stack(action_probas).to(device)
                rewards = torch.tensor(rewards).float().to(device)

                # 1. Get discounted rewards
                discounted_rewards = self.discount_rewards(rewards, gamma)
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

                # 2. Compute loss
                loss = self.loss_REINFORCE(actions, action_probas, discounted_rewards)

                # 3. Accumulate gradients
                loss.backward()  # This will accumulate gradients into the model parameters

                # 4. Perform parameter update after every `grad_accum_interval`
                if grad_counter % grad_accum_interval == 0:
                    optimizer.step()
                    optimizer.zero_grad()  # Reset gradients after update

                # Logic for printing and saving videos
                running_reward = (
                    reward_sum
                    if running_reward is None
                    else running_reward * 0.99 + reward_sum * 0.01
                )
                if episode_number % self.progress_show_interval == 0:
                    print(
                        "Resetting the Pong environment. Episode {} total reward: {} Running mean: {}".format(
                            episode_number, reward_sum, running_reward
                        )
                    )
                # Save video and create a new recorder at the specified interval
                if self.save_video and episode_number % self.video_save_interval == 0:
                    env_recorder.close() # This saves the video
                    if episode_number < max_episodes:
                        env_recorder = VideoRecorder(self.env, path=os.path.join(self.video_path, f'Pong_ep_{episode_number + self.video_save_interval}.mp4'), enabled=True) # this creates a new video
                                # Reset after episode
                reward_sum = 0

                xs, actions, action_probas, rewards = [], [], [], []
                prev_x = None
                observation = env_recorder.env.reset()[0]
                episode_number += 1
                grad_counter += 1  # Update counter to match episodes

        env_recorder.close()  # Get the last video saved

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("Pong-v4", render_mode="rgb_array")
    policy_model = PolicyModel(D=80*80, H=200, seed=12288743)
    policy_model.to(device)

    video_config = {
        'progress_show_interval': 50,
        'save_video': True,
        'video_path': './videos2',
        'video_save_interval': 100
    }

    reinforcement_learning = ReinforcementLearningPong(device, policy_model, env, **video_config)

    hyper_parameters = { 
        'max_episodes': 5000,
        'grad_accum_interval': 5,
        'learning_rate': 1e-3,
        'decay_rate': 0.99,
        'gamma': 0.99
    }
    reinforcement_learning.run_reinforcement_learning(render=False, **hyper_parameters)