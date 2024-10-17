from tqdm import tqdm
import numpy as np
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import os

def frame_preprocessing(observation_frame):
    observation_frame = observation_frame[35:195]
    if len(observation_frame.shape) == 3:
        observation_frame = observation_frame[::2, ::2, 0]  # 3D array
    else:
        observation_frame = observation_frame[::2, ::2]  # 2D array
    observation_frame[observation_frame == 144] = 0  # Erase the background (type 1).
    observation_frame[observation_frame == 109] = 0  # Erase the background (type 2).
    observation_frame[observation_frame != 0] = 1  # Set the items (rackets, ball) to 1.
    return observation_frame.astype(float)

class PolicyModel:

    def __init__(self, D=80*80, H=200, seed=12288743):
        self.rng = np.random.default_rng(seed)
        self.D = D
        self.H = H
        self.layers = {'W1': self.rng.standard_normal(size=(H, D)) / np.sqrt(D), 
                       'W2': self.rng.standard_normal(size=H) / np.sqrt(H)}

    def policy_forward(self, x):
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))
        h = np.dot(self.layers['W1'], x)
        h[h < 0] = 0
        logit = np.dot(self.layers['W2'], h)
        p = sigmoid(logit)
        return p, h
    
    def policy_backward(self, eph, epdlogp, epx):
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.layers['W2'])
        dh[eph <= 0] = 0
        dW1 = np.dot(dh.T, epx)  # Update for W1 using input states (epx)
        return {'W1': dW1, 'W2': dW2}
    
class ReinforcementLearningPong:

    def __init__(self, policy_model, env, **kwargs):
        self.policy_model = policy_model
        self.env = env
        
        self.progress_show_interval = kwargs.get('progress_show_interval', 100)
        self.save_video = kwargs.get('save_video', True)
        self.video_path = kwargs.get('video_path', './videos')
        self.video_save_interval = kwargs.get('video_save_interval', 10)

    def discount_rewards(self, r, gamma):
        discounted_r = np.zeros_like(r)
        running_add = 0
        # From the last reward to the first...
        for t in reversed(range(0, r.size)):
            # ...reset the reward sum
            if r[t] != 0:
                running_add = 0
            # ...compute the discounted reward
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def update_input(self, prev_x, cur_x, D):
        if prev_x is not None:
            x = cur_x - prev_x
        else:
            x = np.zeros(D)
        return x

    def run_reinforcement_learning(self, render=False, **kwargs):

        max_episodes = kwargs.get('max_episodes', 8000)
        grad_accum_interval = kwargs.get('grad_accum_interval', 3)
        learning_rate = kwargs.get('learning_rate', 1e-4)
        decay_rate = kwargs.get('decay_rate', 0.99)
        gamma = kwargs.get('gamma', 0.99)

        xs = []
        hs = []
        dlogps = []
        drs = []

        grad_buffer = {k: np.zeros_like(v) for k, v in self.policy_model.layers.items()}
        rmsprop_cache = {k: np.zeros_like(v) for k, v in self.policy_model.layers.items()}

        episode_number = 0

        os.makedirs(self.video_path, exist_ok=True)
        env_recorder = VideoRecorder(self.env, path=os.path.join(self.video_path, f'Pong_ep_{episode_number}.mp4'), enabled=True) # this creates the video recorder for the 0th episode

        observation = env_recorder.env.reset()[0]
        prev_x = None
        reward_sum = 0
        running_reward = None
        while episode_number <= max_episodes:
            if render:
                env_recorder.env.render()

            cur_x = frame_preprocessing(observation).ravel()
            x = self.update_input(prev_x, cur_x, self.policy_model.D)
            prev_x = cur_x

            aprob, h = self.policy_model.policy_forward(x)
            
            action = 2 if self.policy_model.rng.uniform() < aprob else 3

            xs.append(x)
            hs.append(h)

            y = 1 if action == 2 else 0
            dlogps.append(y - aprob)
            observation, reward, terminated, truncated, info = env_recorder.env.step(action)

            if self.save_video and episode_number % self.video_save_interval == 0:
                env_recorder.capture_frame()

            reward_sum += reward
            drs.append(reward)

            if terminated:
                epx = np.vstack(xs)
                eph = np.vstack(hs)
                epdlogp = np.vstack(dlogps)
                epr = np.vstack(drs)

                xs = []
                hs = []
                dlogps = []
                drs = []

                discounted_epr = self.discount_rewards(epr, gamma)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)

                epdlogp *= discounted_epr
                grad = self.policy_model.policy_backward(eph, epdlogp, epx)

                for k in self.policy_model.layers.keys():
                    grad_buffer[k] += grad[k]

                if episode_number % grad_accum_interval == 0:
                    for k, v in self.policy_model.layers.items():
                        g = grad_buffer[k]
                        rmsprop_cache[k] = (
                            decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                        )
                        self.policy_model.layers[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                        grad_buffer[k] = np.zeros_like(v)

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

                reward_sum = 0
                observation = env_recorder.env.reset()[0]
                prev_x = None

                # Every 10 episodes, save the video.
                if self.save_video and episode_number % self.video_save_interval == 0:
                    env_recorder.close() # This saves the video
                    if episode_number < max_episodes:
                        env_recorder = VideoRecorder(self.env, path=os.path.join(self.video_path, f'Pong_ep_{episode_number + self.video_save_interval}.mp4'), enabled=True) # this creates a new video
                episode_number += 1
            # if reward != 0:
            #     print(
            #         "Episode {}: Game finished. Reward: {}...".format(episode_number, reward)
            #         + ("" if reward == -1 else " POSITIVE REWARD!")
            #     )

        env_recorder.close() # Get the last video saved

if __name__ == "__main__":

    env = gym.make("Pong-v4", render_mode="rgb_array")
    policy_model = PolicyModel(D=80*80, H=200, seed=12288743)

    video_config = {
        'progress_show_interval': 100,
        'save_video': True,
        'video_path': '/content/drive/MyDrive/pong_rl/videos',
        'video_save_interval': 1000
    }

    reinforcement_learning = ReinforcementLearningPong(policy_model, env, **video_config)

    hyper_parameters = { 
        'max_episodes': 10000,
        'grad_accum_interval': 5,
        'learning_rate': 1e-3,
        'decay_rate': 0.99,
        'gamma': 0.99
    }
    reinforcement_learning.run_reinforcement_learning(render=False, **hyper_parameters)