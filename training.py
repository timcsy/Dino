import gymnasium as gym
import screen_games
from wrapper import DinoWrapper
from stable_baselines3 import PPO

env = gym.make('screen_games/ScreenEnv-v0')
env = DinoWrapper(env, macro='record.json')

model = PPO('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=2048, progress_bar=True)
model.save('dino_ppo')