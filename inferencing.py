import gymnasium as gym
import screen_games
from wrapper import DinoWrapper
from stable_baselines3 import PPO

env = gym.make('screen_games/ScreenEnv-v0')
env = DinoWrapper(env, macro='record.json')

model = PPO.load('dino_ppo', env=env)
observation, info = env.reset()

episode = 0
while episode < 3:
    action, _state = model.predict(observation)
    print(action, observation['timestamp'][0])
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        episode += 1
        observation, info = env.reset()

env.close()