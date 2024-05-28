import gymnasium as gym
import screen_games
from wrapper import DinoWrapper

env = gym.make('screen_games/ScreenEnv-v0')
env = DinoWrapper(env, macro='record.json')

observation, info = env.reset()

episode = 0
while episode < 3:
    action = 0
    events = list(map(lambda x: x['event'], info['inputs']))
    if 'pw' in events:
        action = 1
    elif 'ps' in events:
        action = 2
    print(action, observation['timestamp'][0])
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        episode += 1
        observation, info = env.reset()

env.close()