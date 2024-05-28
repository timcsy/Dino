import gymnasium as gym
import screen_games
from wrapper import DinoWrapper
from stable_baselines3 import PPO

from stable_baselines3.common.preprocessing import preprocess_obs
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image

import os
import sys
import time
from PIL import Image

def explain(cam, preprocessed_obs, img):
    grayscale_cam = cam(input_tensor=preprocessed_obs)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return visualization

env = gym.make('screen_games/ScreenEnv-v0')
env = DinoWrapper(env, macro='record.json')

model = PPO.load('dino_ppo', env=env)

layer = 3
if len(sys.argv) > 1:
    layer = int(sys.argv[1])
if not os.path.exists('cam'):
    os.makedirs('cam')

target_layers = [model.policy.features_extractor.extractors.screen.cnn[layer]]
cam = GradCAMPlusPlus(model=model.policy.features_extractor.extractors.screen, target_layers=target_layers)

observation, info = env.reset()

episode = 0
while episode < 3:
    action, _state = model.predict(observation)

    obs = model.policy.obs_to_tensor(observation)[0]
    preprocessed_obs = preprocess_obs(obs, model.policy.observation_space, normalize_images=model.policy.normalize_images)
    cam_img = explain(cam, preprocessed_obs['screen'], observation['screen'] / 255)
    img = Image.fromarray(cam_img)
    img.save(f'cam/{int(time.time() * 1000)}.jpg')

    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        episode += 1
        observation, info = env.reset()

env.close()