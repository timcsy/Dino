# Dino with Reinforcement Learning

## Setup

Install Python 3 packages:
```
pip install requirements.txt
```

Edit your `.env` file, for example:
```
SCREENENV_LEFT=0
SCREENENV_TOP=290
SCREENENV_WIDTH=1920
SCREENENV_HEIGHT=480
SCREENENV_NEW_WIDTH=144
SCREENENV_NEW_HEIGHT=36
SCREENENV_FULLSCREEN=False
```

Run `record.py`, to record the keyboard and mouse actions you want to do on reset:
```
python record.py
```

Press Alt + R to stop recording.

## Ussage

### Ussage - Basic

To open the environment, you can do the following lines:
```=Python
import gymnasium as gym
import screen_games
from wrapper import DinoWrapper

env = gym.make('screen_games/ScreenEnv-v0')
env = DinoWrapper(env, macro='record.json')
```

### Ussage - Random

To run the game randomly, run `dino.py`:
```
python dino.py
```

### Ussage - Manual

To run the game manually, run `manual.py`:
```
python manual.py
```

### Ussage - Training

To train the reinforcement learning AI, run `training.py`:
```
python training.py
```
Remember to change the model name inside the code.

To continue train the reinforcement learning AI, run `training_cont.py`:
```
python training_cont.py
```
Remember to change the model name inside the code.

### Ussage - Inferencing

To inference by the reinforcement learning AI, run `inferencing.py`:
```
python inferencing.py
```
Remember to change the model name inside the code.

### Ussage - XAI (eXplaining AI)

To explaning the reinforcement learning AI by CAM (a kind of XAI), run `cam.py`:
```
python cam.py <layer_num>

For example (For different Neural Network Layer, say <layer_num> is 3 or 4):

python cam.py 3

python cam.py 4

Default layer is 3:

python cam.py
```
Remember to change the model name inside the code.