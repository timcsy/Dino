from gymnasium import Wrapper
from gymnasium import spaces
import time
import numpy as np
from eventstreaming import stream

class DinoWrapper(Wrapper):
    def __init__(self, env, macro=None):
        super().__init__(env)
        self.env = env

        # observation_space: {
        #   'screen': (36, 144, 3),
        #   'timestamp': [float32]
        # }

        # We have 3 actions, corresponding to 0: "none", 1: "up", 2: "down"
        # self.action_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)

        self.macro = macro
        self.count = 0
    
    def reset(self, *, seed=None, options=None):
        def on_reset():
            # Focus on the game
            if self.macro is not None:
                stream.play_io_event(records=self.macro)
            else:
                state = 'start'
                while True:
                    input = stream.get_io_event()
                    while input is None:
                        time.sleep(0.001)
                        input = stream.get_io_event()
                    if state =='start' and input['event'] == 'p ctrl':
                        state = 'p ctrl'
                    elif state =='p ctrl' and input['event'] == 'pr':
                        state = 'end'
                        break

        observation, info = self.env.reset(seed=seed, options=options, on_reset=on_reset)

        return observation, info

    def step(self, action):
        if action == 0 or self.count > 3:
            pass
        elif action == 1:
            stream.send_io_event('p up')
            time.sleep(0.2)
            stream.send_io_event('r up')
        # elif action == 2:
        #     stream.send_io_event('p down')
        #     time.sleep(0.5)
        #     stream.send_io_event('r down')
        
        observation, _, _, truncated, info = self.env.step(action)

        terminated = False
        screen = observation['screen']
        mean = np.mean(screen[18:25, 68:75, 0]) # Look at the center
        if mean > 200:
            self.count = 0
        elif mean < 180:
            self.count += 1
        if self.count > 50:
            self.count = 0
            terminated = True

        reward = -1.0 if terminated else 0.01

        return observation, reward, terminated, truncated, info