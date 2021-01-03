import numpy as np
from baselines.common.vec_env import VecEnvObservationWrapper
from gym.spaces.box import Box

COLORS = {
    'maze': [
        [128,128,128],
        [0,0,255],
    ],
    'heist': [
        [128,128,128],
        [0,0,128],
        [0,0,255],
        [0,255,0],
    ],
    'chaser': [
        [128,128,128],
        [0,255,0],
        [0,0,255],
    ],
    'bigfish': [
        [0,255,0],
        [0,0,255],
    ],
    'dodgeball': [
        [255,0,0],
        [128,128,128],
        [0,0,255],
        [0,0,32],
        [0,0,64],
        [0,0,80],
        [0,0,112],
    ],
    'starpilot': [
        [255,0,0],
        [0,0,255],
        [0,0,128],
    ],
    
}

def num_channels(env_name):
    return len(COLORS[env_name])


def onehot(lbl, env_name):
    # return lbl
    onehot_lbl = np.zeros(lbl.shape[:3]+(num_channels(env_name),), np.uint8)
    for i, color in enumerate(COLORS[env_name]):
        onehot_lbl[...,i] = (color==lbl)[...,0]
    return onehot_lbl


class VecExtractDictObsOnehot(VecEnvObservationWrapper):
    """ Hack to onehot encoder the label input """
    def __init__(self, venv, env_name):
        
        self.venv = venv
        self.env_name = env_name
        
        obs_space = Box(0,1,shape=(64,64,num_channels(env_name)))
        super().__init__(venv, observation_space=obs_space)
        
    def process(self, obs):
        return onehot(obs['lbl'], self.env_name)
