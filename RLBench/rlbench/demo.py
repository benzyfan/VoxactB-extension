import numpy as np


class Demo(object):

    def __init__(self, observations, random_seed=None):
        self._observations = observations
        self.random_seed = random_seed
        self.variation_number = 0


        print(f"DEBUG: Demo created with {len(observations)} observations")
        for i, obs in enumerate(observations):
            if i < 3: 
                print(f"DEBUG: Obs {i} attrs: {dir(obs)}")
                print(f"DEBUG: Obs {i} target_object_pos: {getattr(obs, 'target_object_pos', 'NOT_FOUND')}")

    def __len__(self):
        return len(self._observations)

    def __getitem__(self, i):
        return self._observations[i]

    def restore_state(self):
        np.random.set_state(self.random_seed)
