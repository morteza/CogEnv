from typing import Dict
from absl import app
from absl import logging

import time

from pathlib import Path

from android_env import loader
from dm_env import specs
import numpy as np


# params
avd_name = 'CogEnv_API_32'
home_dir = Path.home()
n_steps = 1000


# task_path = 'cog_env/proto/web.textproto'
task_path = 'cog_env/proto/belval_matrices.textproto'
android_avd_home = Path.home() / '.android/avd'
android_sdk_root = Path.home() / 'Android/Sdk'
emulator_path = Path.home() / 'Android/Sdk/emulator/emulator'
adb_path = Path.home() / 'Android/Sdk/platform-tools/adb'


def main(_):

  with loader.load(avd_name=avd_name,
                   task_path=task_path,
                   android_avd_home=str(android_avd_home),
                   android_sdk_root=str(android_sdk_root),
                   emulator_path=str(emulator_path),
                   adb_path=str(adb_path),
                   run_headless=False) as env:

    action_spec = env.action_spec()

    # env._coordinator._simulator.send_touch([(20,20,True,1), (20,20,False,1)])
    
    def get_random_action() -> Dict[str, np.ndarray]:
      """Returns a random AndroidEnv action."""
      action = {}
      for k, v in action_spec.items():
        if isinstance(v, specs.DiscreteArray):
          action[k] = np.random.randint(low=0, high=v.num_values, dtype=v.dtype)
        else:
          action[k] = np.random.random(size=v.shape).astype(v.dtype)
      return action

    env.reset()

    for step in range(n_steps):
      action = get_random_action()
      timestep = env.step(action=action)
      # logging.info(f'Step {step}, action: {action}, reward: {timestep.reward}')
      # logging.info(f'Extras: {env.task_extras()}')
      time.sleep(2)

if __name__ == '__main__':
  app.run(main)
