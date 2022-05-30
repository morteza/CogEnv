from typing import Dict
from absl import app
from absl import logging

from pathlib import Path

from android_env import loader
from dm_env import specs
import numpy as np


# params
avd_name = 'Pixel_3a_XL_API_31'
home_dir = Path.home()
tasks_root = '/Users/morteza/workspace/cog_env/proto'
task_path = f'{tasks_root}/chrome.textproto'
n_steps = 1000


def main(_):

  with loader.load(avd_name=avd_name,
                   task_path=task_path,
                   android_avd_home=home_dir / '.android/avd',
                   android_sdk_root=home_dir / 'Android/Sdk',
                   emulator_path=home_dir / 'Android/Sdk/emulator/emulator',
                   adb_path=home_dir / 'Android/Sdk/platform-tools/adb',
                   run_headless=False) as env:

    action_spec = env.action_spec()

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
      logging.info(f'>>>>>>>>>>> {env.task_extras()}')


if __name__ == '__main__':
  app.run(main)
