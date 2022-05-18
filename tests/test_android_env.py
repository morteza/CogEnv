from typing import Dict
from absl import app
from absl import logging

from android_env import loader
from dm_env import specs
import numpy as np


# params
avd_name = 'Pixel_4_API_32'
tasks_root = '/home/morteza/Desktop'
task_path = f'{tasks_root}/chrome.textproto'
n_steps = 1000


def main(_):

  with loader.load(
      avd_name=avd_name,
      task_path=task_path,
      android_avd_home='/home/morteza/.android/avd',
      android_sdk_root='/home/morteza/Android/Sdk',
      emulator_path='/home/morteza/Android/Sdk/emulator/emulator',
      adb_path='/home/morteza/Android/Sdk/platform-tools/adb',
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
      logging.info(f'Step {step}, action: {action}, reward: {timestep.reward}')


if __name__ == '__main__':
  app.run(main)
