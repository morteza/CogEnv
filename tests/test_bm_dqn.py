"""Acme DQN agent interacting with AndroidEnv."""

from absl import app
import acme
from acme import specs
from acme.agents.tf import dqn
from acme.tf import networks
from android_env import loader

from pathlib import Path

from android_env.wrappers.flat_interface_wrapper import FlatInterfaceWrapper
from cog_env.wrappers import BelvalMatricesWrapper


avd_name = 'CogEnv_API_32'
home_dir = Path.home()
task_path = 'cog_env/proto/belval_matrices.textproto'
android_avd_home = Path.home() / '.android/avd'
android_sdk_root = Path.home() / 'Android/Sdk'
emulator_path = Path.home() / 'Android/Sdk/emulator/emulator'
adb_path = Path.home() / 'Android/Sdk/platform-tools/adb'
n_steps = 1000
num_episodes = 100
num_choices = 8


def main(_):

  with loader.load(avd_name=avd_name,
                   task_path=task_path,
                   android_avd_home=str(android_avd_home),
                   android_sdk_root=str(android_sdk_root),
                   emulator_path=str(emulator_path),
                   adb_path=str(adb_path),
                   run_headless=False) as env:

    env = BelvalMatricesWrapper(env, num_choices=num_choices)
    env = FlatInterfaceWrapper(env)
    env_spec = specs.make_environment_spec(env)

    agent = dqn.DQN(
        environment_spec=env_spec,
        network=networks.DQNAtariNetwork(num_actions=num_choices),
        batch_size=10,
        samples_per_insert=2,
        min_replay_size=10)

    loop = acme.EnvironmentLoop(env, agent)
    loop.run(num_episodes=num_episodes)


if __name__ == '__main__':
  app.run(main)
