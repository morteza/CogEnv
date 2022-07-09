"""Acme DQN agent interacting with CogEnv."""

from time import sleep
from absl import app
import acme
from acme import specs
from acme.agents.tf import dqn
from acme.tf import networks
from cogenv.environment import CogEnv

import matplotlib.pyplot as plt

avd_name = 'CogEnv_API_32'
task_name = 'belval_matrices'
num_episodes = 10
num_choices = 8


def manual_loop(_):

  with CogEnv.create(task_name, avd_name, headless=False) as env:

    env_spec = specs.make_environment_spec(env)

    agent = dqn.DQN(
        environment_spec=env_spec,
        network=networks.DQNAtariNetwork(num_actions=num_choices),
        batch_size=10,
        samples_per_insert=2,
        n_step=1,
        min_replay_size=10)

    for episode_idx in range(num_episodes):
      timestep = env.reset()
      agent.observe_first(timestep)

      timestep_idx = 0

      # Run an episode.
      while not timestep.last():
        action = agent.select_action(timestep.observation)
        timestep = env.step(action)

        agent.observe(action, next_timestep=timestep)

        plt.imshow(timestep.observation)
        plt.savefig(f'tmp/images/e{episode_idx}s{timestep_idx}.png')

        timestep_idx += 1
        sleep(1.)


def acme_loop(_):

  with CogEnv.create(task_name, avd_name, headless=False) as env:

    env_spec = specs.make_environment_spec(env)

    agent = dqn.DQN(
        environment_spec=env_spec,
        network=networks.DQNAtariNetwork(num_actions=num_choices),
        batch_size=10,
        samples_per_insert=2,
        n_step=1,
        min_replay_size=10)

    loop = acme.EnvironmentLoop(env, agent)
    loop.run(num_episodes=num_episodes)


if __name__ == '__main__':
  app.run(manual_loop)
