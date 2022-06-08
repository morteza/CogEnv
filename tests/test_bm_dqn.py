"""Acme DQN agent interacting with CogEnv."""

from absl import app
import acme
from acme import specs
from acme.agents.tf import dqn
from acme.tf import networks
from cog_env.environment import CogEnv


avd_name = 'CogEnv_API_32'
task_name = 'belval_matrices'
num_episodes = 100
num_choices = 8


def main(_):

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
  app.run(main)
