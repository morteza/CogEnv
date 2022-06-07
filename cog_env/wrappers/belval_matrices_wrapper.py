
from typing import Sequence, Dict

from android_env.components import action_type
from android_env.wrappers import base_wrapper
import dm_env
from dm_env import specs
import numpy as np

from android_env.wrappers.float_pixels_wrapper import FloatPixelsWrapper
from acme import wrappers as acme_wrappers


class BelvalMatricesWrapper(base_wrapper.BaseWrapper):
  """Behaverse BM actions wrapper."""

  def __init__(self, env: dm_env.Environment, num_choices=8):
    env = self.apply_base_wrappers(env)
    super().__init__(env)
    self._parent_action_spec = self._env.action_spec()
    self.num_choices = num_choices
    self._assert_base_env()
    self._env_steps = 0

  def stats(self):
    logs = self._env.stats()
    logs.update({'env_steps': self._env_steps})
    return logs

  def apply_base_wrappers(self, env):
    """Applies a series of wrappers to the environment."""
    # env = ImageRescaleWrapper(env, zoom_factors=(0.25, 0.25))
    env = FloatPixelsWrapper(env)
    # env = FlatInterfaceWrapper(env)
    # env = acme_wrappers.SinglePrecisionWrapper(env)
    return env

  def _assert_base_env(self):
    """Verify that the wrapped environment is the default AndroidEnv environment."""

    assert len(self._parent_action_spec) == 2
    assert not self._parent_action_spec['action_type'].shape
    assert self._parent_action_spec['touch_position'].shape == (2,)

  def step(self, action: Dict[str, int]) -> dm_env.TimeStep:
    self._env_steps += 2  # 2 steps per action (touch and lift)
    tap_actions = self._process_action(action)
    total_reward = 0.0
    step_type, discount, observation = dm_env.StepType.MID, 0.0, None

    print(tap_actions)
    for a in tap_actions:
      step_type, reward, discount, observation = self._env.step(a)
      if reward:
        total_reward += reward
      if step_type == dm_env.StepType.LAST:
        break

    return dm_env.TimeStep(
        step_type=step_type,
        reward=total_reward,
        discount=discount,
        observation=observation)

  def _process_action(self, action: Dict[str, int]) -> Sequence[Dict[str, np.ndarray]]:
    """Transforms BM action into AndroidEnv action."""

    touch = {
        'action_type':
            np.array(action_type.ActionType.TOUCH,
                     dtype=self._parent_action_spec['action_type'].dtype),
        'touch_position':
            np.array(self._get_touch_position(action['action_id']),
                     dtype=self._parent_action_spec['touch_position'].dtype)
    }

    tap = [touch]

    lift = touch.copy()
    lift['action_type'] = np.array(
        action_type.ActionType.LIFT,
        dtype=self._parent_action_spec['action_type'].dtype)

    tap.append(lift)

    return tap

  def _get_touch_position(self, action_id: int, orientation='landscape') -> Sequence[float]:
    """Compute the position of BM touches

    Args:
      action_id: A Belval Matrices choice action.
    Returns:
      touch_position: The XY coordinate of the action.
    """

    # center coords of the BM choices (in 320x240)
    # x=100,140,180,220
    # y=170,210

    buttons_pos = [(100, 170), (140, 170), (180, 170), (220, 170),
                   (100, 210), (140, 210), (180, 210), (220, 210)]
    x, y = buttons_pos[action_id]

    x_noise, y_noise = np.random.uniform(-5, 5, size=2)

    # FIXME should be taken from the screen info
    width = 320.
    height = 240.

    x = (x + x_noise) / width
    y = (y + y_noise) / height

    if orientation == 'landscape':
      return 1.0 - y, x
    else:
      return x, y

  def action_spec(self) -> Dict[str, specs.Array]:
    """Action spec of the wrapped BM environment."""

    return {
        'action_id': specs.DiscreteArray(num_values=self.num_choices, name='action_id')
    }
