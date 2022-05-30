from typing import Sequence, Dict

from android_env.components import action_type
from android_env.wrappers import base_wrapper
import dm_env
from dm_env import specs
import numpy as np


class BelvalMatricesActionWrapper(base_wrapper.BaseWrapper):
  """Behaverse Common actions wrapper."""

  def __init__(self, env: dm_env.Environment):
    super().__init__(env)
    self._parent_action_spec = self._env.action_spec()
    self._n_choices = 8
    self._assert_base_env()

  def _assert_base_env(self):
    assert len(self._parent_action_spec) == 2
    assert not self._parent_action_spec['action_type'].shape
    assert self._parent_action_spec['touch_position'].shape == (2,)

  def step(self, action: Dict[str, int]) -> dm_env.TimeStep:
    """Take a step in the base environment."""

    return self._env.step(self._process_action(action))

  def _process_action(self, action: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Transforms action into AndroidEnv action spec."""

    return {
        'action_type':
            np.array(self._get_action_type(action['action_id']),
                     dtype=self._parent_action_spec['action_type'].dtype),
        'touch_position':
            np.array(self._get_touch_position(action['action_id']),
                     dtype=self._parent_action_spec['touch_position'].dtype)
    }

  def _get_action_type(self, action_id: int) -> action_type.ActionType:
    """Compute action type corresponding to the given action_id."""

    if self._redundant_actions:
      assert action_id < self._num_action_types * self._grid_size
      return action_id // self._grid_size

    else:
      assert action_id <= self._grid_size + 1
      if action_id < self._grid_size:
        return action_type.ActionType.TOUCH
      elif action_id == self._grid_size:
        return action_type.ActionType.LIFT
      else:
        return action_type.ActionType.REPEAT

  def _get_touch_position(self, action_id: int) -> Sequence[float]:
    """Compute the position corresponding to the given action_id.

    Args:
      action_id: A Belval Matrices choice action.
    Returns:
      touch_position: The XY coordinate of the action.
    """

    x_pos = # WIDTH
    y_pos = # HEIGHT

    return [x_pos, y_pos]

  def action_spec(self) -> Dict[str, specs.Array]:
    """Action spec of the wrapped environment."""

    return {
        'action_id':
            specs.DiscreteArray(
                num_values=self._n_choices,
                name='action_id')
    }
