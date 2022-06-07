
from typing import Sequence, Dict

from android_env.components import action_type
from android_env.wrappers import base_wrapper
import dm_env
from dm_env import specs
import numpy as np

from android_env.wrappers.discrete_action_wrapper import DiscreteActionWrapper
from android_env.wrappers.image_rescale_wrapper import ImageRescaleWrapper
from android_env.wrappers.float_pixels_wrapper import FloatPixelsWrapper
from android_env.wrappers.flat_interface_wrapper import FlatInterfaceWrapper
from acme import wrappers as acme_wrappers


class BelvalMatricesWrapper(base_wrapper.BaseWrapper):
  """Behaverse BM actions wrapper."""

  def __init__(self, env: dm_env.Environment, num_choices=8):
    env = self.apply_base_wrappers(env)
    super().__init__(env)
    self._parent_action_spec = self._env.action_spec()
    self.num_choices = num_choices
    self._assert_base_env()

  def apply_base_wrappers(self, env):
    """Applies a series of wrappers to the environment."""
    # env = DiscreteActionWrapper(self._env, action_grid=(10, 10))
#    env = ImageRescaleWrapper(env, zoom_factors=(0.25, 0.25))
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
    """Take a step in the base environment."""

    return self._env.step(self._process_action(action))

  def _process_action(self, action: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Transforms BM action into AndroidEnv action."""

    return {
        'action_type':
            np.array(action_type.ActionType.TOUCH,
                     dtype=self._parent_action_spec['action_type'].dtype),
        'touch_position':
            np.array(self._get_touch_position(action['action_id']),
                     dtype=self._parent_action_spec['touch_position'].dtype)
    }

  def _get_touch_position(self, action_id: int) -> Sequence[float]:
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

    return buttons_pos[action_id]

  def action_spec(self) -> Dict[str, specs.Array]:
    """Action spec of the wrapped BM environment."""

    return {
        'action_id': specs.DiscreteArray(num_values=self.num_choices, name='action_id')
    }
