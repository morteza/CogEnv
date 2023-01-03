# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Function for loading CogEnv."""

import os
from os import PathLike

from pathlib import Path

import android_env

from android_env.environment import AndroidEnv
from android_env.components import coordinator as coordinator_lib
from android_env.components import task_manager as task_manager_lib
from android_env.components.simulators.emulator import emulator_simulator
from android_env.proto import task_pb2

from google.protobuf import text_format

from android_env.wrappers.flat_interface_wrapper import FlatInterfaceWrapper
from android_env.wrappers.tap_action_wrapper import TapActionWrapper
from .wrappers import BelvalMatricesWrapper


class CogEnv(AndroidEnv):
  def __init__(self, coordinator: coordinator_lib.Coordinator):
    super().__init__(coordinator)

  @classmethod
  def create(cls,
             task_name: str,
             avd_name: str,
             android_avd_home: PathLike = Path.home() / '.android/avd',
             android_sdk_root: PathLike = Path.home() / 'Android/Sdk',
             emulator_path: PathLike = Path.home() / 'Android/Sdk/emulator/emulator',
             adb_path: PathLike = Path.home() / 'Android/Sdk/platform-tools/adb',
             **kwargs) -> 'CogEnv':
    """Creates a CogEnv instance.

    Args:
      task_name: Path to the task textproto file.
      avd_name: Name of the AVD (Android Virtual Device).
      android_avd_home: Path to the AVD (Android Virtual Device).
      android_sdk_root: Root directory of the SDK.
      emulator_path: Path to the emulator binary.
      adb_path: Path to the ADB (Android Debug Bridge).
      run_headless: If True, the emulator display is turned off.
    Returns:
      env: A CogEnv envrionment.
    """

    run_headless = kwargs.get('run_headless', False)

    task_path = Path('cogenv/proto') / f'{task_name}.textproto'

    # Load the Behaverse task proto.
    task = task_pb2.Task()
    with open(task_path, 'r') as proto_file:
      task = text_format.Parse(proto_file.read(), task)

    # Create AndroidEnv simulator.
    simulator = emulator_simulator.EmulatorSimulator(
        adb_controller_args=dict(
            adb_path=os.path.expanduser(adb_path),
            adb_server_port=5037,
        ),
        verbose_logs=True,
        emulator_launcher_args=dict(
            avd_name=avd_name,
            android_avd_home=os.path.expanduser(android_avd_home),
            android_sdk_root=os.path.expanduser(android_sdk_root),
            emulator_path=os.path.expanduser(emulator_path),
            run_headless=run_headless,
            gpu_mode='swiftshader_indirect',
            # adb_port=5037,
            # emulator_console_port=5554,
            # grpc_port=8554,
        ),
    )

    task_manager = task_manager_lib.TaskManager(task)
    coordinator = coordinator_lib.Coordinator(simulator, task_manager)

    env = CogEnv(coordinator)
    env = BelvalMatricesWrapper(env, num_choices=8)
    env = FlatInterfaceWrapper(env)

    return env
