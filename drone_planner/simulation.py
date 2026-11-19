from __future__ import annotations

import time
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .contracts import DronePlan


HOME = np.array([0.18, 0.0, 0.4], dtype=float)


@dataclass
class SimulationResult:
    success: bool
    reached_waypoints: int
    total_waypoints: int
    max_error: float
    final_error: float
    terminated_early: bool
    trajectory: list[list[float]]
    errors: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class TD3Simulator:
    """Runs only one selected drone plan against the repository's existing TD3 policy."""

    def __init__(self, policy_dir: Path = Path("sources/TD3_resources/pytorch_models"), policy_name: str = "TD3_NewAviary_0", gui: bool = False, max_steps_per_waypoint: int = 720, threshold: float = 0.05):
        self.policy_dir = policy_dir
        self.policy_name = policy_name
        self.gui = gui
        self.max_steps_per_waypoint = max_steps_per_waypoint
        self.threshold = threshold

    def run(self, plan: DronePlan) -> SimulationResult:
        try:
            vendor_root = Path(__file__).resolve().parents[1] / "gym-pybullet-drones-0.5.2"
            if str(vendor_root) not in sys.path:
                sys.path.insert(0, str(vendor_root))
            from sources.TD3_resources.NewAviary import NewAviary
            from sources.TD3_resources.TD3 import TD3
        except ImportError as error:
            raise RuntimeError("单机仿真依赖未安装；请先按 README 安装 pybullet、gym 和 torch。") from error

        env = NewAviary(freq=240, gui=self.gui, record=False, box_bound=[0.6, 0.6, 0.6], hover=False)
        points = [np.array(point, dtype=float) for point in plan.waypoints]
        if plan.return_home:
            points.append(HOME.copy())
        env.EPISODE_LEN_SEC = len(points) * self.max_steps_per_waypoint / env.SIM_FREQ + 2
        env.INIT_XYZS[0] = HOME.copy()
        observation = env.reset()
        policy = TD3(env.observation_space.shape[0], env.action_space.shape[0], float(env.action_space.high[0]))
        policy.load(self.policy_name, str(self.policy_dir))

        trajectory: list[list[float]] = [HOME.tolist()]
        reached, max_error, errors = 0, 0.0, []
        terminated_early = False
        try:
            for waypoint_index, target in enumerate(points, start=1):
                env.desired_pos = target
                reached_target = False
                for _ in range(self.max_steps_per_waypoint):
                    action = policy.select_action(np.asarray(observation))
                    observation, _, done, _ = env.step(action)
                    position = np.asarray(env._getDroneStateVector(0)[0:3], dtype=float)
                    trajectory.append(position.tolist())
                    error = float(np.linalg.norm(position - target))
                    max_error = max(max_error, error)
                    if error <= self.threshold:
                        reached_target = True
                        reached += 1
                        break
                    if bool(np.asarray(done).any()):
                        terminated_early = True
                        errors.append(f"第 {waypoint_index} 个航点前环境提前结束。")
                        break
                if not reached_target:
                    if not terminated_early:
                        errors.append(f"第 {waypoint_index} 个航点未在时限内达到 {self.threshold}m 阈值。")
                    break
                if terminated_early:
                    break
        finally:
            final_position = np.asarray(env._getDroneStateVector(0)[0:3], dtype=float)
            env.close()
        final_error = float(np.linalg.norm(final_position - points[-1]))
        return SimulationResult(reached == len(points) and not terminated_early, reached, len(points), max_error, final_error, terminated_early, trajectory, errors)


class PIDSimulator:
    """Single-drone PyBullet baseline using the vendored DSL PID controller."""

    def __init__(self, gui: bool = False, max_steps_per_waypoint: int = 720, control_freq: int = 48, threshold: float = 0.05):
        self.gui = gui
        self.max_steps_per_waypoint = max_steps_per_waypoint
        self.control_freq = control_freq
        self.threshold = threshold

    def run(self, plan: DronePlan) -> SimulationResult:
        try:
            vendor_root = Path(__file__).resolve().parents[1] / "gym-pybullet-drones-0.5.2"
            if str(vendor_root) not in sys.path:
                sys.path.insert(0, str(vendor_root))
            from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
            from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
        except ImportError as error:
            raise RuntimeError("PID 单机仿真依赖未安装；请先按 README 安装 pybullet、gym、scipy 与 torch。") from error

        env = CtrlAviary(num_drones=1, initial_xyzs=np.array([HOME]), freq=240, gui=self.gui, record=False, obstacles=False, user_debug_gui=False)
        points = [np.array(point, dtype=float) for point in plan.waypoints]
        if plan.return_home:
            points.append(HOME.copy())
        controller = DSLPIDControl(env)
        control_every = max(1, int(env.SIM_FREQ / self.control_freq))
        action = {"0": np.zeros(4)}
        observation = env.reset()
        trajectory: list[list[float]] = [HOME.tolist()]
        reached, max_error, errors = 0, 0.0, []
        terminated_early = False
        try:
            for waypoint_index, target in enumerate(points, start=1):
                reached_target = False
                for step in range(self.max_steps_per_waypoint):
                    if step % control_every == 0:
                        action["0"], _, _ = controller.computeControlFromState(
                            control_timestep=control_every * env.TIMESTEP,
                            state=observation["0"]["state"],
                            target_pos=target,
                        )
                    observation, _, _, _ = env.step(action)
                    position = np.asarray(observation["0"]["state"][0:3], dtype=float)
                    trajectory.append(position.tolist())
                    error = float(np.linalg.norm(position - target))
                    max_error = max(max_error, error)
                    if error <= self.threshold:
                        reached_target = True
                        reached += 1
                        break
                    if position[0] < -0.2 or position[0] > 0.2 or position[1] < -0.2 or position[1] > 0.2 or position[2] < 0.2 or position[2] > 0.5:
                        terminated_early = True
                        errors.append(f"第 {waypoint_index} 个航点前实际轨迹越出执行空间。")
                        break
                if not reached_target:
                    if not terminated_early:
                        errors.append(f"第 {waypoint_index} 个航点未在时限内达到 {self.threshold}m 阈值。")
                    break
        finally:
            final_position = np.asarray(observation["0"]["state"][0:3], dtype=float)
            env.close()
        final_error = float(np.linalg.norm(final_position - points[-1]))
        return SimulationResult(reached == len(points) and not terminated_early, reached, len(points), max_error, final_error, terminated_early, trajectory, errors)
