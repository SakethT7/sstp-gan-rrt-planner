#!/usr/bin/env python3
"""
Strict sample-conditioned LQR-CBF basic-RRT expert dataset generator.

This script intentionally prioritizes expert-quality data over runtime.
It generates random 2D circular-obstacle environments, solves each with an
sample-conditioned LQR-CBF basic-RRT planner for a linear control-affine double-integrator robot, and
saves GAN-ready training arrays:

  train_X: parent state + exact sampled target XY + absolute obstacles
  train_y: accepted LQR-CBF child state
  validation_X / validation_y: environment-disjoint validation rows
  testing_X / testing_y: environment-disjoint held-out rows

State:   x = [px, py, vx, vy]
Control: u = [ax, ay]
Dynamics: discrete exact double integrator

CBF safety verification:
  obstacles use relative-degree-2 HOCBF constraints:
    h = ||p-c||^2 - R^2
    hdot = 2(p-c)^T v
    hddot = 2 v^T v + 2(p-c)^T u
    hddot + k1*hdot + k0*h >= 0
  boundaries use equivalent 1D HOCBF constraints.

The strict default planner uses:
  - sample first
  - nearest existing parent second
  - LQR local steering
  - exact 2D active-set HOCBF control projection
  - conservative continuous swept-edge rejection
  - one accepted child per sample

RRT* rewiring and CEM are deliberately disabled.  Those operations would make
the exported label disagree with the basic-RRT runtime contract used by the
target-conditioned GAN.

Author: generated for WSL research prototyping.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.linalg import solve_discrete_are
    from scipy.ndimage import label as connected_component_labels
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script needs scipy. Install with: pip install numpy scipy"
    ) from exc


Array = np.ndarray
SCRIPT_VERSION = "7.3.0-strict-one-step-sample-parent-child-cbf-qp"
CONDITION_SCHEMA = (
    "[parent_x,parent_y,parent_vx,parent_vy,sample_x,sample_y,"
    "absolute_obstacle_x_y_r...]"
)


@dataclass
class Obstacle:
    cx: float
    cy: float
    r: float


@dataclass
class PlannerConfig:
    width: float = 50.0
    height: float = 50.0
    robot_radius: float = 0.55
    safety_margin: float = 0.25
    min_obstacles: int = 6
    max_obstacles: int = 18
    min_obstacle_radius: float = 1.2
    max_obstacle_radius: float = 3.5
    obstacle_clearance: float = 0.7
    dt: float = 0.12
    lqr_horizon_steps: int = 75
    lqr_goal_tolerance: float = 1.35
    lqr_velocity_tolerance: float = 1.25
    step_len: float = 7.0
    max_iterations: int = 4500
    goal_sample_rate: float = 0.10
    max_speed: float = 6.0
    max_accel: float = 5.0
    q_pos: float = 7.5
    q_vel: float = 1.2
    r_accel: float = 0.24
    cbf_k0: float = 2.0
    cbf_k1: float = 2.8
    min_start_goal_dist: float = 30.0
    max_env_generation_tries: int = 3000
    max_problem_generation_tries: int = 2000
    connectivity_grid_resolution: float = 1.0
    keep_optimizing_after_first_solution: bool = True
    log_every: int = 250
    basic_rrt_only: bool = True
    supervision_lookahead_steps: int = 1
    max_accepted_expansions: int = 200


@dataclass
class DatasetConfig:
    num_envs: int = 300
    paths_per_env: int = 1
    n_nearest_obstacles: int = 8
    lookahead_steps: int = 1
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    seed: int = 7
    output_dir: str = "V7_RRT_SAMPLE_DATASET"
    save_expansion_logs: bool = True
    strict_nonempty: bool = True
    max_rows_per_problem: int = 200


class Node:
    __slots__ = (
        "state",
        "parent",
        "cost",
        "edge_states",
        "edge_controls",
        "edge_clearance",
        "edge_cbf_min",
        "children",
    )

    def __init__(self, state: Array):
        self.state = np.asarray(state, dtype=float).reshape(4)
        self.parent: Optional[int] = None
        self.cost: float = 0.0
        self.edge_states: Optional[Array] = None
        self.edge_controls: Optional[Array] = None
        self.edge_clearance: float = float("inf")
        self.edge_cbf_min: float = float("inf")
        self.children: set[int] = set()


class LQRCBFBasicRRT:
    def __init__(
        self,
        start: Array,
        goal: Array,
        obstacles: List[Obstacle],
        cfg: PlannerConfig,
        rng: np.random.Generator,
    ):
        self.start = np.asarray(start, dtype=float).reshape(4)
        self.goal = np.asarray(goal, dtype=float).reshape(4)
        self.obstacles = obstacles
        self.cfg = cfg
        self.rng = rng
        self.nodes: List[Node] = [Node(self.start)]
        self.best_goal_index: Optional[int] = None
        self.best_goal_cost: float = float("inf")
        self.expansion_logs: List[Dict] = []
        self.accepted_expansions: List[Dict] = []
        self.solution_indices: List[int] = []

        dt = cfg.dt
        self.A = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.B = np.array(
            [
                [0.5 * dt * dt, 0.0],
                [0.0, 0.5 * dt * dt],
                [dt, 0.0],
                [0.0, dt],
            ],
            dtype=float,
        )
        Q = np.diag([cfg.q_pos, cfg.q_pos, cfg.q_vel, cfg.q_vel])
        R = np.diag([cfg.r_accel, cfg.r_accel])
        P = solve_discrete_are(self.A, self.B, Q, R)
        self.K = np.linalg.solve(R + self.B.T @ P @ self.B, self.B.T @ P @ self.A)

    def point_in_bounds(self, p: Array) -> bool:
        rr = self.cfg.robot_radius + self.cfg.safety_margin
        return rr <= p[0] <= self.cfg.width - rr and rr <= p[1] <= self.cfg.height - rr

    def clearance(self, state: Array) -> float:
        p = state[:2]
        rr = self.cfg.robot_radius + self.cfg.safety_margin
        c = min(p[0] - rr, self.cfg.width - rr - p[0], p[1] - rr, self.cfg.height - rr - p[1])
        for obs in self.obstacles:
            d = np.linalg.norm(p - np.array([obs.cx, obs.cy])) - (obs.r + rr)
            if d < c:
                c = d
        return float(c)

    def hocbf_min_residual(self, state: Array, control: Array) -> float:
        p = state[:2]
        v = state[2:]
        u = control
        rr = self.cfg.robot_radius + self.cfg.safety_margin
        k0, k1 = self.cfg.cbf_k0, self.cfg.cbf_k1
        residuals: List[float] = []
        for obs in self.obstacles:
            c = np.array([obs.cx, obs.cy], dtype=float)
            R = obs.r + rr
            rel = p - c
            h = float(rel @ rel - R * R)
            hdot = float(2.0 * rel @ v)
            hddot = float(2.0 * v @ v + 2.0 * rel @ u)
            residuals.append(hddot + k1 * hdot + k0 * h)
        # Relative-degree-2 boundary CBFs.
        bounds = [
            (p[0] - rr, v[0], u[0]),
            (self.cfg.width - rr - p[0], -v[0], -u[0]),
            (p[1] - rr, v[1], u[1]),
            (self.cfg.height - rr - p[1], -v[1], -u[1]),
        ]
        for h, hdot, hddot in bounds:
            residuals.append(float(hddot + k1 * hdot + k0 * h))
        return float(min(residuals)) if residuals else float("inf")

    def cbf_control_halfspaces(self, state: Array) -> Tuple[Array, Array]:
        """Return A, b for all constraints A @ u >= b."""
        p = np.asarray(state, dtype=float)[:2]
        v = np.asarray(state, dtype=float)[2:]
        rr = self.cfg.robot_radius + self.cfg.safety_margin
        k0, k1 = self.cfg.cbf_k0, self.cfg.cbf_k1
        rows: List[Array] = []
        rhs: List[float] = []
        for obs in self.obstacles:
            center = np.asarray([obs.cx, obs.cy], dtype=float)
            rel = p - center
            radius = obs.r + rr
            h = float(rel @ rel - radius * radius)
            hdot = float(2.0 * rel @ v)
            # 2 rel^T u + 2 v^T v + k1 hdot + k0 h >= 0.
            rows.append(2.0 * rel)
            rhs.append(float(-2.0 * (v @ v) - k1 * hdot - k0 * h))

        # Relative-degree-2 workspace-boundary constraints.
        rows.extend(
            [
                np.asarray([1.0, 0.0]),
                np.asarray([-1.0, 0.0]),
                np.asarray([0.0, 1.0]),
                np.asarray([0.0, -1.0]),
            ]
        )
        rhs.extend(
            [
                float(-k1 * v[0] - k0 * (p[0] - rr)),
                float(k1 * v[0] - k0 * (self.cfg.width - rr - p[0])),
                float(-k1 * v[1] - k0 * (p[1] - rr)),
                float(k1 * v[1] - k0 * (self.cfg.height - rr - p[1])),
            ]
        )

        # Componentwise acceleration box, also expressed as halfspaces.
        amax = float(self.cfg.max_accel)
        rows.extend(
            [
                np.asarray([1.0, 0.0]),
                np.asarray([-1.0, 0.0]),
                np.asarray([0.0, 1.0]),
                np.asarray([0.0, -1.0]),
            ]
        )
        rhs.extend([-amax, -amax, -amax, -amax])
        return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)

    def project_control_to_cbf(
        self, state: Array, nominal_control: Array
    ) -> Optional[Array]:
        """Exact Euclidean projection for a two-control convex CBF QP.

        The optimizer is the closest feasible acceleration to nominal LQR:
        minimize ||u-u_nom||^2 subject to all linear HOCBF and box constraints.
        In two dimensions the optimum is either u_nom, a projection onto one
        active boundary, or an intersection of two active boundaries, so finite
        active-set enumeration is exact and needs no external QP package.
        """
        u_nom = np.asarray(nominal_control, dtype=float).reshape(2)
        A, b = self.cbf_control_halfspaces(state)
        tolerance = 1e-8

        def feasible(candidate: Array) -> bool:
            return bool(np.all(A @ candidate >= b - tolerance))

        clipped = np.clip(u_nom, -self.cfg.max_accel, self.cfg.max_accel)
        if feasible(clipped):
            return clipped

        best: Optional[Array] = None
        best_distance = float("inf")

        def consider(candidate: Array) -> None:
            nonlocal best, best_distance
            if not np.isfinite(candidate).all() or not feasible(candidate):
                return
            distance = float(np.sum((candidate - u_nom) ** 2))
            if distance < best_distance:
                best = candidate.copy()
                best_distance = distance

        for i in range(len(A)):
            ai = A[i]
            denominator = float(ai @ ai)
            if denominator <= 1e-14:
                continue
            candidate = u_nom + (b[i] - float(ai @ u_nom)) * ai / denominator
            consider(candidate)

        # A polygon vertex can be closer than a feasible projection onto some
        # other single face, so compare every two-face intersection as well.
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                matrix = np.stack([A[i], A[j]], axis=0)
                determinant = float(np.linalg.det(matrix))
                if abs(determinant) <= 1e-12:
                    continue
                consider(np.linalg.solve(matrix, np.asarray([b[i], b[j]])))
        return best

    def state_safe(self, state: Array) -> bool:
        if not self.point_in_bounds(state[:2]):
            return False
        return self.clearance(state) >= 0.0

    def swept_step_clearance(
        self, state: Array, next_state: Array, control: Array
    ) -> float:
        """Conservative continuous-position clearance over one dynamics step.

        Under constant acceleration, the exact position curve differs from the
        straight chord joining its endpoints by at most ``||u|| dt^2 / 8``.
        Inflating every obstacle and the workspace boundary by that amount
        therefore certifies the entire continuous step, not only saved states.
        """
        a = np.asarray(state, dtype=float)[:2]
        b = np.asarray(next_state, dtype=float)[:2]
        u = np.clip(
            np.asarray(control, dtype=float),
            -self.cfg.max_accel,
            self.cfg.max_accel,
        )
        curve_bound = float(np.linalg.norm(u) * self.cfg.dt * self.cfg.dt / 8.0)
        rr = self.cfg.robot_radius + self.cfg.safety_margin

        # A line segment lies inside the convex box whenever both endpoints do.
        # The curve-bound subtraction protects the possible quadratic bow away
        # from that chord.
        boundary_clearance = min(
            a[0] - rr,
            b[0] - rr,
            self.cfg.width - rr - a[0],
            self.cfg.width - rr - b[0],
            a[1] - rr,
            b[1] - rr,
            self.cfg.height - rr - a[1],
            self.cfg.height - rr - b[1],
        ) - curve_bound
        clearance = float(boundary_clearance)
        for obs in self.obstacles:
            center = np.asarray([obs.cx, obs.cy], dtype=float)
            chord = b - a
            denominator = float(chord @ chord)
            if denominator <= 1e-16:
                chord_distance = float(np.linalg.norm(center - a))
            else:
                t = float(np.clip(((center - a) @ chord) / denominator, 0.0, 1.0))
                chord_distance = float(np.linalg.norm(center - (a + t * chord)))
            obstacle_clearance = chord_distance - (obs.r + rr + curve_bound)
            clearance = min(clearance, float(obstacle_clearance))
        return clearance

    def simulate_step(self, state: Array, control: Array) -> Array:
        u = np.clip(control, -self.cfg.max_accel, self.cfg.max_accel)
        nxt = self.A @ state + self.B @ u
        speed = float(np.linalg.norm(nxt[2:]))
        if speed > self.cfg.max_speed:
            nxt[2:] *= self.cfg.max_speed / max(speed, 1e-12)
        return nxt

    def edge_valid(self, states: Array, controls: Array) -> Tuple[bool, float, float]:
        min_clearance = float("inf")
        min_cbf = float("inf")
        for i in range(len(controls)):
            s = states[i]
            u = controls[i]
            cl = self.swept_step_clearance(s, states[i + 1], u)
            cbf = self.hocbf_min_residual(s, u)
            min_clearance = min(min_clearance, cl)
            min_cbf = min(min_cbf, cbf)
            if cl < 0.0 or cbf < -1e-6:
                return False, min_clearance, min_cbf
        final_clearance = self.clearance(states[-1])
        min_clearance = min(min_clearance, final_clearance)
        if final_clearance < 0.0:
            return False, min_clearance, min_cbf
        return True, min_clearance, min_cbf

    def steer(self, from_state: Array, target_state: Array, exact: bool = False) -> Optional[Tuple[Array, Array, float, float, float]]:
        from_state = np.asarray(from_state, dtype=float).reshape(4)
        target_state = np.asarray(target_state, dtype=float).reshape(4)
        target = target_state.copy()
        if not exact:
            vec = target[:2] - from_state[:2]
            dist = float(np.linalg.norm(vec))
            if dist > self.cfg.step_len:
                target[:2] = from_state[:2] + vec / dist * self.cfg.step_len
                target[2:] = 0.0
        x = from_state.copy()
        states = [x.copy()]
        controls = []
        # The expert action and the inserted RRT edge are the same fixed local
        # horizon. Do not reject a safe child because a hypothetical, uninserted
        # continuation farther along a 75-step rollout would later collide.
        local_horizon = min(
            self.cfg.lqr_horizon_steps,
            self.cfg.supervision_lookahead_steps,
        )
        for _ in range(local_horizon):
            err = x - target
            nominal_u = -self.K @ err
            u = self.project_control_to_cbf(x, nominal_u)
            if u is None:
                return None
            x = self.simulate_step(x, u)
            controls.append(u.copy())
            states.append(x.copy())
            pos_err = float(np.linalg.norm(x[:2] - target[:2]))
            vel_err = float(np.linalg.norm(x[2:] - target[2:]))
            if pos_err <= self.cfg.lqr_goal_tolerance and vel_err <= self.cfg.lqr_velocity_tolerance:
                break
        states_arr = np.asarray(states, dtype=float)
        controls_arr = np.asarray(controls, dtype=float)
        if len(controls_arr) == 0:
            return None
        ok, min_clearance, min_cbf = self.edge_valid(states_arr, controls_arr)
        if not ok:
            return None
        cost = self.edge_cost(states_arr, controls_arr)
        return states_arr, controls_arr, cost, min_clearance, min_cbf

    def edge_cost(self, states: Array, controls: Array) -> float:
        dp = np.diff(states[:, :2], axis=0)
        length_cost = float(np.sum(np.linalg.norm(dp, axis=1)))
        control_cost = float(0.03 * np.sum(np.linalg.norm(controls, axis=1) ** 2) * self.cfg.dt)
        speed_cost = float(0.01 * np.sum(np.linalg.norm(states[:, 2:], axis=1)) * self.cfg.dt)
        clearance_vals = np.array([max(self.clearance(s), 1e-3) for s in states], dtype=float)
        clearance_cost = float(0.02 * np.sum(1.0 / clearance_vals) * self.cfg.dt)
        return length_cost + control_cost + speed_cost + clearance_cost

    def sample_state(self) -> Array:
        # Goal bias.
        if self.rng.random() < self.cfg.goal_sample_rate:
            return self.goal.copy()
        rr = self.cfg.robot_radius + self.cfg.safety_margin
        for _ in range(200):
            p = np.array([
                self.rng.uniform(rr, self.cfg.width - rr),
                self.rng.uniform(rr, self.cfg.height - rr),
            ])
            state = np.array([p[0], p[1], 0.0, 0.0], dtype=float)
            if self.state_safe(state):
                return state
        return self.goal.copy()

    def nearest_index(self, sample: Array) -> int:
        pts = np.array([n.state[:2] for n in self.nodes])
        d = np.linalg.norm(pts - sample[:2], axis=1)
        return int(np.argmin(d))

    def planning(self) -> Optional[int]:
        if not self.cfg.basic_rrt_only:
            raise RuntimeError(
                "This strict dataset generator only supports basic RRT. "
                "RRT* parent replacement would break the sample/nearest-parent label contract."
            )
        if self.cfg.supervision_lookahead_steps < 1:
            raise ValueError("supervision_lookahead_steps must be at least 1")
        t0 = time.time()
        for it in range(self.cfg.max_iterations):
            # Runtime contract: choose exactly one sample before inspecting the
            # nearest parent.  The same sample is exported in the training row.
            sample = self.sample_state()
            nearest_idx = self.nearest_index(sample)
            nearest = self.nodes[nearest_idx]
            result = self.steer(nearest.state, sample, exact=False)
            if result is None:
                if self.cfg.log_every and it % self.cfg.log_every == 0:
                    self.expansion_logs.append({"iter": it, "status": "rejected_lqr_cbf", "tree_size": len(self.nodes)})
                continue
            states, controls, _, _, _ = result

            # Add the same fixed-horizon substep that the GAN is asked to learn.
            # This avoids labeling an intermediate state while inserting a
            # different, farther endpoint into the expert tree.
            child_step = min(self.cfg.supervision_lookahead_steps, len(states) - 1)
            edge_states = states[: child_step + 1].copy()
            edge_controls = controls[:child_step].copy()
            valid, clearance, cbf_min = self.edge_valid(edge_states, edge_controls)
            if not valid:
                # Defensive only: a prefix of an accepted edge should be safe.
                raise RuntimeError("Accepted LQR-CBF edge has an invalid supervision prefix")
            edge_cost = self.edge_cost(edge_states, edge_controls)
            node = Node(edge_states[-1])
            node.parent = nearest_idx
            node.cost = nearest.cost + edge_cost
            node.edge_states = edge_states
            node.edge_controls = edge_controls
            node.edge_clearance = clearance
            node.edge_cbf_min = cbf_min
            self.nodes.append(node)
            new_idx = len(self.nodes) - 1
            nearest.children.add(new_idx)

            # Store immutable copies at the insertion boundary.  Every exported
            # row therefore has one exact sample, its exact nearest parent, and
            # the exact accepted child produced from that pair.
            self.accepted_expansions.append({
                "iteration": int(it),
                "sample_state": sample.copy(),
                "parent_state": nearest.state.copy(),
                "parent_index": int(nearest_idx),
                "child_state": node.state.copy(),
                "child_index": int(new_idx),
                "first_control": edge_controls[0].copy(),
                "edge_clearance": float(clearance),
                "edge_cbf_min": float(cbf_min),
                "sampled_goal": bool(np.array_equal(sample, self.goal)),
            })

            # Strict basic RRT terminates in the goal region; it never inserts
            # an extra unsampled goal-connector node.
            if np.linalg.norm(node.state[:2] - self.goal[:2]) <= self.cfg.lqr_goal_tolerance * 1.5:
                self.solution_indices.append(new_idx)
                if node.cost < self.best_goal_cost:
                    self.best_goal_cost = node.cost
                    self.best_goal_index = new_idx
            if self.cfg.log_every and it % self.cfg.log_every == 0:
                self.expansion_logs.append({
                    "iter": it,
                    "status": "accepted",
                    "tree_size": len(self.nodes),
                    "sample": sample.tolist(),
                    "nearest_parent_index": int(nearest_idx),
                    "accepted_child_index": int(new_idx),
                    "best_goal_cost": self.best_goal_cost if math.isfinite(self.best_goal_cost) else None,
                    "elapsed_sec": time.time() - t0,
                })
            if self.best_goal_index is not None and not self.cfg.keep_optimizing_after_first_solution:
                break
            if (
                self.best_goal_index is not None
                and self.cfg.max_accepted_expansions > 0
                and len(self.accepted_expansions) >= self.cfg.max_accepted_expansions
            ):
                break
        return self.best_goal_index

    def extract_full_trajectory(self, goal_idx: int) -> Tuple[Array, Array]:
        chain: List[int] = []
        idx: Optional[int] = goal_idx
        while idx is not None:
            chain.append(idx)
            idx = self.nodes[idx].parent
        chain = chain[::-1]
        state_segments: List[Array] = []
        control_segments: List[Array] = []
        for j, node_idx in enumerate(chain):
            node = self.nodes[node_idx]
            if j == 0:
                state_segments.append(node.state.reshape(1, 4))
            else:
                # Drop duplicate first state of each edge.
                if node.edge_states is not None and len(node.edge_states) > 1:
                    state_segments.append(node.edge_states[1:])
                if node.edge_controls is not None:
                    control_segments.append(node.edge_controls)
        states = np.vstack(state_segments) if state_segments else np.zeros((0, 4), dtype=float)
        controls = np.vstack(control_segments) if control_segments else np.zeros((0, 2), dtype=float)
        return states, controls


def distance_point_to_obstacles(p: Array, obstacles: Sequence[Obstacle], robot_radius: float, safety_margin: float) -> float:
    rr = robot_radius + safety_margin
    if not obstacles:
        return float("inf")
    return float(min(np.linalg.norm(p - np.array([o.cx, o.cy])) - (o.r + rr) for o in obstacles))


def sample_obstacles(rng: np.random.Generator, cfg: PlannerConfig) -> List[Obstacle]:
    for _ in range(cfg.max_env_generation_tries):
        obstacles: List[Obstacle] = []
        n_obs = int(rng.integers(cfg.min_obstacles, cfg.max_obstacles + 1))
        tries = 0
        while len(obstacles) < n_obs and tries < n_obs * 400:
            tries += 1
            r = float(rng.uniform(cfg.min_obstacle_radius, cfg.max_obstacle_radius))
            cx = float(rng.uniform(r + cfg.robot_radius + 1.0, cfg.width - r - cfg.robot_radius - 1.0))
            cy = float(rng.uniform(r + cfg.robot_radius + 1.0, cfg.height - r - cfg.robot_radius - 1.0))
            ok = True
            for o in obstacles:
                if np.linalg.norm(np.array([cx - o.cx, cy - o.cy])) < (r + o.r + cfg.obstacle_clearance):
                    ok = False
                    break
            if ok:
                obstacles.append(Obstacle(cx, cy, r))
        if len(obstacles) >= cfg.min_obstacles:
            return obstacles
    raise RuntimeError("Could not generate non-overlapping obstacles. Reduce obstacle count/radii.")


def sample_safe_state(rng: np.random.Generator, obstacles: Sequence[Obstacle], cfg: PlannerConfig) -> Array:
    rr = cfg.robot_radius + cfg.safety_margin
    for _ in range(cfg.max_problem_generation_tries):
        p = np.array([
            rng.uniform(rr + 1.0, cfg.width - rr - 1.0),
            rng.uniform(rr + 1.0, cfg.height - rr - 1.0),
        ])
        if distance_point_to_obstacles(p, obstacles, cfg.robot_radius, cfg.safety_margin) > 1.0:
            return np.array([p[0], p[1], 0.0, 0.0], dtype=float)
    raise RuntimeError("Could not sample safe start/goal state.")


class FreeSpaceConnectivity:
    """Conservative grid components in robot configuration space."""

    def __init__(self, obstacles: Sequence[Obstacle], cfg: PlannerConfig):
        resolution = float(cfg.connectivity_grid_resolution)
        if resolution <= 0.0:
            raise ValueError("connectivity_grid_resolution must be positive")
        rr = cfg.robot_radius + cfg.safety_margin
        nx = max(3, int(math.ceil((cfg.width - 2.0 * rr) / resolution)) + 1)
        ny = max(3, int(math.ceil((cfg.height - 2.0 * rr) / resolution)) + 1)
        self.xs = np.linspace(rr, cfg.width - rr, nx, dtype=float)
        self.ys = np.linspace(rr, cfg.height - rr, ny, dtype=float)
        xx, yy = np.meshgrid(self.xs, self.ys, indexing="ij")
        free = np.ones((nx, ny), dtype=bool)
        for obs in obstacles:
            radius = obs.r + rr
            free &= (xx - obs.cx) ** 2 + (yy - obs.cy) ** 2 > radius * radius
        # Four-neighbor connectivity is conservative: diagonal corner cutting
        # cannot make two components appear connected.
        structure = np.asarray(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8
        )
        self.labels, self.component_count = connected_component_labels(
            free, structure=structure
        )
        self.free_indices = np.argwhere(free)
        self.resolution = resolution

    def component_at(self, state: Array) -> int:
        if len(self.free_indices) == 0:
            return 0
        p = np.asarray(state, dtype=float)[:2]
        ix = int(np.argmin(np.abs(self.xs - p[0])))
        iy = int(np.argmin(np.abs(self.ys - p[1])))
        label = int(self.labels[ix, iy])
        if label != 0:
            return label
        # A safe continuous point can fall beside an occupied grid center. Use
        # the nearest free center only within a small, explicit grid tolerance.
        coords = np.column_stack(
            [self.xs[self.free_indices[:, 0]], self.ys[self.free_indices[:, 1]]]
        )
        distance = np.linalg.norm(coords - p[None, :], axis=1)
        nearest = int(np.argmin(distance))
        if distance[nearest] > 1.5 * self.resolution:
            return 0
        cell = self.free_indices[nearest]
        return int(self.labels[int(cell[0]), int(cell[1])])


def sample_problem(
    rng: np.random.Generator,
    obstacles: Sequence[Obstacle],
    cfg: PlannerConfig,
    connectivity: Optional[FreeSpaceConnectivity] = None,
) -> Tuple[Array, Array]:
    connectivity = connectivity or FreeSpaceConnectivity(obstacles, cfg)
    for _ in range(cfg.max_problem_generation_tries):
        start = sample_safe_state(rng, obstacles, cfg)
        goal = sample_safe_state(rng, obstacles, cfg)
        if (
            np.linalg.norm(goal[:2] - start[:2]) >= cfg.min_start_goal_dist
            and connectivity.component_at(start) != 0
            and connectivity.component_at(start) == connectivity.component_at(goal)
        ):
            return start, goal
    raise RuntimeError(
        "Could not sample a separated start/goal pair in the same conservative "
        "configuration-space component."
    )


def encode_state(state: Array, cfg: PlannerConfig) -> Array:
    return np.array(
        [
            state[0] / cfg.width,
            state[1] / cfg.height,
            state[2] / cfg.max_speed,
            state[3] / cfg.max_speed,
        ],
        dtype=np.float32,
    )


def encode_sample_xy(sample: Array, cfg: PlannerConfig) -> Array:
    sample = np.asarray(sample, dtype=float).reshape(4)
    return np.asarray(
        [sample[0] / cfg.width, sample[1] / cfg.height], dtype=np.float32
    )


def absolute_obstacle_features(
    current: Array,
    obstacles: Sequence[Obstacle],
    cfg: PlannerConfig,
    n: int,
) -> Array:
    """Encode absolute normalized obstacle geometry in nearest-first order.

    The old pipeline encoded center offsets and later decoded them as absolute
    centers.  This function makes the representation explicit and reversible.
    """
    p = np.asarray(current, dtype=float)[:2]
    ordered = sorted(
        obstacles,
        key=lambda o: (
            float(np.linalg.norm(p - np.asarray([o.cx, o.cy], dtype=float)) - o.r),
            float(o.cx),
            float(o.cy),
            float(o.r),
        ),
    )
    scale = max(cfg.width, cfg.height)
    rows = [
        [o.cx / cfg.width, o.cy / cfg.height, o.r / scale]
        for o in ordered[:n]
    ]
    while len(rows) < n:
        rows.append([0.0, 0.0, 0.0])
    return np.asarray(rows, dtype=np.float32).reshape(-1)


def make_strict_rrt_condition(
    parent: Array,
    sample: Array,
    obstacles: Sequence[Obstacle],
    pcfg: PlannerConfig,
    dcfg: DatasetConfig,
) -> Array:
    return np.concatenate(
        [
            encode_state(parent, pcfg),
            encode_sample_xy(sample, pcfg),
            absolute_obstacle_features(
                parent, obstacles, pcfg, dcfg.n_nearest_obstacles
            ),
        ],
        axis=0,
    ).astype(np.float32)


def expansions_to_training_rows(
    planner: LQRCBFBasicRRT,
    goal: Array,
    obstacles: Sequence[Obstacle],
    pcfg: PlannerConfig,
    dcfg: DatasetConfig,
    env_id: int,
    problem_id: int,
) -> Dict[str, List]:
    """Convert accepted basic-RRT expansions into exact supervised tuples."""
    records = planner.accepted_expansions
    if dcfg.max_rows_per_problem > 0 and len(records) > dcfg.max_rows_per_problem:
        # Deterministic coverage over the full run avoids retaining only the
        # earliest, easiest part of a long tree expansion.
        selected = np.linspace(
            0, len(records) - 1, num=dcfg.max_rows_per_problem, dtype=np.int64
        )
        records = [records[int(i)] for i in selected]

    out: Dict[str, List] = {
        "G_X": [],
        "G_y": [],
        "G_y_raw": [],
        "G_u": [],
        "G_env_id": [],
        "G_problem_id": [],
        "G_iteration": [],
        "G_parent_index": [],
        "G_child_index": [],
        "G_sample_xy": [],
        "G_sample_state_raw": [],
        "G_goal_state": [],
        "G_sampled_goal": [],
        "G_clearance": [],
        "G_cbf_min": [],
        "raw_current_states": [],
        "raw_goal_states": [],
        "absolute_obstacles_per_row": [],
    }
    encoded_goal = encode_state(goal, pcfg)
    scenario_id = int(env_id) * 1_000_000 + int(problem_id)
    for rec in records:
        parent = np.asarray(rec["parent_state"], dtype=float)
        sample = np.asarray(rec["sample_state"], dtype=float)
        child = np.asarray(rec["child_state"], dtype=float)
        cond = make_strict_rrt_condition(
            parent, sample, obstacles, pcfg, dcfg
        )
        child_enc = encode_state(child, pcfg)
        sample_xy = encode_sample_xy(sample, pcfg)
        if not np.array_equal(cond[4:6], sample_xy):
            raise RuntimeError("Conditioned sample does not match recorded RRT sample")
        if not np.isfinite(cond).all() or not np.isfinite(child_enc).all():
            raise RuntimeError("Non-finite value in strict RRT supervision row")
        out["G_X"].append(cond)
        out["G_y"].append(child_enc)
        out["G_y_raw"].append(child.astype(np.float32))
        out["G_u"].append(np.asarray(rec["first_control"], dtype=np.float32))
        out["G_env_id"].append(int(env_id))
        out["G_problem_id"].append(scenario_id)
        out["G_iteration"].append(int(rec["iteration"]))
        out["G_parent_index"].append(int(rec["parent_index"]))
        out["G_child_index"].append(int(rec["child_index"]))
        out["G_sample_xy"].append(sample_xy)
        out["G_sample_state_raw"].append(sample.astype(np.float32))
        out["G_goal_state"].append(encoded_goal)
        out["G_sampled_goal"].append(bool(rec["sampled_goal"]))
        out["G_clearance"].append(float(rec["edge_clearance"]))
        out["G_cbf_min"].append(float(rec["edge_cbf_min"]))
        out["raw_current_states"].append(parent.astype(np.float32))
        out["raw_goal_states"].append(goal.astype(np.float32))
        out["absolute_obstacles_per_row"].append(
            absolute_obstacle_features(
                parent, obstacles, pcfg, dcfg.n_nearest_obstacles
            ).reshape(dcfg.n_nearest_obstacles, 3)
        )
    return out


def extend_rows(dst: Dict[str, List], src: Dict[str, List]) -> None:
    for k, v in src.items():
        dst.setdefault(k, []).extend(v)


def stack_or_empty(rows: List, shape_tail: Tuple[int, ...], dtype=np.float32) -> Array:
    if not rows:
        return np.zeros((0, *shape_tail), dtype=dtype)
    return np.asarray(rows, dtype=dtype)


def create_split_indices(env_ids: Array, dcfg: DatasetConfig, rng: np.random.Generator) -> Tuple[Array, Array, Array]:
    if len(env_ids) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    unique_envs = np.unique(env_ids)
    rng.shuffle(unique_envs)
    n = len(unique_envs)
    if n < 3:
        raise RuntimeError(
            "At least three solved environments are required for nonempty "
            "training, validation, and testing splits"
        )
    n_val = max(1, int(round(n * dcfg.val_fraction)))
    test_fraction = max(0.0, 1.0 - dcfg.train_fraction - dcfg.val_fraction)
    n_test = max(1, int(round(n * test_fraction)))
    if n_val + n_test >= n:
        n_val = 1
        n_test = 1
    n_train = n - n_val - n_test
    train_envs = set(unique_envs[:n_train].tolist())
    val_envs = set(unique_envs[n_train:n_train + n_val].tolist())
    test_envs = set(unique_envs[n_train + n_val:n_train + n_val + n_test].tolist())
    idx_train, idx_val, idx_test = [], [], []
    for i, e in enumerate(env_ids):
        if int(e) in train_envs:
            idx_train.append(i)
        elif int(e) in val_envs:
            idx_val.append(i)
        elif int(e) in test_envs:
            idx_test.append(i)
        else:
            idx_train.append(i)
    return np.asarray(idx_train, dtype=int), np.asarray(idx_val, dtype=int), np.asarray(idx_test, dtype=int)


def save_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate strict sample-conditioned LQR-CBF basic-RRT data."
    )
    parser.add_argument("--output_dir", type=str, default="V7_RRT_SAMPLE_DATASET")
    parser.add_argument("--num_envs", type=int, default=300)
    parser.add_argument("--paths_per_env", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=4500)
    parser.add_argument("--n_nearest_obstacles", type=int, default=18)
    parser.add_argument("--lookahead_steps", type=int, default=1)
    parser.add_argument("--max_rows_per_problem", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min_obstacles", type=int, default=6)
    parser.add_argument("--max_obstacles", type=int, default=18)
    parser.add_argument("--width", type=float, default=50.0)
    parser.add_argument("--height", type=float, default=50.0)
    parser.add_argument("--step_len", type=float, default=7.0)
    parser.add_argument("--goal_sample_rate", type=float, default=0.18)
    parser.add_argument("--save_expansion_logs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_envs < 10:
        raise ValueError("--num_envs must be at least 10 for nonempty environment-level splits")
    if args.paths_per_env < 1 or args.max_iterations < 1:
        raise ValueError("paths_per_env and max_iterations must be positive")
    if args.lookahead_steps != 1:
        raise ValueError(
            "Strict V7.3 requires --lookahead_steps 1 so every label is the "
            "exact next dynamically feasible RRT state"
        )
    if args.max_rows_per_problem < 0:
        raise ValueError("--max_rows_per_problem cannot be negative (zero means unlimited)")
    if args.min_obstacles < 0 or args.max_obstacles < args.min_obstacles:
        raise ValueError("Invalid obstacle-count range")
    if args.n_nearest_obstacles < args.max_obstacles:
        raise ValueError(
            "--n_nearest_obstacles must be at least --max_obstacles so the safety "
            "benchmark never silently drops an obstacle"
        )
    if not math.isclose(args.width, args.height, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("The strict normalized circular-obstacle schema requires width == height")
    if args.width <= 0 or not 0.0 <= args.goal_sample_rate <= 1.0:
        raise ValueError("Invalid workspace size or goal sample rate")

    dcfg = DatasetConfig(
        output_dir=args.output_dir,
        num_envs=args.num_envs,
        paths_per_env=args.paths_per_env,
        n_nearest_obstacles=args.n_nearest_obstacles,
        lookahead_steps=args.lookahead_steps,
        seed=args.seed,
        save_expansion_logs=args.save_expansion_logs,
        max_rows_per_problem=args.max_rows_per_problem,
    )
    pcfg = PlannerConfig(
        width=args.width,
        height=args.height,
        min_obstacles=args.min_obstacles,
        max_obstacles=args.max_obstacles,
        max_iterations=args.max_iterations,
        step_len=args.step_len,
        goal_sample_rate=args.goal_sample_rate,
        keep_optimizing_after_first_solution=True,
        basic_rrt_only=True,
        supervision_lookahead_steps=args.lookahead_steps,
        max_accepted_expansions=args.max_rows_per_problem,
    )

    out_dir = Path(dcfg.output_dir).expanduser().resolve()
    source_file = Path(__file__).resolve()
    if out_dir in {Path("/").resolve(), Path.home().resolve()}:
        raise ValueError(f"Unsafe output directory: {out_dir}")
    if out_dir == source_file or out_dir in source_file.parents:
        raise ValueError("Output directory would contain or replace the running script")
    if out_dir.is_symlink():
        raise ValueError("Output directory must not be a symbolic link")
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {out_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {out_dir}. Use a new directory or --overwrite."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(dcfg.seed)
    random.seed(dcfg.seed)

    all_rows: Dict[str, List] = {}
    trajectory_records: List[Dict] = []
    expansion_records: List[Dict] = []
    solved = 0
    attempted = 0
    t0 = time.time()

    print(f"SCRIPT VERSION: {SCRIPT_VERSION}")
    print("Generating strict LQR-CBF basic-RRT sample/parent/child data")
    print(f"CONDITION SCHEMA: {CONDITION_SCHEMA}")
    print(f"output_dir={out_dir}")
    print(
        f"num_envs={dcfg.num_envs}, paths_per_env={dcfg.paths_per_env}, "
        f"max_iterations={pcfg.max_iterations}"
    )

    for env_id in range(dcfg.num_envs):
        obstacles = sample_obstacles(rng, pcfg)
        connectivity = FreeSpaceConnectivity(obstacles, pcfg)
        obs_json = [asdict(o) for o in obstacles]
        print(
            f"\nENV {env_id + 1}/{dcfg.num_envs}: obstacles={len(obstacles)}, "
            f"free_components={connectivity.component_count}"
        )
        for path_id in range(dcfg.paths_per_env):
            attempted += 1
            try:
                start, goal = sample_problem(
                    rng, obstacles, pcfg, connectivity=connectivity
                )
            except RuntimeError as exc:
                print(f"  problem {path_id + 1}/{dcfg.paths_per_env}: invalid environment: {exc}")
                continue
            planner = LQRCBFBasicRRT(start, goal, obstacles, pcfg, rng)
            print(
                f"  problem {path_id + 1}/{dcfg.paths_per_env}: planning...",
                flush=True,
            )
            goal_idx = planner.planning()
            if goal_idx is None:
                print("    failed: no strict basic-RRT goal-region solution")
                if dcfg.save_expansion_logs:
                    for row in planner.expansion_logs:
                        row.update(
                            {"env_id": env_id, "problem_id": path_id, "solved": False}
                        )
                        expansion_records.append(row)
                continue

            solved += 1
            states, controls = planner.extract_full_trajectory(goal_idx)
            rows = expansions_to_training_rows(
                planner, goal, obstacles, pcfg, dcfg, env_id, path_id
            )
            extend_rows(all_rows, rows)
            trajectory_records.append(
                {
                    "env_id": env_id,
                    "problem_id": path_id,
                    "solved": True,
                    "num_obstacles": len(obstacles),
                    "conservative_free_space_component_count": int(
                        connectivity.component_count
                    ),
                    "start": start.tolist(),
                    "goal": goal.tolist(),
                    "obstacles": obs_json,
                    "best_cost": float(planner.best_goal_cost),
                    "tree_size": len(planner.nodes),
                    "accepted_expansions": len(planner.accepted_expansions),
                    "exported_rows": len(rows.get("G_X", [])),
                    "trajectory_states": states.tolist(),
                    "trajectory_controls": controls.tolist(),
                    "min_clearance_path": float(
                        min(planner.clearance(s) for s in states)
                    ) if len(states) else None,
                }
            )
            print(
                f"    solved: path_states={len(states)}, accepted={len(planner.accepted_expansions)}, "
                f"exported={len(rows.get('G_X', []))}, tree={len(planner.nodes)}",
                flush=True,
            )
            if dcfg.save_expansion_logs:
                for row in planner.expansion_logs:
                    row.update(
                        {"env_id": env_id, "problem_id": path_id, "solved": True}
                    )
                    expansion_records.append(row)

    condition_dim = 6 + 3 * dcfg.n_nearest_obstacles
    G_X = stack_or_empty(all_rows.get("G_X", []), (condition_dim,))
    G_y = stack_or_empty(all_rows.get("G_y", []), (4,))
    if dcfg.strict_nonempty and len(G_X) == 0:
        raise RuntimeError(
            "No strict expert rows were generated. Increase max_iterations or reduce difficulty."
        )
    G_y_raw = stack_or_empty(all_rows.get("G_y_raw", []), (4,))
    G_u = stack_or_empty(all_rows.get("G_u", []), (2,))
    G_env_id = np.asarray(all_rows.get("G_env_id", []), dtype=np.int32)
    G_problem_id = np.asarray(all_rows.get("G_problem_id", []), dtype=np.int64)
    G_iteration = np.asarray(all_rows.get("G_iteration", []), dtype=np.int32)
    G_parent_index = np.asarray(all_rows.get("G_parent_index", []), dtype=np.int32)
    G_child_index = np.asarray(all_rows.get("G_child_index", []), dtype=np.int32)
    G_sample_xy = stack_or_empty(all_rows.get("G_sample_xy", []), (2,))
    G_sample_state_raw = stack_or_empty(
        all_rows.get("G_sample_state_raw", []), (4,)
    )
    G_goal_state = stack_or_empty(all_rows.get("G_goal_state", []), (4,))
    G_sampled_goal = np.asarray(all_rows.get("G_sampled_goal", []), dtype=np.bool_)
    G_clearance = np.asarray(all_rows.get("G_clearance", []), dtype=np.float32)
    G_cbf_min = np.asarray(all_rows.get("G_cbf_min", []), dtype=np.float32)
    raw_current_states = stack_or_empty(
        all_rows.get("raw_current_states", []), (4,)
    )
    raw_goal_states = stack_or_empty(all_rows.get("raw_goal_states", []), (4,))
    absolute_obstacles = stack_or_empty(
        all_rows.get("absolute_obstacles_per_row", []),
        (dcfg.n_nearest_obstacles, 3),
    )

    arrays_with_rows = {
        "G_y": G_y,
        "G_y_raw": G_y_raw,
        "G_u": G_u,
        "G_env_id": G_env_id,
        "G_problem_id": G_problem_id,
        "G_iteration": G_iteration,
        "G_parent_index": G_parent_index,
        "G_child_index": G_child_index,
        "G_sample_xy": G_sample_xy,
        "G_sample_state_raw": G_sample_state_raw,
        "G_goal_state": G_goal_state,
        "G_sampled_goal": G_sampled_goal,
        "G_clearance": G_clearance,
        "G_cbf_min": G_cbf_min,
        "raw_current_states": raw_current_states,
        "raw_goal_states": raw_goal_states,
        "absolute_obstacles": absolute_obstacles,
    }
    for name, array in arrays_with_rows.items():
        if len(array) != len(G_X):
            raise RuntimeError(f"Row mismatch: G_X={len(G_X)} but {name}={len(array)}")
    if not np.array_equal(G_X[:, 4:6], G_sample_xy):
        raise RuntimeError("Saved condition samples do not exactly equal saved sample arrays")

    train_idx, val_idx, test_idx = create_split_indices(G_env_id.copy(), dcfg, rng)
    for name, idx in (
        ("training", train_idx),
        ("validation", val_idx),
        ("testing", test_idx),
    ):
        if len(idx) == 0:
            raise RuntimeError(f"{name} split is empty; generate more solved environments")
    split_envs = [set(G_env_id[idx].tolist()) for idx in (train_idx, val_idx, test_idx)]
    if (
        split_envs[0] & split_envs[1]
        or split_envs[0] & split_envs[2]
        or split_envs[1] & split_envs[2]
    ):
        raise RuntimeError("Environment leakage detected across train/validation/testing")

    dataset_path = out_dir / "rrt_sample_gan_dataset_v7.npz"
    np.savez_compressed(
        dataset_path,
        train_X=G_X[train_idx],
        train_y=G_y[train_idx],
        validation_X=G_X[val_idx],
        validation_y=G_y[val_idx],
        testing_X=G_X[test_idx],
        testing_y=G_y[test_idx],
        train_sample_xy=G_sample_xy[train_idx],
        validation_sample_xy=G_sample_xy[val_idx],
        testing_sample_xy=G_sample_xy[test_idx],
        training_goal_state=G_goal_state[train_idx],
        validation_goal_state=G_goal_state[val_idx],
        testing_goal_state=G_goal_state[test_idx],
        train_environment_id=G_env_id[train_idx],
        validation_environment_id=G_env_id[val_idx],
        testing_environment_id=G_env_id[test_idx],
        train_scenario_id=G_problem_id[train_idx],
        validation_scenario_id=G_problem_id[val_idx],
        testing_scenario_id=G_problem_id[test_idx],
        train_iteration=G_iteration[train_idx],
        validation_iteration=G_iteration[val_idx],
        testing_iteration=G_iteration[test_idx],
        train_parent_index=G_parent_index[train_idx],
        validation_parent_index=G_parent_index[val_idx],
        testing_parent_index=G_parent_index[test_idx],
        train_child_index=G_child_index[train_idx],
        validation_child_index=G_child_index[val_idx],
        testing_child_index=G_child_index[test_idx],
        target_source=np.asarray(["recorded_rrt_sample_xy"] * 3, dtype="<U32"),
        condition_schema=np.asarray(CONDITION_SCHEMA),
        obstacle_coordinate_schema=np.asarray(
            "absolute_normalized_center_x_center_y_radius_nearest_first"
        ),
        generator_script_version=np.asarray(SCRIPT_VERSION),
        basic_rrt_contract=np.asarray(
            "sample->nearest_parent->lqr_cbf_steer->safety_check->one_child"
        ),
        workspace_min=np.asarray(0.0, dtype=np.float32),
        workspace_max=np.asarray(1.0, dtype=np.float32),
        collision_margin=np.asarray(
            (pcfg.robot_radius + pcfg.safety_margin) / max(pcfg.width, pcfg.height),
            dtype=np.float32,
        ),
        goal_radius=np.asarray(
            (pcfg.lqr_goal_tolerance * 1.5) / max(pcfg.width, pcfg.height),
            dtype=np.float32,
        ),
        continuous_edge_safety_schema=np.asarray(
            "constant_acceleration_curve_bound_plus_swept_chord"
        ),
        geometric_connectivity_filter=np.asarray(
            "four_neighbor_configuration_space_grid_same_component"
        ),
        cbf_control_filter_schema=np.asarray(
            "exact_2d_active_set_projection_of_lqr_onto_hocbf_halfspaces"
        ),
        nominal_dt=np.asarray(pcfg.dt, dtype=np.float32),
        supervision_lookahead_steps=np.asarray(
            pcfg.supervision_lookahead_steps, dtype=np.int32
        ),
        position_scale=np.asarray([pcfg.width, pcfg.height], dtype=np.float32),
        velocity_scale=np.asarray(pcfg.max_speed, dtype=np.float32),
        n_obstacles_encoded=np.asarray(dcfg.n_nearest_obstacles, dtype=np.int32),
        G_X=G_X,
        G_y=G_y,
        G_y_raw=G_y_raw,
        G_u=G_u,
        G_env_id=G_env_id,
        G_problem_id=G_problem_id,
        G_iteration=G_iteration,
        G_parent_index=G_parent_index,
        G_child_index=G_child_index,
        G_sample_xy=G_sample_xy,
        G_sample_state_raw=G_sample_state_raw,
        G_goal_state=G_goal_state,
        G_sampled_goal=G_sampled_goal,
        G_clearance=G_clearance,
        G_cbf_min=G_cbf_min,
        raw_current_states=raw_current_states,
        raw_goal_states=raw_goal_states,
        absolute_obstacles_per_row=absolute_obstacles,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    save_jsonl(out_dir / "solved_trajectories.jsonl", trajectory_records)
    if dcfg.save_expansion_logs:
        save_jsonl(out_dir / "expansion_logs.jsonl", expansion_records)

    cfg_json = {
        "script_version": SCRIPT_VERSION,
        "condition_schema": CONDITION_SCHEMA,
        "planner_config": asdict(pcfg),
        "dataset_config": asdict(dcfg),
    }
    (out_dir / "config.json").write_text(
        json.dumps(cfg_json, indent=2), encoding="utf-8"
    )
    summary = {
        "status": "ok",
        "script_version": SCRIPT_VERSION,
        "dataset_path": str(dataset_path),
        "attempted_problems": attempted,
        "solved_problems": solved,
        "solve_rate_percent": 100.0 * solved / attempted if attempted else 0.0,
        "all_rows": int(len(G_X)),
        "condition_dim": int(G_X.shape[1]),
        "supervision_lookahead_steps": int(pcfg.supervision_lookahead_steps),
        "training_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "testing_rows": int(len(test_idx)),
        "training_environments": len(split_envs[0]),
        "validation_environments": len(split_envs[1]),
        "testing_environments": len(split_envs[2]),
        "environment_overlap_count": 0,
        "sample_column_exact_match": True,
        "obstacle_coordinate_schema": (
            "absolute_normalized_center_x_center_y_radius_nearest_first"
        ),
        "collision_margin_normalized": float(
            (pcfg.robot_radius + pcfg.safety_margin) / max(pcfg.width, pcfg.height)
        ),
        "goal_radius_normalized": float(
            (pcfg.lqr_goal_tolerance * 1.5) / max(pcfg.width, pcfg.height)
        ),
        "continuous_edge_safety_schema": (
            "constant_acceleration_curve_bound_plus_swept_chord"
        ),
        "geometric_connectivity_filter": (
            "four_neighbor_configuration_space_grid_same_component"
        ),
        "cbf_control_filter_schema": (
            "exact_2d_active_set_projection_of_lqr_onto_hocbf_halfspaces"
        ),
        "elapsed_sec": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    readme = f"""# V7 strict RRT-sample GAN dataset

Each row is one accepted basic-RRT expansion:

1. Sample a target.
2. Select the nearest existing tree node.
3. Run deterministic LQR steering toward that exact sample.
4. Reject an unsafe CBF edge.
5. Add one fixed-horizon child and save the exact tuple.

Condition schema:
`{CONDITION_SCHEMA}`

Obstacle centers are absolute normalized coordinates. Radius is normalized by
the square workspace size. Obstacles are ordered nearest-first for each parent;
zero-radius triples are padding. Train, validation, and testing are split by
whole environment, not by row. The held-out testing split is never used for
checkpoint selection. Every expert acceleration is the exact two-dimensional
projection of nominal LQR onto the HOCBF/acceleration halfspaces. Accepted edges
also pass a conservative continuous swept-edge certificate between states.

Main dataset: `rrt_sample_gan_dataset_v7.npz`
"""
    (out_dir / "README_dataset.md").write_text(readme, encoding="utf-8")

    print("\nDONE")
    print(json.dumps(summary, indent=2))
    print(f"Saved to: {dataset_path}")


if __name__ == "__main__":
    main()
