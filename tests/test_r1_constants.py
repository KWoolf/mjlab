"""Tests for r1_constants.py."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.unitree_r1 import r1_constants
from mjlab.entity import Entity
from mjlab.scripts.csv_to_npz import CSV_JOINT_NAMES
from mjlab.utils.string import resolve_expr

FOOT_PATTERN = r"^(left|right)_foot[1-7]_collision$"


@pytest.fixture(scope="module")
def r1_entity() -> Entity:
  return Entity(r1_constants.get_r1_robot_cfg())


@pytest.fixture(scope="module")
def r1_model(r1_entity: Entity) -> mujoco.MjModel:
  return r1_entity.spec.compile()


def test_r1_entity_creation(r1_entity) -> None:
  assert r1_entity.num_joints == 26
  assert r1_entity.num_actuators == 26
  assert not r1_entity.is_fixed_base


def test_each_joint_has_one_actuator_group() -> None:
  joint_names = CSV_JOINT_NAMES["r1"]
  for name in joint_names:
    groups = [
      a
      for a in r1_constants.R1_ARTICULATION.actuators
      if any(re.fullmatch(p, name) for p in a.target_names_expr)
    ]
    assert len(groups) == 1, f"{name} matches {len(groups)} actuator groups"


def test_csv_joint_order_matches_model(r1_entity) -> None:
  # GMR's robot_motion_to_csv.py writes joints in model order.
  assert tuple(r1_entity.joint_names) == CSV_JOINT_NAMES["r1"]


def test_keyframe_joint_positions(r1_entity, r1_model) -> None:
  key = r1_model.key("init_state")
  joint_pos = r1_constants.HOME_KEYFRAME.joint_pos
  assert joint_pos is not None
  expected = resolve_expr(joint_pos, r1_entity.joint_names, 0.0)
  for name, value in zip(r1_entity.joint_names, expected, strict=True):
    qpos_idx = r1_model.joint(name).qposadr[0]
    np.testing.assert_allclose(key.qpos[qpos_idx], value, rtol=1e-5)


def test_foot_collision_geoms(r1_model) -> None:
  foot_geoms = []
  for i in range(r1_model.ngeom):
    geom = r1_model.geom(i)
    if re.match(FOOT_PATTERN, geom.name):
      foot_geoms.append(geom.name)
      assert geom.condim == 3
      assert geom.priority == 1
      assert geom.friction[0] == 0.6
    elif "_collision" in geom.name:
      assert geom.condim == 1
  assert len(foot_geoms) == 14
