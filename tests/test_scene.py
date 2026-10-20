"""Scene randomisation tests (determinism, reachability, split semantics)."""

from __future__ import annotations

import numpy as np

from so101_sim import kinematics as kin
from so101_sim.scene import SEEN_COLORS, UNSEEN_COLORS, Scene


def test_scene_is_deterministic_for_a_seed():
    a = Scene(seed=42)
    b = Scene(seed=42)
    assert np.allclose(a.target.xy, b.target.xy)
    assert a.target.color == b.target.color
    assert np.allclose(a.box.xy, b.box.xy)
    assert np.allclose(a.camera.position, b.camera.position)


def test_different_seeds_give_different_scenes():
    a = Scene(seed=1)
    b = Scene(seed=2)
    assert (not np.allclose(a.target.xy, b.target.xy)
            or a.target.color != b.target.color)


def test_scene_objects_are_reachable():
    for seed in range(10):
        scene = Scene(seed=seed)
        for block in scene.blocks:
            assert kin.can_reach(block.center3d)
        assert kin.can_reach(np.array([scene.box.xy[0], scene.box.xy[1], 0.05]))


def test_block_and_box_are_separated():
    for seed in range(10):
        scene = Scene(seed=seed)
        assert np.linalg.norm(scene.target.xy - scene.box.xy) > 0.11


def test_seen_split_uses_seen_colours():
    colors = {Scene(seed=s, split="seen").target.color for s in range(20)}
    assert colors <= set(SEEN_COLORS)


def test_unseen_split_is_out_of_distribution():
    scenes = [Scene(seed=s, split="unseen") for s in range(10)]
    assert all(s.target.color in UNSEEN_COLORS for s in scenes)
    seen_radii = [np.linalg.norm(Scene(seed=s).target.xy - kin.BASE_XYZ[:2])
                  for s in range(10)]
    unseen_radii = [np.linalg.norm(s.target.xy - kin.BASE_XYZ[:2]) for s in scenes]
    assert np.mean(unseen_radii) > np.mean(seen_radii)


def test_camera_mounting_error_is_bounded():
    base = Scene(seed=0).camera.position
    errors = [np.linalg.norm(Scene(seed=s).camera.position - base) for s in range(20)]
    assert max(errors) < 0.10  # jitter sigma ~1.2 cm


def test_language_scene_has_distractor_and_instruction_names_the_target():
    scene = Scene(seed=3, split="lang", n_blocks=2, target_color="red")
    assert len(scene.blocks) == 2
    assert scene.target.color == "red"
    assert "红色" in scene.instruction
    assert all(b.color != "red" for b in scene.blocks[1:])
