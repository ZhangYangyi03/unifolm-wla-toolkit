# -*- coding: utf-8 -*-
"""make_demo.py - write a tiny synthetic dataset in the wla-prep source layout.

Not a simulator: a smooth 2-second reach on a table, 3 episodes, 30 fps.
Its only job is to make the pipeline runnable and checkable end to end without
a robot, a GPU or a download.

    python demo/make_demo.py demo/data
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unified_repr as U


def episode(seed, fps=30, T=90, H=30):
    rng = np.random.default_rng(seed)
    t = np.arange(T + H) / fps
    x = 0.35 + 0.15 * np.sin(2 * np.pi * 0.4 * t + seed)
    y = -0.10 + 0.05 * np.cos(2 * np.pi * 0.3 * t)
    z = 0.85 + 0.08 * np.sin(2 * np.pi * 0.25 * t + 0.5)
    roll = 0.05 * np.sin(2 * np.pi * 0.2 * t)
    pitch = -0.10 + 0.05 * np.cos(2 * np.pi * 0.2 * t)
    yaw = 1.10 + 0.20 * np.sin(2 * np.pi * 0.15 * t)
    pose = np.stack([x, y, z, roll, pitch, yaw], axis=1)           # (T+H, 6) xyz+rpy
    grip = 0.5 + 0.5 * np.tanh(3 * np.sin(2 * np.pi * 0.5 * t))    # 0..1 open/close
    return pose, grip


def write(dest, n_ep=3, fps=30, T=90, H=30):
    os.makedirs(dest, exist_ok=True)
    meta = {'fps': fps, 'pose_format': 'rpy',
            'modules': ['left_ee', 'left_gripper'],
            'chunk_size': H,
            'note': 'synthetic demo, pose is xyz+rpy in the world/robot base frame'}
    with open(os.path.join(dest, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    for e in range(n_ep):
        pose, grip = episode(e, fps, T, H)
        cur_pose = pose[:T]
        cur_grip = grip[:T]
        act_pose = np.stack([pose[t + 1:t + 1 + H] for t in range(T)])   # (T,H,6)
        act_grip = np.stack([grip[t + 1:t + 1 + H] for t in range(T)])   # (T,H)
        np.savez_compressed(os.path.join(dest, 'episode_%03d.npz' % e),
                            **{'cur/left_ee': cur_pose,
                               'cur/left_gripper': cur_grip,
                               'action/left_ee': act_pose,
                               'action/left_gripper': act_grip})
    print('wrote %d episodes to %s  (fps=%d T=%d H=%d)' % (n_ep, dest, fps, T, H))


if __name__ == '__main__':
    write(sys.argv[1] if len(sys.argv) > 1 else 'demo/data')
