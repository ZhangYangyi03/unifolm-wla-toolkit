# -*- coding: utf-8 -*-
"""make_demo_full.py - synthetic data that exercises most unified slots.

Both end effectors, both grippers, waist, torso, base velocity, base inertial
state and both legs. Still a toy trajectory, but every slot in the 54/60 layout
that this toolkit writes is covered, so the pipeline can be checked end to end.

    python demo/make_demo_full.py demo/data_full
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unified_repr as U


def episode(seed, fps=30, T=60, H=30):
    rng = np.random.default_rng(seed)
    t = np.arange(T + H) / fps

    def arm(phase, sign):
        x = 0.35 + sign * 0.05 + 0.10 * np.sin(2 * np.pi * 0.4 * t + phase)
        y = sign * 0.18 + 0.05 * np.cos(2 * np.pi * 0.3 * t + phase)
        z = 0.85 + 0.07 * np.sin(2 * np.pi * 0.25 * t + phase)
        r = 0.05 * np.sin(2 * np.pi * 0.2 * t)
        p = sign * 0.10 + 0.05 * np.cos(2 * np.pi * 0.2 * t)
        yw = sign * 1.10 + 0.20 * np.sin(2 * np.pi * 0.15 * t + phase)
        return np.stack([x, y, z, r, p, yw], axis=1)

    yaw = 0.15 * np.sin(2 * np.pi * 0.1 * t)
    R_WB = np.stack([U.rot_rpy([0.0, 0.0, y]) for y in yaw])           # (T+H,3,3)
    w = np.stack([np.zeros(len(t)), np.zeros(len(t)),
                  0.15 * 2 * np.pi * 0.1 * np.cos(2 * np.pi * 0.1 * t)], axis=1)
    base_v = np.stack([0.10 * np.sin(2 * np.pi * 0.2 * t),
                       0.05 * np.cos(2 * np.pi * 0.2 * t)], axis=1)
    waist = np.stack([0.10 * np.sin(2 * np.pi * 0.2 * t + 0.2)], axis=1)
    waist = np.concatenate([waist, np.zeros((len(t), 2))], axis=1)
    legs = np.stack([0.20 * np.sin(2 * np.pi * 0.5 * t)] * 6, axis=1)
    grip = 0.5 + 0.5 * np.tanh(3 * np.sin(2 * np.pi * 0.5 * t))
    return dict(left_ee=arm(0.0, 1), right_ee=arm(1.3, -1), left_gripper=grip,
                right_gripper=grip[::-1], waist=waist, R_WB=R_WB,
                angular_velocity=w, base_vel=base_v, left_leg=legs, right_leg=legs)


def write(dest, n_ep=3, fps=30, T=60, H=30):
    os.makedirs(dest, exist_ok=True)
    meta = {'fps': fps, 'pose_format': 'rpy', 'chunk_size': H,
            'modules': ['left_ee', 'left_gripper', 'right_ee', 'right_gripper',
                        'waist', 'base_vel', 'left_leg', 'right_leg']}
    json.dump(meta, open(os.path.join(dest, 'meta.json'), 'w', encoding='utf-8'), indent=2)
    for e in range(n_ep):
        d = episode(e, fps, T, H)
        out = {}
        pose_mods = [k for k in d if k in U.POSE_MODULES]
        for k, v in d.items():
            if k in pose_mods:
                out['cur/' + k] = v[:T]
                out['action/' + k] = np.stack([v[t + 1:t + 1 + H] for t in range(T)])
            elif k == 'R_WB':
                out['cur/R_WB'] = v[:T]
            elif k == 'angular_velocity':
                out['cur/angular_velocity'] = v[:T]
            elif k == 'base_vel':
                out['cur/base_vel'] = v[:T]
                out['action/base_vel'] = np.stack([v[t + 1:t + 1 + H] for t in range(T)])
            elif k in ('waist', 'left_leg', 'right_leg', 'left_gripper', 'right_gripper'):
                arr = v[:, None] if v.ndim == 1 else v
                out['cur/' + k] = arr[:T]
                out['action/' + k] = np.stack([arr[t + 1:t + 1 + H] for t in range(T)])
        np.savez_compressed(os.path.join(dest, 'episode_%03d.npz' % e), **out)
    print('wrote %d episodes (multi-slot) to %s' % (n_ep, dest))


if __name__ == '__main__':
    write(sys.argv[1] if len(sys.argv) > 1 else 'demo/data_full')
