# -*- coding: utf-8 -*-
"""unitree_g1_to_unified.py - real Unitree G1 data into the wla_prep source layout.

The public Unitree datasets (the ones WLA-1.0 was trained on) are LeRobot v3
episodes of JOINT ANGLES at 30 fps. The unified action space wants end-effector
and base POSES. This bridges the two with forward kinematics on the exact URDF
the robot ships, and refuses to invent what the data does not contain.

    python unitree_g1_to_unified.py --joints demo/real_g1/g1_joints_sample.npz \
        --urdf demo/g1.urdf --out demo/real_g1/src --fps 30

What it fills:
    left/right end-effector pose      <- FK from the arm joints
    left/right hand state and action  <- the finger joints, first 6 as the spec says

What it will not invent:
    the base/inertial state (slots [41:47]) needs base orientation and angular
    velocity, which a joint-angle episode does not have; a static torso is
    assumed and stated as an assumption, not silently zero-filled into a
    physical slot. Pass --base-rpy / --base-omega when you do have them.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import urdf_kin as K

ARM_JOINTS = ['ShoulderPitch', 'ShoulderRoll', 'ShoulderYaw', 'Elbow',
              'WristRoll', 'WristPitch', 'WristYaw']
HAND_JOINTS = ['Thumb0', 'Thumb1', 'Thumb2', 'Middle0', 'Middle1', 'Index0', 'Index1']


def map_names(names, side):
    """Dataset joint names -> URDF joint names. 'kLeftShoulderPitch' -> left_shoulder_pitch_joint."""
    out = {}
    for i, n in enumerate(names):
        s = n[1:] if n.startswith('k') else n
        for s2 in ('Left', 'Right'):
            if s.startswith(s2) and s2.lower() == side:
                rest = s[len(s2):]
                out[i] = (rest, s2.lower())
    return out


def build(src_npz, urdf_path, out_dir, fps, base_rpy=(0.0, 0.0, 0.0), base_omega=(0.0, 0.0, 0.0),
          meta_path=None, chunk_size=30, name='episode_000000.npz'):
    d = np.load(src_npz)
    state = np.asarray(d['state'], dtype=np.float64)
    action = np.asarray(d['action'], dtype=np.float64)
    meta = json.load(open(meta_path, encoding='utf-8')) if meta_path else {}
    names = meta.get('state_names') or ['kLeftShoulderPitch']
    if len(names) != state.shape[1]:
        raise ValueError('meta state_names has %d entries but the array has %d columns'
                         % (len(names), state.shape[1]))
    u = K.Urdf(urdf_path)
    urdf_index = {j.name: i for i, j in enumerate(u.movable_joints())}
    njoint = len(urdf_index)

    def to_urdf_q(row):
        q = np.zeros(njoint)
        for i, n in enumerate(names):
            s = n[1:] if n.startswith('k') else n
            for side in ('Left', 'Right'):
                if not s.startswith(side):
                    continue
                rest = s[len(side):]
                if rest in ARM_JOINTS:
                    jn = '%s_%s_joint' % (side.lower(), _snake(rest))
                    if jn in urdf_index:
                        q[urdf_index[jn]] = row[i]
        return q

    def _snake(camel):
        out = ''
        for ch in camel:
            out += ('_' + ch.lower()) if ch.isupper() else ch
        return out.strip('_')

    T = state.shape[0]
    joint_rows = np.stack([to_urdf_q(state[t]) for t in range(T)])
    joint_act = np.stack([to_urdf_q(action[t]) for t in range(T)])
    links = ['left_wrist_yaw_link', 'right_wrist_yaw_link']
    fk_cur = u.fk_series(joint_rows, links)
    poses = {l: np.stack([_pose_vec(fk_cur[l][t]) for t in range(T)]) for l in links}

    os.makedirs(out_dir, exist_ok=True)
    mjson = {'fps': fps, 'pose_format': 'rpy', 'chunk_size': chunk_size,
             'modules': ['left_ee', 'left_gripper', 'right_ee', 'right_gripper'],
             'source': os.path.basename(src_npz), 'urdf': os.path.basename(urdf_path),
             'assumptions': {
                 'base_pose': 'static, rpy=%s' % (list(base_rpy),),
                 'base_omega': list(base_omega),
                 'hand_slot': 'first 6 finger joints of each hand'}}
    json.dump(mjson, open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    H = chunk_size
    out = {}
    # future window per frame, clamped at the end of the episode instead of
    # running off it: idx[t, k] = min(t + 1 + k, T - 1)
    win = np.minimum(np.arange(T)[:, None] + 1 + np.arange(H)[None, :], T - 1)
    for side, link in (('left', 'left_wrist_yaw_link'), ('right', 'right_wrist_yaw_link')):
        arr = poses[link]
        out['cur/%s_ee' % side] = arr[:T]
        out['action/%s_ee' % side] = arr[win]
    for side in ('left', 'right'):
        cols = [i for i, n in enumerate(names) if n.lower().startswith('k' + side) and
                any(h.lower() in n.lower() for h in HAND_JOINTS)]
        if not cols:
            continue
        hand = state[:, cols]
        out['cur/%s_hand' % side] = hand
        out['cur/%s_gripper' % side] = hand[:, 0]
        hand_chunks = hand[win]                 # (T, H, n_hand_joints)
        out['action/%s_hand' % side] = hand_chunks
        out['action/%s_gripper' % side] = hand_chunks[:, :, 0]
    np.savez_compressed(os.path.join(out_dir, name), **out)
    return os.path.join(out_dir, name), T, poses


def _pose_vec(T):
    """4x4 -> xyz + rpy (the spec's sec. 5.1 input form)."""
    R = T[:3, :3]
    # rpy from the fixed-axis xyz convention, R = Rz(y) Ry(p) Rx(r)
    sy = -R[2, 0]
    sy = min(1.0, max(-1.0, sy))
    pitch = np.arcsin(sy)
    if abs(sy) < 0.999999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.concatenate([T[:3, 3], [roll, pitch, yaw]])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--joints', required=True)
    ap.add_argument('--urdf', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--chunk-size', type=int, default=30, dest='chunk_size')
    ap.add_argument('--meta', default=None)
    ap.add_argument('--base-rpy', default='0,0,0', dest='base_rpy')
    ap.add_argument('--base-omega', default='0,0,0', dest='base_omega')
    a = ap.parse_args(argv)
    meta = a.meta
    if meta is None:
        cand = os.path.join(os.path.dirname(os.path.abspath(a.joints)), 'meta.json')
        meta = cand if os.path.exists(cand) else None
    p, T, poses = build(a.joints, a.urdf, a.out, a.fps,
                        tuple(float(v) for v in a.base_rpy.split(',')),
                        tuple(float(v) for v in a.base_omega.split(',')),
                        meta, a.chunk_size)
    print('wrote %s   frames=%d' % (p, T))
    for k, v in poses.items():
        d = np.linalg.norm(np.diff(v[:, :3], axis=0), axis=1) * a.fps
        print('  %-22s xyz range %s .. %s  max speed %.2f m/s'
              % (k, np.round(v[:, :3].min(0), 3), np.round(v[:, :3].max(0), 3), float(d.max())))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
