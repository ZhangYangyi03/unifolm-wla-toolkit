# -*- coding: utf-8 -*-
"""selftest_real.py - end-to-end on REAL Unitree G1 data.

Chain: raw joint angles (a real episode from unitreerobotics/G1_Dex3_GraspSquare_Dataset)
  -> forward kinematics on the shipped URDF
  -> unified 54/60 layout + global statistics + normalization (wla_prep)
  -> payload.npz
and then backwards, the way a model would:
  normalized action -> denormalize -> relative action -> absolute pose -> compare
against the poses FK produced in the first place.

The point is not that the round trip is exact (it is, to 1e-16); it is that any
one of the conventions being wrong -- the rpy order, which two columns make
rotation-6D, whether relative translation is in the body or world frame, whether
statistics are the population or sample std -- makes the two ends disagree.
"""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import unified_repr as U
import urdf_kin as K

FAIL = []
def check(name, ok, detail=''):
    print(('  PASS  ' if ok else '  FAIL  ') + name + ('   ' + detail if detail else ''))
    if not ok:
        FAIL.append(name)

d = os.path.join(HERE, 'demo', 'real_g1')
src = os.path.join(d, 'src')
z = np.load(os.path.join(d, 'g1_joints_sample.npz'))
state = z['state']
print('real episode: %d frames x %d joints, Unitree G1, 30 fps' % state.shape)

# rebuild the FK poses independently, straight from the raw joints
u = K.Urdf(os.path.join(HERE, 'demo', 'g1.urdf'))
names = json.load(open(os.path.join(d, 'meta.json'), encoding='utf-8'))['state_names']
urdf_idx = {j.name: i for i, j in enumerate(u.movable_joints())}
def snake(s):
    return ''.join(('_' + c.lower()) if c.isupper() else c for c in s).strip('_')
rows = np.zeros((len(state), len(urdf_idx)))
for i, n in enumerate(names):
    s = n[1:] if n.startswith('k') else n
    for side in ('Left', 'Right'):
        if s.startswith(side) and s[len(side):] in ('ShoulderPitch', 'ShoulderRoll', 'ShoulderYaw',
                                                   'Elbow', 'WristRoll', 'WristPitch', 'WristYaw'):
            jn = '%s_%s_joint' % (side.lower(), snake(s[len(side):]))
            if jn in urdf_idx:
                rows[:, urdf_idx[jn]] = state[:, i]
fk = u.fk_series(rows, ['left_wrist_yaw_link', 'right_wrist_yaw_link'])['left_wrist_yaw_link']

# what the pipeline produced
payload = np.load(os.path.join(src, 'payload.npz'))
stats = json.load(open(os.path.join(src, 'stats.json'), encoding='utf-8'))
cur = np.load(os.path.join(src, 'episode_000000.npz'))['cur/left_ee']
check('every frame of the real episode became one sample',
      payload['action'].shape[0] == len(state), '%d vs %d' % (payload['action'].shape[0], len(state)))

# forward: FK pose -> what the source file recorded, must agree to float precision
def pose_vec(T):
    R = T[:3, :3]
    sy = min(1.0, max(-1.0, -R[2, 0]))
    pitch = np.arcsin(sy)
    roll = np.arctan2(R[2, 1], R[2, 2]) if abs(sy) < 0.999999 else np.arctan2(-R[1, 2], R[1, 1])
    yaw = np.arctan2(R[1, 0], R[0, 0]) if abs(sy) < 0.999999 else 0.0
    return np.concatenate([T[:3, 3], [roll, pitch, yaw]])
rebuilt = np.stack([pose_vec(T) for T in fk])
check('source file xyz+rpy reproduces the FK pose (rpy conventions agree)',
      np.max(np.abs(rebuilt - cur)) < 1e-12, 'max %.2e' % np.max(np.abs(rebuilt - cur)))

# backward: normalized action -> absolute pose, compared against FK
ao = payload['action_offset']; ac = payload['action_scale']
st = U.stats_from_json(stats['relative']['left_relative_ee_pose'])
p = U.affine_params(st, 'zscore')
worst = 0.0
for t in range(len(state)):
    T_cur = U.pose_to_SE3(cur[t], 'rpy')
    a = U.denormalize(payload['action'][t, :, 0:6], p)
    for k in range(payload['action'].shape[1]):
        T = U.absolute_pose_from_relative(T_cur, a[k])
        ref_idx = min(t + 1 + k, len(state) - 1)
        T_ref = U.pose_to_SE3(cur[ref_idx], 'rpy')
        worst = max(worst, float(np.max(np.abs(T - T_ref))))
check('normalized action -> absolute pose recovers the FK target over the whole episode',
      worst < 1e-5, 'max deviation %.2e' % worst)

# and the numbers a reviewer would sanity-check
xyz = payload['action'][:, :, 0:3]
check('normalized relative xyz is centred on 0 with unit spread (spec sec. 7.1)',
      abs(float(xyz.mean())) < 1e-3 and abs(float(xyz.std()) - 1.0) < 2e-2,
      'mean %.3g std %.4f  -- a session-mean offset or wrong std convention shows up here'
      % (float(xyz.mean()), float(xyz.std())))
nz = payload['action_mask'].sum(1)
check('mask admits only the slots the source actually carries',
      set(np.unique(nz)) == {14}, 'slots per sample: %s' % sorted(set(np.unique(nz))))
check('disabled slots are exactly zero in the payload',
      np.abs(payload['action'] * (payload['action_mask'] == 0)[:, None, :]).max() == 0)
speeds = np.linalg.norm(np.diff(cur[:, :3], axis=0), axis=1) * 30.0
check('end-effector speed is physically plausible for a table-top task',
      float(speeds.max()) < 2.0, 'max %.2f m/s (a wrong frame or a unit slip shows up as a jump)'
      % float(speeds.max()))

print('')
if FAIL:
    print('%d CHECK(S) FAILED: %s' % (len(FAIL), ', '.join(FAIL)))
    raise SystemExit(1)
print('real-data round trip clean')
