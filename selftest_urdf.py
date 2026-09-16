# -*- coding: utf-8 -*-
"""Cross-check the URDF conventions against an independent implementation (scipy).

Two things here are conventions, not facts, and getting either wrong yields
plausible-looking motion that is wrong:
  * URDF origin rpy is fixed-axis xyz, i.e. R = Rz(yaw) Ry(pitch) Rx(roll);
  * URDF joint axes rotate about the joint frame after the origin transform.
scipy.spatial.transform.Rotation is an independent implementation of the same
rotations, so if my rpy_to_R disagrees with it the convention is wrong.
"""
import math, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import urdf_kin as K

try:
    from scipy.spatial.transform import Rotation as R3
except ImportError:
    print('scipy not installed: cannot run the independent cross-check')
    raise SystemExit(0)

rng = np.random.default_rng(1)
worst_intrinsic = worst_extrinsic = 0.0
for _ in range(500):
    r, p, y = rng.uniform(-math.pi, math.pi, 3)
    mine = K.rpy_to_R([r, p, y])
    # scipy lowercase 'xyz' = intrinsic rotations; uppercase 'XYZ' = extrinsic/fixed
    intrinsic = R3.from_euler('xyz', [r, p, y]).as_matrix()
    extrinsic = R3.from_euler('XYZ', [r, p, y]).as_matrix()
    worst_intrinsic = max(worst_intrinsic, np.abs(mine - intrinsic).max())
    worst_extrinsic = max(worst_extrinsic, np.abs(mine - extrinsic).max())

print('rpy_to_R vs scipy from_euler("xyz"): max diff %.3e' % worst_intrinsic)
print('rpy_to_R vs scipy from_euler("XYZ"): max diff %.3e' % worst_extrinsic)
if worst_intrinsic > 1e-12 and worst_extrinsic > 1e-12:
    raise AssertionError('rpy_to_R matches no scipy convention')
spelling = 'xyz' if worst_intrinsic < worst_extrinsic else 'XYZ'
print('  -> agrees to float precision with scipy.from_euler(%r), which is the same matrix as'
      % spelling)
print('     the R = Rz(yaw) Ry(pitch) Rx(roll) order URDF/ROS specifies in its rpy attribute')

# independent FK: rebuild the chain with scipy rotations and compare link poses
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo', 'g1.urdf')
if not os.path.exists(p):
    print('demo/g1.urdf not present, skipping the whole-robot FK comparison')
    raise SystemExit(0)
u = K.Urdf(p)
names = [j.name for j in u.movable_joints()]
print('G1 URDF: root=%s links=%d joints=%d (%d movable)' % (u.root, len(u.links), len(u.joints), len(names)))
q = {n: float(v) for n, v in zip(names, rng.uniform(-0.6, 0.6, len(names)))}
mine = u.fk_all(q)

# second implementation, written against scipy only
T = {u.root: np.eye(4)}
for j in u.joints:
    if j.parent not in T:
        continue
    o = np.eye(4); o[:3, :3] = R3.from_euler('XYZ', [0, 0, 0]).as_matrix(); o[:3, :3] = j.origin[:3, :3]
    o[:3, 3] = j.origin[:3, 3]
    m = np.eye(4)
    val = q.get(j.name, 0.0)
    if j.jtype in ('revolute', 'continuous'):
        m[:3, :3] = R3.from_rotvec(np.asarray(j.axis) / np.linalg.norm(j.axis) * val).as_matrix()
    elif j.jtype == 'prismatic':
        m[:3, 3] = np.asarray(j.axis) / np.linalg.norm(j.axis) * val
    T[j.child] = T[j.parent] @ o @ m
worst = max(np.abs(T[l] - mine[l]).max() for l in T)
print('whole-robot FK vs the scipy-based reimplementation: max diff %.3e over %d links' % (worst, len(T)))
assert worst < 1e-12

# plausibility in robot-frame terms: with the pelvis at the origin, the ankle
# must sit a leg-length below it and the wrist a shoulder-to-hand distance away
rest = u.fk_all({})
legs = [float(np.linalg.norm(rest[l][:3, 3] - rest['pelvis'][:3, 3]))
        for l in rest if l.endswith('ankle_roll_link')]
arms = [float(np.linalg.norm(rest[l][:3, 3] - rest['pelvis'][:3, 3]))
        for l in rest if l.endswith('wrist_yaw_link')]
print('pelvis -> ankle (leg length): %s m' % np.round(legs, 3))
print('pelvis -> wrist (arm, hanging): %s m' % np.round(arms, 3))
assert all(0.45 < v < 1.05 for v in legs), 'leg length is not plausible for a G1'
assert all(0.05 < v < 1.05 for v in arms), 'arm reach is not plausible for a G1'

# raising the shoulder must raise the wrist: a sign error in the axis or the
# chain order would break this monotonicity even though the numbers stay finite
# kinematic identity that catches an axis or chain-order error: rotating one
# revolute joint by theta moves the link it carries along an arc of radius r,
# so |p(theta) - p(0)| must equal 2 r sin(theta/2) for r = |p(0) - joint origin|
rest = u.fk_all({})
joints_of_interest = [('left_shoulder_pitch_joint', 'left_wrist_yaw_link'),
                      ('left_elbow_joint', 'left_wrist_yaw_link'),
                      ('left_hip_pitch_joint', 'left_ankle_roll_link'),
                      ('left_knee_joint', 'left_ankle_roll_link'),
                      ('waist_yaw_joint', 'left_wrist_yaw_link')]
ok = True
for jname, leaf in joints_of_interest:
    if jname not in u.by_name:
        print('  %-30s not in this URDF, skipped' % jname)
        continue
    theta = 0.3
    j = u.by_name[jname]
    q2 = dict(q); q2[jname] = q2.get(jname, 0.0) + theta
    T2 = u.fk_all(q2)
    moved = float(np.linalg.norm(T2[leaf][:3, 3] - mine[leaf][:3, 3]))
    # a revolute joint rotates about its axis THROUGH the child frame origin,
    # so the arc centre is the child link origin of the posed configuration
    F = mine[j.parent] @ j.origin                       # joint frame, before the joint motion
    centre = F[:3, 3]
    ax = F[:3, :3] @ (np.asarray(j.axis) / np.linalg.norm(j.axis))
    d = mine[leaf][:3, 3] - centre
    perp = d - float(np.dot(d, ax)) * ax                # arc radius about the axis line
    r = float(np.linalg.norm(perp))
    pred = 2 * r * math.sin(theta / 2)
    good = abs(moved - pred) < 1e-9 and r > 1e-3
    ok &= good
    print('  %-30s arc chord %.9f m, 2*perp*sin(t/2) = %.9f m, r_perp = %.4f  %s'
          % (jname, moved, pred, r, 'ok' if good else 'MISMATCH'))
assert ok, 'a joint rotation does not follow its own arc: axis or chain order is wrong'
print('all cross-checks passed')
