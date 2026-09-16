# -*- coding: utf-8 -*-
"""Self-test for the unified action/state representation. No GPU, no data needed.

Every check is an identity that must hold by construction:
  * SE(3) relative action -> absolute pose recovers the target pose
  * rotation-6D -> R recovers R
  * normalization maps the chosen quantiles to -1 / +1
  * masks follow spec sec. 3.4 (unavailable module => zeros and mask 0)
  * linear resampling clamps at the endpoints instead of extrapolating
  * gripper binarization matches the spec pseudocode exactly
"""
import math
import numpy as np
import unified_repr as U

FAIL = []

def check(name, ok, detail=''):
    print(('  PASS  ' if ok else '  FAIL  ') + name + ('   ' + detail if detail else ''))
    if not ok:
        FAIL.append(name)

rng = np.random.default_rng(0)

def rand_R():
    w = rng.normal(size=3); w = w / np.linalg.norm(w) * rng.uniform(0, np.pi)
    return U.so3_exp(w)

print('1. SE(3) relative action round trip')
worst = 0.0
for _ in range(200):
    Tc = np.eye(4); Tc[:3, :3] = rand_R(); Tc[:3, 3] = rng.uniform(-2, 2, 3)
    Tf = np.eye(4); Tf[:3, :3] = rand_R(); Tf[:3, 3] = rng.uniform(-2, 2, 3)
    err, _ = U.check_roundtrip(Tc, Tf)
    worst = max(worst, err)
check('relative -> absolute within 1e-9 over 200 random pairs', worst < 1e-9, 'max err %.2e' % worst)

print('2. rotation-6D (first two columns, row-major) round trip')
worst = max(U.check_state_roundtrip(rand_R()) for _ in range(200))
check('rotation-6D -> R within 1e-9', worst < 1e-9, 'max err %.2e' % worst)

print('3. relative translation is in the CURRENT frame, not world delta')
Tc = np.eye(4); Tc[:3, :3] = U.rot_rpy([0.3, -0.2, 1.1]); Tc[:3, 3] = [0.4, -0.1, 0.9]
Tf = np.eye(4); Tf[:3, :3] = Tc[:3, :3]; Tf[:3, 3] = Tc[:3, 3] + [0.05, 0.0, 0.0]
a = U.relative_action(Tc, Tf)
check('world +x step keeps its length but is re-expressed in the body frame',
      abs(np.linalg.norm(a[:3]) - 0.05) < 1e-12 and abs(a[3:]).max() < 1e-12,
      'p_rel=%s' % np.round(a[:3], 4))

print('4. normalization parameters map the chosen quantiles to -1 / +1')
st = {'q01': np.array([-2.0, 0.0]), 'q99': np.array([2.0, 10.0])}
o, c = U.affine_params(st, 'minmax_q')
check('minmax_q: q01 -> -1 and q99 -> +1',
      np.allclose(U.normalize(st['q01'], (o, c)), -1) and np.allclose(U.normalize(st['q99'], (o, c)), 1))
stz = {'global_mean': np.array([1.0, -1.0]), 'global_std': np.array([0.5, 2.0])}
o, c = U.affine_params(stz, 'zscore')
check('zscore: mean -> 0 and mean+std -> +1',
      np.allclose(U.normalize(stz['global_mean'], (o, c)), 0)
      and np.allclose(U.normalize(stz['global_mean'] + stz['global_std'], (o, c)), 1))
o, c = U.affine_params({'q01': np.array([0.0]), 'q99': np.array([0.0])}, 'minmax_q')
check('degenerate scale is floored to 1 (spec 9.1)', c[0] == 1.0)
check('missing statistics fall back to identity', U.affine_params({}, 'minmax_q') is None)

print('5. gripper binarization matches the spec pseudocode')
g = np.array([-0.2, 0.1, 0.95, 0.3, -0.95, 0.2])
ref = np.zeros(6); carry = 1 if g[-1] > 0 else 0
for k in range(5, -1, -1):
    if g[k] > 0.9: carry = 1
    elif g[k] < -0.9: carry = 0
    ref[k] = carry
check('backward fill from the tail', np.array_equal(U.binarize_gripper(g), ref), str(U.binarize_gripper(g)))

print('6. linear resampling clamps outside the source range, no extrapolation')
A = np.stack([np.arange(3.0), np.arange(3.0) * -1], axis=1)          # 3 points at 10 fps -> [0, 0.2] s
out = U.resample_linear(A, src_fps=10, tgt_fps=10)                    # targets reach 0.9 s
check('shape is (target_fps, d)', out.shape == (10, 2), str(out.shape))
check('beyond the source range the nearest endpoint is held, not extrapolated',
      np.array_equal(out[0], A[0]) and np.array_equal(out[3], A[-1]) and np.array_equal(out[-1], A[-1]),
      'out[3]=%s out[-1]=%s' % (out[3], out[-1]))
A2 = np.stack([np.arange(3.0), np.arange(3.0) * 2], axis=1)           # points 0,1,2 at 10 fps
out2 = U.resample_linear(A2, src_fps=10, tgt_fps=20)                  # t = 0, 0.05, 0.10, ...
check('inside the range it interpolates linearly, halfway between two source points',
      np.allclose(out2[1], (A2[0] + A2[1]) / 2), 'out2[1]=%s' % out2[1])

print('7. unified action encoding, masks and slot layout')
cfg = {'chunk_size': 4, 'modules': ['left_ee', 'left_gripper', 'right_ee', 'right_gripper', 'base_vel'],
       'pose_format': 'rpy'}
H = cfg['chunk_size']
cur = {'left_ee': [0.4, -0.1, 0.9, 0.3, -0.2, 1.1],
       'right_ee': [-0.4, -0.1, 0.9, 0.3, 0.2, -1.1],
       'left_gripper': 0.8, 'right_gripper': 0.8,
       'base_vel': [0.0, 0.0]}
fut = {'left_ee': np.tile([0.45, -0.1, 0.9, 0.3, -0.2, 1.1], (H, 1)),
       'right_ee': np.tile([-0.4, -0.1, 0.9, 0.3, 0.2, -1.1], (H, 1)),
       'left_gripper': np.linspace(0.8, 0.1, H), 'right_gripper': np.linspace(0.8, 0.1, H),
       'base_vel': np.tile([0.1, 0.0], (H, 1))}
a, m = U.encode_action_sample(cur, fut, cfg)
check('action shape is (H, 54)', a.shape == (H, U.D_A), str(a.shape))
check('mask marks exactly the enabled modules',
      m.sum() == 6 + 1 + 6 + 1 + 2, 'sum=%g' % m.sum())
check('disabled modules stay zero', np.all(a[:, U.ACTION_SLICES['waist'][0]:U.ACTION_SLICES['waist'][1]] == 0))
R_cur = U.rot_rpy([0.3, -0.2, 1.1])
# spec 6.1: p_rel = R_t^T (p_{t+k} - p_t) -- the world delta rotated INTO the body frame
expected = R_cur.T @ np.array([0.05, 0.0, 0.0])
check('left ee relative translation is expressed in the current frame',
      np.allclose(a[0, 0:3], expected, atol=1e-12), 'a[0,0:3]=%s expected=%s'
      % (np.round(a[0, 0:3], 5), np.round(expected, 5)))
try:
    U.encode_action_sample(cur, fut, {'chunk_size': H, 'modules': ['left_ee', 'waist'], 'pose_format': 'rpy'})
    check('waist absent from the sample is simply skipped (mask 0)', True)
except Exception as e:
    check('waist absent from the sample is simply skipped (mask 0)', False, str(e))
try:
    bad = dict(fut); bad['waist'] = np.zeros((H, 2))
    U.encode_action_sample(cur, bad, {'chunk_size': H, 'modules': ['waist'], 'pose_format': 'rpy'})
    check('narrow module raises instead of zero-padding (spec 3.4)', False, 'no exception')
except ValueError as e:
    check('narrow module raises instead of zero-padding (spec 3.4)', True, str(e)[:60])

print('8. unified state encoding')
R = U.rot_rpy([0.0, 0.0, 0.5])
cur = {'left_ee': [0.4, -0.1, 0.9, 0.0, 0.0, 0.5], 'left_gripper': 1.0,
       'waist': [0.1, 0.2, 0.3], 'R_WB': R, 'angular_velocity': [0.01, 0.02, 0.03],
       'left_leg': [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]}
s, sm = U.encode_state_sample(cur, {'modules': ['left_ee', 'left_gripper', 'waist', 'base_inertial', 'left_leg'], 'pose_format': 'rpy'})
check('state shape is (60,)', s.shape == (U.D_S,), str(s.shape))
g = s[41:44]
check('gravity direction in body frame is unit length', abs(np.linalg.norm(g) - 1) < 1e-12, '|g|=%.6f' % np.linalg.norm(g))
check('rotation-6D of the state rebuilds the rotation',
      np.max(np.abs(U.rotation6d_to_R(s[3:9]) - R)) < 1e-12)
check('state mask is all ones for the enabled modules', sm.sum() == 9 + 1 + 3 + 6 + 6, 'sum=%g' % sm.sum())

print('9. global relative statistics shapes (spec 7.2)')
chunks = [np.stack([U.relative_action(np.eye(4), np.eye(4)) for _ in range(7)]) for _ in range(3)]
st = U.collect_global_relative_statistics(chunks)
check('every global statistic has shape (6,)',
      all(np.asarray(st[k]).shape == (6,) for k in ('global_max', 'global_min', 'global_q01', 'global_q99', 'global_mean', 'global_std')))
check('population std is used (zero spread on identical samples)', np.allclose(st['global_std'], 0))

print('10. statistics merging is equal-task-weight, not equal-sample-weight (spec 12.1)')
def gaussian_stats(mean, std, n):
    m = np.asarray(mean, dtype=np.float64); sd = np.asarray(std, dtype=np.float64)
    return {'mean': m, 'std': sd, 'min': m - 3 * sd, 'max': m + 3 * sd,
            'q01': m - 2 * sd, 'q99': m + 2 * sd, 'count': n}
# task A: 10 samples around 0, task B: 1000 samples around 10 -- equal task weight
a = gaussian_stats([0.0], [1.0], 10)
b = gaussian_stats([10.0], [1.0], 1000)
mp = U.merge_ordinary([a, b])
check('two tasks of very different sizes get equal weight (mean is 5, not ~9.9)',
      abs(float(mp['mean'][0]) - 5.0) < 1e-12, 'merged mean %.6f' % float(mp['mean'][0]))
check('pooled std follows the law of total variance (sqrt(1 + 25) = 5.099)',
      abs(float(mp['std'][0]) - math.sqrt(26.0)) < 1e-12, 'merged std %.6f' % float(mp['std'][0]))
mw = U.merge_ordinary([a, b], weights=[10, 1000])
check('sample weights are available but must be asked for explicitly',
      float(mw['mean'][0]) > 9.8, 'sample-weighted mean %.4f' % float(mw['mean'][0]))
check('spec routing key is named for what it is',
      U.merge_by_task({'t1': {'ordinary': {'waist': a}}, 't2': {'ordinary': {'waist': b}}})['merge_rule']
      == 'equal task weight (spec sec. 12.1)')

print('')
if FAIL:
    print('%d CHECK(S) FAILED: %s' % (len(FAIL), ', '.join(FAIL)))
    raise SystemExit(1)
print('all checks passed')
