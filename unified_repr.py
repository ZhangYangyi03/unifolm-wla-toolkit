# -*- coding: utf-8 -*-
# unified_repr.py - UnifoLM-WLA-1.0 unified action/state representation.
# Implements "Robot Action, State, and Statistics Processing Specification"
# (unitreerobotics/unifolm-wla, docs/robot_action_state_processing_en.md).
# Pure numpy. Works on a data-prep machine with no GPU; the output npz/json is
# what a post-train job on AutoDL consumes.
import json, math
import numpy as np

D_A = 54          # unified action dim
D_S = 60          # unified state dim
EPS = 1e-8
SCALE_FLOOR = 1e-6

ACTION_SLICES = {
    'left_ee':      (0, 6),   'left_gripper': (6, 7),   'left_hand':  (7, 13),
    'right_ee':     (13, 19), 'right_gripper':(19, 20), 'right_hand': (20, 26),
    'waist':        (26, 29), 'torso':        (29, 32),
    'base_vel':     (32, 34), 'base_yaw':     (34, 35),
    'base_pose':    (35, 41), 'height':       (41, 42),
    'left_leg':     (42, 48), 'right_leg':    (48, 54),
}
STATE_SLICES = {
    'left_ee':      (0, 9),   'left_gripper': (9, 10),  'left_hand':  (10, 16),
    'right_ee':     (16, 25), 'right_gripper':(25, 26), 'right_hand': (26, 32),
    'waist':        (32, 35), 'torso':        (35, 38),
    'base_vel':     (38, 40), 'base_yaw':     (40, 41),
    'base_inertial':(41, 47), 'height':       (47, 48),
    'left_leg':     (48, 54), 'right_leg':    (54, 60),
}
# slice -> canonical fill width actually used (>=1) for the mask
POSE_MODULES = ('left_ee', 'right_ee', 'base_pose')

def _slots(table, name):
    if name not in table:
        raise KeyError('unknown module %r' % name)
    return table[name]


# ---------------- SO(3) / SE(3) ----------------

def hat(w):
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]], dtype=np.float64)

def so3_exp(phi):
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    th = float(np.linalg.norm(phi))
    if th < 1e-12:
        # small-angle expansion, second order
        W = hat(phi)
        return np.eye(3) + W + 0.5 * (W @ W)
    u = phi / th
    W = hat(u)
    return np.eye(3) + math.sin(th) * W + (1.0 - math.cos(th)) * (W @ W)

def so3_log(R):
    # robust log; returns rotation vector, norm == angle
    R = np.asarray(R, dtype=np.float64)
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        # near identity: vee of skew part
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if abs(math.pi - th) < 1e-6:
        # angle near pi: use the symmetric part, sign from skew
        A = 0.5 * (R + np.eye(3))
        d = np.clip(np.diag(A), 0.0, None)
        k = int(np.argmax(d))
        u = np.zeros(3)
        u[k] = math.sqrt(max(d[k], 0.0))
        for i in range(3):
            if i != k:
                u[i] = A[i, k] / (2.0 * u[k]) if u[k] > EPS else 0.0
        n = np.linalg.norm(u)
        if n > EPS:
            u = u / n
        s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if float(np.dot(s, u)) < 0:
            u = -u
        return th * u
    return (th / (2.0 * math.sin(th))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

def rot_rpy(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx          # fixed-axis xyz convention: R = Rz(y)Ry(p)Rx(r)

def quat_to_R(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < EPS:
        raise ValueError('zero quaternion')
    qx, qy, qz, qw = q / n
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),   1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)

def pose_to_SE3(pose, fmt):
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    p = pose[:3]
    if fmt == 'rpy':
        R = rot_rpy(pose[3:6])
    elif fmt == 'quat':
        R = quat_to_R(pose[3:7])          # [qx,qy,qz,qw]
    elif fmt == 'rotvec':
        R = so3_exp(pose[3:6])
    else:
        raise ValueError('pose_format must be rpy|quat|rotvec, got %r' % fmt)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def se3_inverse(T):
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti

def rotation6d(R):
    # first two columns, row-major: [R00,R10,R20, R01,R11,R21]
    return np.array([R[0, 0], R[1, 0], R[2, 0], R[0, 1], R[1, 1], R[2, 1]])

def rotation6d_to_R(r6):
    a1 = np.asarray(r6[:3], dtype=np.float64)
    a2 = np.asarray(r6[3:], dtype=np.float64)
    n1 = np.linalg.norm(a1)
    if n1 < EPS:
        raise ValueError('degenerate rotation-6D first column')
    b1 = a1 / n1
    t = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(t)
    if n2 < EPS:
        raise ValueError('degenerate rotation-6D second column')
    b2 = t / n2
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


# ---------------- unified action / state ----------------

def relative_action(T_cur, T_fut):
    # a_rel = [R_t^T (p_f - p_t), Log(R_t^T R_f)] in R^6
    Rel = se3_inverse(T_cur) @ T_fut
    return np.concatenate([Rel[:3, 3], so3_log(Rel[:3, :3])])

def absolute_pose_from_relative(T_cur, a6):
    T = np.eye(4)
    T[:3, :3] = so3_exp(a6[3:6])
    T[:3, 3] = a6[:3]
    return T_cur @ T

def encode_action_sample(cur, fut, cfg):
    # cur: {'left_ee': pose..., 'left_gripper': float, ...}
    # fut: {'left_ee': (H,6|7) pose chunk, 'left_gripper': (H,), ...}
    H = cfg['chunk_size']
    a = np.zeros((H, D_A), dtype=np.float64)
    m = np.zeros(D_A, dtype=np.float64)
    fmt = cfg.get('pose_format', 'rotvec')
    for name in cfg['modules']:
        lo, hi = _slots(ACTION_SLICES, name)
        width = hi - lo
        if name in POSE_MODULES:
            if name not in cur or name not in fut:
                continue
            T_cur = pose_to_SE3(cur[name], fmt)
            futv = np.asarray(fut[name], dtype=np.float64)
            if futv.ndim != 2 or futv.shape[1] < 6:
                raise ValueError('%s future pose chunk must be (H,>=6)' % name)
            if futv.shape[0] != H:
                raise ValueError('%s chunk has %d steps, cfg says %d' % (name, futv.shape[0], H))
            for k in range(H):
                a[k, lo:lo + 6] = relative_action(T_cur, pose_to_SE3(futv[k, :6], fmt))
            m[lo:lo + 6] = 1.0
        else:
            if name not in fut:
                continue
            futv = np.asarray(fut[name], dtype=np.float64).reshape(H, -1)
            if futv.shape[1] < width:
                raise ValueError('%s provides %d dims, unified slot %s needs %d'
                                 % (name, futv.shape[1], name, width))
            a[:, lo:hi] = futv[:, :width]      # wider input: keep leading components
            m[lo:hi] = 1.0
    return a, m

def encode_state_sample(cur, cfg):
    s = np.zeros(D_S, dtype=np.float64)
    m = np.zeros(D_S, dtype=np.float64)
    fmt = cfg.get('pose_format', 'rotvec')
    for name in cfg['modules']:
        lo, hi = _slots(STATE_SLICES, name)
        if name in ('left_ee', 'right_ee'):
            if name not in cur:
                continue
            T = pose_to_SE3(cur[name], fmt)
            s[lo:lo + 3] = T[:3, 3]
            s[lo + 3:lo + 9] = rotation6d(T[:3, :3])
            m[lo:lo + 9] = 1.0
        elif name == 'base_inertial':
            if 'R_WB' not in cur or 'angular_velocity' not in cur:
                continue
            R = np.asarray(cur['R_WB'], dtype=np.float64).reshape(3, 3)
            g = R.T @ np.array([0.0, 0.0, -1.0])
            g = g / max(np.linalg.norm(g), EPS)
            w = np.asarray(cur['angular_velocity'], dtype=np.float64).reshape(3)
            s[lo:lo + 3] = g
            s[lo + 3:lo + 6] = w                 # raw; normalized by its own stats later
            m[lo:lo + 6] = 1.0
        elif name in ('base_vel', 'base_yaw', 'height'):
            key = {'base_vel': 'base_vel', 'base_yaw': 'base_yaw', 'height': 'height'}[name]
            if key not in cur:
                continue
            v = np.asarray(cur[key], dtype=np.float64).reshape(-1)
            width = hi - lo
            s[lo:hi] = v[:width]
            m[lo:hi] = 1.0
        else:
            if name not in cur:
                continue
            v = np.asarray(cur[name], dtype=np.float64).reshape(-1)
            width = hi - lo
            if v.shape[0] < width:
                raise ValueError('%s state provides %d dims, slot needs %d' % (name, v.shape[0], width))
            s[lo:hi] = v[:width]
            m[lo:hi] = 1.0
    return s, m


# ---------------- global relative statistics ----------------

REL_KEYS = {'left_ee': 'left_relative_ee_pose',
            'right_ee': 'right_relative_ee_pose',
            'base_pose': 'relative_base_pose'}

def collect_global_relative_statistics(chunks):
    # chunks: list of (N,H,6) arrays already in relative form
    X = np.concatenate([np.asarray(c, dtype=np.float64).reshape(-1, 6) for c in chunks], axis=0)
    return {
        'global_max': np.max(X, axis=0),
        'global_min': np.min(X, axis=0),
        'global_q01': np.quantile(X, 0.01, axis=0),
        'global_q99': np.quantile(X, 0.99, axis=0),
        'global_mean': np.mean(X, axis=0),
        'global_std': np.std(X, axis=0),       # population std
        'count': int(X.shape[0]),
    }

def ordinary_statistics(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    return {
        'min': np.min(X, axis=0), 'max': np.max(X, axis=0),
        'q01': np.quantile(X, 0.01, axis=0), 'q99': np.quantile(X, 0.99, axis=0),
        'mean': np.mean(X, axis=0), 'std': np.std(X, axis=0),
        'count': int(X.shape[0]),
    }

def stats_to_json(st):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in st.items()}

def stats_from_json(d):
    return {k: (np.asarray(v, dtype=np.float64) if isinstance(v, list) else v) for k, v in d.items()}


# ---------------- normalization (spec sec. 9/10/11) ----------------

def _affine(o, c):
    o = np.asarray(o, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64).copy()
    c[np.abs(c) < SCALE_FLOOR] = 1.0        # sec. 9.1 scale protection
    return o, c

def affine_params(st, kind):
    """Return (offset, scale) from one statistics dict.
    kind in {'minmax_q','zscore','minmax'} -- spec sec. 9.2-9.4 priority order."""
    if not st:
        return None
    g = lambda *ks: next((st[k] for k in ks if st.get(k) is not None), None)
    if kind == 'minmax_q':
        l = g('global_q01', 'q01'); h = g('global_q99', 'q99')
        if l is not None and h is not None:
            return _affine((l + h) / 2.0, (h - l) / 2.0)
        l = g('global_min', 'min'); h = g('global_max', 'max')
        if l is not None and h is not None:
            return _affine((l + h) / 2.0, (h - l) / 2.0)
        return None
    if kind == 'zscore':
        mu = g('global_mean', 'mean'); sd = g('global_std', 'std')
        if mu is not None and sd is not None:
            return _affine(mu, sd)
        return None
    if kind == 'minmax':
        l = g('global_min', 'min'); h = g('global_max', 'max')
        if l is not None and h is not None:
            return _affine((l + h) / 2.0, (h - l) / 2.0)
        return None
    raise ValueError('kind must be minmax_q|zscore|minmax')

def normalize(x, params):
    if params is None:
        return np.asarray(x, dtype=np.float64).copy()
    o, c = params
    return (np.asarray(x, dtype=np.float64) - o) / c

def denormalize(xn, params):
    if params is None:
        return np.asarray(xn, dtype=np.float64).copy()
    o, c = params
    return np.asarray(xn, dtype=np.float64) * c + o

def binarize_gripper(normalized):
    """Spec sec. 14: definite >0.9 -> 1, <-0.9 -> 0, backward fill from the tail."""
    g = np.asarray(normalized, dtype=np.float64).reshape(-1)
    out = np.zeros(len(g), dtype=np.float64)
    carry = 1.0 if g[-1] > 0 else 0.0
    for k in range(len(g) - 1, -1, -1):
        if g[k] > 0.9:
            carry = 1.0
        elif g[k] < -0.9:
            carry = 0.0
        out[k] = carry
    return out


# ---------------- resampling (spec sec. 15) ----------------

def resample_linear(A, src_fps, tgt_fps):
    """A: (N_s, d) -> (tgt_fps, d); nearest endpoint outside the source range."""
    A = np.asarray(A, dtype=np.float64)
    Ns = A.shape[0]
    src = np.arange(Ns) / float(src_fps)
    tgt = np.arange(tgt_fps) / float(tgt_fps)
    out = np.empty((tgt_fps, A.shape[1]), dtype=np.float64)
    for j, t in enumerate(tgt):
        if t <= src[0]:
            out[j] = A[0]
        elif t >= src[-1]:
            out[j] = A[-1]
        else:
            i = int(np.searchsorted(src, t, side='right') - 1)
            w = (t - src[i]) / (src[i + 1] - src[i])
            out[j] = (1 - w) * A[i] + w * A[i + 1]
    return out

def bspline_basis(t, degree, knots):
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    n = len(knots) - degree - 1
    B = np.zeros((len(t), n))
    for i in range(n):
        B[:, i] = _cox_de_boor(t, degree, knots, i)
    return B

def _cox_de_boor(t, degree, knots, i):
    if degree == 0:
        out = np.zeros_like(t)
        for k in range(len(t)):
            if (knots[i] <= t[k] < knots[i + 1]) or (t[k] == knots[-1] and knots[i] <= t[k] <= knots[i + 1]):
                out[k] = 1.0
        return out
    left = np.zeros_like(t); right = np.zeros_like(t)
    d1 = knots[i + degree] - knots[i]
    d2 = knots[i + degree + 1] - knots[i + 1]
    if d1 > 0:
        left = (t - knots[i]) / d1 * _cox_de_boor(t, degree - 1, knots, i)
    if d2 > 0:
        right = (knots[i + degree + 1] - t) / d2 * _cox_de_boor(t, degree - 1, knots, i + 1)
    return left + right

def resample_bspline(A, src_fps, tgt_fps, degree=3, lam=1e-9):
    """Spec sec. 15.2. A: (2*fps+1, d) two-second context, returns the first second."""
    A = np.asarray(A, dtype=np.float64)
    Ns = A.shape[0]
    K = max(4, int(Ns // 2) + 1)
    n_knots = K + degree + 1
    knots = np.linspace(0.0, (n_knots - 1) / float(n_knots - 1 - degree) if False else 1.0, n_knots)
    knots = np.concatenate([np.zeros(degree), np.linspace(0, 1, K - degree + 1), np.ones(degree)])
    ts = np.arange(Ns) / float(src_fps)
    tt = np.arange(tgt_fps) / float(tgt_fps)
    Bs = bspline_basis(ts, degree, knots)
    Bt = bspline_basis(tt, degree, knots)
    W = Bt @ np.linalg.solve(Bs.T @ Bs + lam * np.eye(Bs.shape[1]), Bs.T)
    return W @ A


# ---------------- statistics merging (spec sec. 12/13) ----------------

def merge_ordinary(stats_list, weights=None):
    """Equal-weight merge of ordinary statistics (spec sec. 12.1).

    Spec sec. 12.1 is explicit: every task has the same weight, and the number of
    raw samples in a task does not change its merge weight. The default weight is
    therefore 1.0 per group, not the group's sample count. Passing `weights`
    overrides that, which is the escape hatch for when training does sample
    uniformly over raw frames instead of uniformly over tasks.
    """
    stats_list = [s for s in stats_list if s]
    if not stats_list:
        return {}
    if weights is None:
        W = np.full(len(stats_list), 1.0 / len(stats_list))
    else:
        W = np.asarray(weights, dtype=np.float64)
        W = W / W.sum()

    means = np.stack([np.asarray(s['mean'], dtype=np.float64) for s in stats_list])
    mu = np.sum([w * m for w, m in zip(W, means)], axis=0)

    out = {'mean': mu}
    for k in ('min', 'max', 'q01', 'q99'):
        vals = [np.asarray(s[k], dtype=np.float64) for s in stats_list if k in s]
        if not vals:
            continue
        if k == 'min':
            out[k] = np.min(np.stack(vals), axis=0)
        elif k == 'max':
            out[k] = np.max(np.stack(vals), axis=0)
        else:
            out[k] = np.sum([w * v for w, v in zip(W, vals)], axis=0)
    if all('std' in s for s in stats_list):
        # law of total variance, equal group weights: pooled var = mean(var_i) + var(mean_i)
        within = np.sum([w * np.asarray(s['std'], dtype=np.float64) ** 2
                         for w, s in zip(W, stats_list)], axis=0)
        between = np.sum([w * (m - mu) ** 2 for w, m in zip(W, means)], axis=0)
        out['std'] = np.sqrt(within + between)
    out['count'] = int(sum(s.get('count', 0) for s in stats_list))
    return out


def merge_relative(stats_list, weights=None):
    """Merge relative-action global statistics (spec sec. 12.1: equal task weight).

    Same rule as merge_ordinary: one statistics group per task, same weight per
    task. The count carried along is informational; it is not the weight.
    """
    stats_list = [s for s in stats_list if s]
    if not stats_list:
        return {}
    if weights is None:
        W = np.full(len(stats_list), 1.0 / len(stats_list))
    else:
        W = np.asarray(weights, dtype=np.float64)
        W = W / W.sum()
    means = np.stack([np.asarray(s['global_mean'], dtype=np.float64) for s in stats_list])
    mu = np.sum([w * m for w, m in zip(W, means)], axis=0)
    within = np.sum([w * np.asarray(s['global_std'], dtype=np.float64) ** 2
                     for w, s in zip(W, stats_list)], axis=0)
    between = np.sum([w * (m - mu) ** 2 for w, m in zip(W, means)], axis=0)
    return {
        'global_mean': mu,
        'global_std': np.sqrt(within + between),
        'global_min': np.min(np.stack([s['global_min'] for s in stats_list]), axis=0),
        'global_max': np.max(np.stack([s['global_max'] for s in stats_list]), axis=0),
        'global_q01': np.sum([w * s['global_q01'] for w, s in zip(W, stats_list)], axis=0),
        'global_q99': np.sum([w * s['global_q99'] for w, s in zip(W, stats_list)], axis=0),
        'count': int(sum(s.get('count', 0) for s in stats_list)),
    }


def merge_by_task(per_task_stats, weights=None):
    """Spec sec. 12.1 end to end: {task_name: stats} -> one merged statistics dict.

    Two tasks may be merged only if they are the same robot embodiment with the
    same module definitions, frames, units and component ordering. This function
    cannot check that for you, so it takes the caller's word and merges; the
    grouping decision stays visible in the code that calls it.
    """
    tasks = [k for k in per_task_stats if per_task_stats[k]]
    if not tasks:
        return {'relative': {}, 'ordinary': {}}
    rel_keys = set().union(*[set(per_task_stats[t].get('relative', {})) for t in tasks])
    ord_keys = set().union(*[set(per_task_stats[t].get('ordinary', {})) for t in tasks])
    w = None if weights is None else [weights[t] for t in tasks]
    return {
        'relative': {k: merge_relative([per_task_stats[t].get('relative', {}).get(k) for t in tasks], w)
                     for k in rel_keys},
        'ordinary': {k: merge_ordinary([per_task_stats[t].get('ordinary', {}).get(k) for t in tasks], w)
                     for k in ord_keys},
        'merged_tasks': tasks,
        'merge_rule': 'equal task weight (spec sec. 12.1)',
    }


def check_roundtrip(T_cur, T_fut, tol=1e-9):
    """relative_action -> absolute_pose_from_relative must recover T_fut."""
    a = relative_action(T_cur, T_fut)
    back = absolute_pose_from_relative(T_cur, a)
    return float(np.max(np.abs(back - T_fut))), a

def check_state_roundtrip(R):
    r6 = rotation6d(R)
    return float(np.max(np.abs(rotation6d_to_R(r6) - R)))
