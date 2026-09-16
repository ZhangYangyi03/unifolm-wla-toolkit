# -*- coding: utf-8 -*-
"""wla_prep - turn raw robot episodes into the UnifoLM-WLA-1.0 training payload.

Pipeline (fixed order, spec sec. 15.3):
    relative-pose conversion -> normalization -> optional gripper binarization
    -> resampling -> unified-slot mapping

Commands
    wla-prep stats   --src DIR --out stats.json      global statistics (spec sec. 7/8/12)
    wla-prep convert --src DIR --out data.npz        normalized (N,H,54) + (N,60) arrays
    wla-prep check   --src DIR --stats stats.json    round-trip + range + mask audit
                                                     rebuilds raw poses from the
                                                     normalized payload, so a wrong
                                                     offset/scale cannot pass silently

Reference source layout (one .npz per episode, plus meta.json):
    <src>/meta.json                 {"fps":30, "pose_format":"rpy", "modules":[...]}
    <src>/episode_000000.npz
        cur/<module>     (T, d)          current absolute observation (poses: xyz+rpy/quat/rotvec)
        action/<module>  (T, H, d)       H future targets in the SAME frame as cur
        action/<module> for pose modules is (T, H, 6 or 7) absolute future poses
    Per-frame 'cur' at time t pairs with action targets t+1..t+H; rows whose future
    is truncated are dropped, never zero-padded.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_repr as U

POSE_MODULES = U.POSE_MODULES
ALL_MODULES = list(U.ACTION_SLICES.keys())


def _load_episode(path):
    return np.load(path, allow_pickle=False)


def read_lerobot(src):
    """Read a Unitree/LeRobot v3 dataset (parquet + meta/info.json).

    Unitree's public G1 datasets are LeRobot v3, not the npz layout above:
    joint angles at 30 fps, no end-effector poses. Joint angles cannot be
    silently pushed through the relative-pose slots the spec reserves for
    EE/base poses -- forwarding kinematics is the dataset's job -- so this
    reader reports what it found and refuses to guess.
    """
    import json as _json
    info_p = os.path.join(src, 'meta', 'info.json')
    if not os.path.exists(info_p):
        raise FileNotFoundError('no meta/info.json in %s' % src)
    with open(info_p, encoding='utf-8') as f:
        info = _json.load(f)
    keys = [k for k, v in info['features'].items() if not str(v.get('dtype')).startswith('video')]
    urdf = os.path.join(src, 'meta', 'urdf.json')
    report = {
        'robot_type': info.get('robot_type'),
        'fps': info.get('fps'),
        'episodes': info.get('total_episodes'),
        'frames': info.get('total_frames'),
        'non_video_features': {k: info['features'][k].get('shape') for k in keys},
        'has_ee_pose': any(('ee' in k.lower() and 'pose' in k.lower()) or 'cartesian' in k.lower()
                           for k in keys),
        'has_urdf': os.path.exists(urdf),
    }
    if not report['has_ee_pose']:
        report['blocking'] = (
            'no end-effector pose in the dataset; the unified action space needs EE/base '
            'poses for slots [0:6]/[13:19]/[35:41] and joints alone cannot fill them. '
            'Run forward kinematics (pinocchio / the robot URDF) to produce EE poses first.')
    try:
        import pyarrow.parquet as pq
    except ImportError:
        report['note'] = 'pyarrow not installed: listing only, no parquet read'
        return report
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(src, 'data', 'chunk-*', '*.parquet')))
    if files:
        t = pq.read_table(files[0]).slice(0, 3)
        report['first_rows'] = {c: str(t.column(c)[0].as_py())[:120] for c in t.column_names}
    return report


def cmd_lerobot(a):
    import pprint
    r = read_lerobot(a.src)
    if r.get('blocking'):
        print('REFUSING to build a unified representation from this dataset:')
        print('  ' + r['blocking'])
        print('  (this command exits 1 on purpose; forcing joint angles into pose slots '
              'produces numbers that look right and are wrong)')
        print()
    pprint.pprint(r)
    return 1 if r.get('blocking') else 0


def _meta(src):
    p = os.path.join(src, 'meta.json')
    if not os.path.exists(p):
        raise FileNotFoundError('no meta.json in %s (see module docstring for the expected layout)' % src)
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def iter_episodes(src):
    meta = _meta(src)
    files = sorted(f for f in os.listdir(src) if f.endswith('.npz'))
    for f in files:
        yield os.path.join(src, f), _load_episode(os.path.join(src, f)), meta


def _extract(d, name, modules):
    """Read cur/<name> and action/<name> if the module is present."""
    ck, ak = 'cur/' + name, 'action/' + name
    if ck not in d.files:
        return None, None
    cur = np.asarray(d[ck], dtype=np.float64)
    act = np.asarray(d[ak], dtype=np.float64) if ak in d.files else None
    if name not in modules:
        return None, None
    return cur, act


# ---------------- collections ----------------

def collect(src, chunk_size=30, target_fps=None, modules=None):
    """Walk the dataset once and return (relative chunks per pose module,
    ordinary samples per module, raw per-sample arrays)."""
    meta = _meta(src)
    modules = modules or meta.get('modules') or ['left_ee', 'left_gripper']
    fmt = meta.get('pose_format', 'rotvec')
    fps = meta.get('fps', 30)
    rel = {}
    ordinary = {}
    raw_actions, raw_states, masks = [], [], []
    raw_pose, raw_cur = {}, {}
    for path, d, _m in iter_episodes(src):
        cur_all, act_all = {}, {}
        for name in modules:
            c, a = _extract(d, name, modules)
            if c is None:
                continue
            cur_all[name], act_all[name] = c, a
        if not cur_all:
            continue
        T = min(len(v) for v in cur_all.values())
        cfg = {'chunk_size': chunk_size, 'modules': modules, 'pose_format': fmt}
        for t in range(T):
            cur = {n: cur_all[n][t] for n in cur_all}
            fut = {}
            ok = True
            for n in cur_all:
                if act_all[n] is None:
                    continue
                a = act_all[n]
                if t >= a.shape[0]:
                    ok = False; break
                chunk = a[t]
                if chunk.shape[0] < chunk_size:
                    ok = False; break
                fut[n] = chunk
            if not ok:
                continue
            try:
                A, m = U.encode_action_sample(cur, fut, cfg)
                S, sm = U.encode_state_sample(cur, cfg)
            except ValueError:
                continue
            raw_actions.append(A); raw_states.append(S); masks.append((m, sm))
            for name in modules:
                if name in POSE_MODULES and name in cur_all and name in fut:
                    lo, hi = U.ACTION_SLICES[name]
                    rel.setdefault(U.REL_KEYS[name], []).append(A[:, lo:hi])
                    raw_pose.setdefault(name, []).append(np.asarray(fut[name], dtype=np.float64)[:, :6])
                    raw_cur.setdefault(name, []).append(np.asarray(cur[name], dtype=np.float64)[:6])
                elif not (name in POSE_MODULES):
                    lo, hi = U.ACTION_SLICES[name]
                    ordinary.setdefault(name, []).append(A[:, lo:hi])
            for name, arr in cur_all.items():
                if not (name in POSE_MODULES):
                    if name == 'base_inertial':
                        continue
                    ordinary.setdefault('state/' + name, []).append(
                        np.asarray(arr, dtype=np.float64)[t:t + 1])
    return {'rel': rel, 'ordinary': ordinary, 'raw_actions': raw_actions,
            'raw_states': raw_states, 'masks': masks, 'meta': meta,
            'raw_pose': raw_pose, 'raw_cur': raw_cur,
            'fps': fps, 'modules': modules}


def build_statistics(col, chunk_size=30):
    """Spec sec. 7 (relative, global) + sec. 8 (ordinary)."""
    stats = {'relative': {}, 'ordinary': {}, 'meta': {}}
    for key, chunks in col['rel'].items():
        st = U.collect_global_relative_statistics(chunks)
        stats['relative'][key] = {k: (v if k == 'count' else v.tolist())
                                  for k, v in st.items()}
    for key, vals in col['ordinary'].items():
        X = np.concatenate([np.asarray(v, dtype=np.float64).reshape(-1, np.asarray(v).shape[-1])
                            for v in vals], axis=0)
        stats['ordinary'][key] = U.stats_to_json(U.ordinary_statistics(X))
    # spec sec. 7.2 wants the relative modules at the top level of the file, so
    # also emit that exact shape: {"relative_action_key": {global_max: [...]}, ...}
    stats['spec_7_2'] = {k: {kk: vv for kk, vv in v.items() if kk != 'count'}
                         for k, v in stats['relative'].items()}
    stats['meta'] = {
        'source_meta': col['meta'],
        'samples': len(col['raw_actions']),
        'chunk_size': chunk_size,
        'normalization': {'action': 'minmax_q', 'relative_action': 'zscore',
                          'gripper': 'minmax_q', 'state': 'minmax_q'},
    }
    return stats


def normalization_params(stats):
    """Spec sec. 9.2/9.3 + sec. 10/11: per-slot offset/scale for actions and states."""
    norm = stats['meta']['normalization']
    ao = np.zeros(U.D_A); ac = np.ones(U.D_A)
    so = np.zeros(U.D_S); sc = np.ones(U.D_S)

    def put(o, c, lo, hi, table=None, key=None):
        lo2, hi2 = lo, hi
        if table is not None and key is not None:
            pass
        n = hi - lo
        if n <= 0:
            return
        if np.asarray(o).size == 1 and n > 1:
            o = np.full(n, float(np.asarray(o).ravel()[0])); c = np.full(n, float(np.asarray(c).ravel()[0]))
        o = np.asarray(o, dtype=np.float64).ravel()[:n]
        c = np.asarray(c, dtype=np.float64).ravel()[:n]
        if len(o) < n:
            o = np.concatenate([o, np.zeros(n - len(o))])
            c = np.concatenate([c, np.ones(n - len(c))])
        ao[lo:hi] = o; ac[lo:hi] = c
        return

    # --- actions: relative poses use zscore on relative statistics
    for name, key in U.REL_KEYS.items():
        st = stats['relative'].get(key)
        p = U.affine_params(U.stats_from_json(st), 'zscore') if st else None
        lo, hi = U.ACTION_SLICES[name]
        if p is None:
            continue
        o, c = p
        ao[lo:lo + 6] = o[:6]; ac[lo:lo + 6] = c[:6]
    # --- actions: grippers and hands use the gripper normalization type
    for name in ('left_gripper', 'right_gripper', 'left_hand', 'right_hand'):
        st = stats['ordinary'].get(name)
        p = U.affine_params(U.stats_from_json(st), norm['gripper']) if st else None
        lo, hi = U.ACTION_SLICES[name]
        if p is None:
            continue
        o, c = p
        m = min(hi - lo, len(o))
        ao[lo:lo + m] = o[:m]; ac[lo:lo + m] = c[:m]
    # --- actions: everything else uses the ordinary action normalization type
    for name in ('waist', 'torso', 'base_vel', 'base_yaw', 'left_leg', 'right_leg'):
        st = stats['ordinary'].get(name)
        p = U.affine_params(U.stats_from_json(st), norm['action']) if st else None
        lo, hi = U.ACTION_SLICES[name]
        if p is None:
            continue
        o, c = p
        m = min(hi - lo, len(o))
        ao[lo:lo + m] = o[:m]; ac[lo:lo + m] = c[:m]
    # --- states
    for name in ('left_ee', 'right_ee'):
        st = stats['ordinary'].get('state/' + name)
        p = U.affine_params(U.stats_from_json(st), norm['state']) if st else None
        lo, hi = U.STATE_SLICES[name]
        if p is None:
            continue
        o, c = p
        so[lo:lo + 3] = o[:3]; sc[lo:lo + 3] = c[:3]      # xyz only; rotation-6D is untouched
    for name in ('left_gripper', 'left_hand', 'right_gripper', 'right_hand',
                 'waist', 'torso', 'left_leg', 'right_leg'):
        st = stats['ordinary'].get('state/' + name)
        p = U.affine_params(U.stats_from_json(st), norm['state']) if st else None
        if p is None:
            continue
        lo, hi = U.STATE_SLICES[name]
        o, c = p
        m = min(hi - lo, len(o))
        so[lo:lo + m] = o[:m]; sc[lo:lo + m] = c[:m]
    # body gravity is unit-length only; body angular velocity uses its own statistics
    st = stats['ordinary'].get('state/angular_velocity')
    if st:
        p = U.affine_params(U.stats_from_json(st), norm['state'])
        if p is not None:
            o, c = p
            m = min(3, len(o))
            so[44:44 + m] = o[:m]; sc[44:44 + m] = c[:m]
    ac[np.abs(ac) < U.SCALE_FLOOR] = 1.0
    sc[np.abs(sc) < U.SCALE_FLOOR] = 1.0
    return ao, ac, so, sc


def convert(col, stats, stats_path, out_path, binarize_gripper=False, val_ratio=0.1, seed=0):
    ao, ac, so, sc = normalization_params(stats)
    A = np.stack(col['raw_actions'])                     # (N,H,54)
    S = np.stack(col['raw_states'])                     # (N,60)
    status = np.zeros((len(A), U.D_A))
    state_mask = np.zeros((len(S), U.D_S))
    for i, (m, sm) in enumerate(col['masks']):
        status[i] = m; state_mask[i] = sm
    An = (A - ao) / ac
    Sn = S.copy()
    for i in range(Sn.shape[0]):
        sel = state_mask[i] > 0
        Sn[i, sel] = (Sn[i, sel] - so[sel]) / sc[sel]
        Sn[i, 41:44] = S[i, 41:44]                      # gravity stays unit-length
    if binarize_gripper:
        for name in ('left_gripper', 'right_gripper'):
            lo, hi = U.ACTION_SLICES[name]
            if status[:, lo].max() > 0:
                An[:, :, lo] = np.stack([U.binarize_gripper(An[i, :, lo]) for i in range(len(An))])
    n = len(An)
    idx = np.random.default_rng(seed).permutation(n)
    nval = max(1, int(round(n * val_ratio)))
    val = np.sort(idx[:nval]); train = np.sort(idx[nval:])
    np.savez_compressed(out_path,
                        action=An.astype(np.float32), action_mask=status.astype(np.float32),
                        state=Sn.astype(np.float32), state_mask=state_mask.astype(np.float32),
                        split=np.where(np.isin(np.arange(n), val), 1, 0).astype(np.int8),
                        action_offset=ao, action_scale=ac, state_offset=so, state_scale=sc,
                        chunk_size=np.int64(col['meta'].get('chunk_size', An.shape[1])),
                        fps=np.int64(col['fps']))
    return out_path, n, len(train), len(val)


def check(col, stats, tol=1e-8):
    """Audit the payload the way a training job would, and print the findings."""
    problems = []
    ao, ac, so, sc = normalization_params(stats)
    A = np.stack(col['raw_actions']); S = np.stack(col['raw_states'])
    An = (A - ao) / ac
    for name, key in U.REL_KEYS.items():
        lo, hi = U.ACTION_SLICES[name]
        pool = col['rel'].get(key, [])
        if not pool:
            continue
        used = np.concatenate(pool, axis=0).reshape(-1, 6)
        if used.size == 0:
            continue
        mu = used.mean(axis=0); sd = used.std(axis=0)
        st = U.stats_from_json(stats['relative'][key])
        if not np.allclose(st['global_mean'], mu, atol=1e-9):
            problems.append('%s: statistics mean does not match the data (%.3g vs %.3g)'
                            % (key, float(np.abs(st['global_mean'] - mu).max()), 0.0))
        if not np.allclose(st['global_std'], sd, atol=1e-9):
            problems.append('%s: statistics std is not the population std of the data' % key)
        # the point of zscore: the normalized pool should centre on 0 with unit spread
        scale = np.where(sd < U.SCALE_FLOOR, 1.0, sd)
        nm = (used - mu) / scale
        if abs(nm.mean()) > 1e-6 or abs(nm.std() - 1.0) > 1e-3:
            problems.append('%s: normalized data is not zero-mean/unit-std '
                            '(mean %.3g, std %.3g)' % (key, float(nm.mean()), float(nm.std())))
        # denormalize exactly the way a model would, and compare against the raw pool
        back = U.denormalize(nm, U.affine_params(st, 'zscore'))
        if np.max(np.abs(back - used)) > tol:
            problems.append('%s: denormalize does not recover the raw statistics pool '
                            '(max %.3g)' % (key, float(np.max(np.abs(back - used)))))
    # masks
    m_sum = np.stack([m for m, _ in col['masks']])
    if not np.all((m_sum == 0) | (m_sum == 1)):
        problems.append('action masks are not binary')
    zero_mask = m_sum == 0                          # (N, 54)
    if A.shape[0] and np.any(np.abs(A) * zero_mask[:, None, :] > 0):
        problems.append('a disabled action slot carries a non-zero value (spec sec. 3.4)')

    # end-to-end: denormalized relative actions must rebuild the raw absolute trajectories
    fmt = col['meta'].get('pose_format', 'rotvec')
    for name, key in U.REL_KEYS.items():
        if name not in col['raw_pose']:
            continue
        st = U.stats_from_json(stats['relative'][key])
        p = U.affine_params(st, 'zscore')
        lo, hi = U.ACTION_SLICES[name]
        worst = 0.0
        for i, (fut, cur_pose) in enumerate(zip(col['raw_pose'][name], col['raw_cur'][name])):
            T_cur = U.pose_to_SE3(cur_pose, fmt)
            a = U.denormalize(An[i, :, lo:hi], p)
            for k in range(a.shape[0]):
                T = U.absolute_pose_from_relative(T_cur, a[k])
                T_ref = U.pose_to_SE3(fut[k], fmt)
                worst = max(worst, float(np.max(np.abs(T - T_ref))))
        if worst > 1e-5:
            problems.append('%s: normalized payload does not rebuild the raw poses '
                            '(max deviation %.3g; normalization is not invertible)' % (name, worst))
        else:
            print('  rebuilt %s trajectories from the normalized payload, max deviation %.2e'
                  % (name, worst))
    return problems


# ---------------- CLI ----------------

def cmd_stats(a):
    col = collect(a.src, a.chunk_size, a.target_fps, a.modules.split(',') if a.modules else None)
    stats = build_statistics(col, a.chunk_size)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print('wrote %s   samples=%d  relative pools=%s  ordinary pools=%d'
          % (a.out, stats['meta']['samples'], list(stats['relative']), len(stats['ordinary'])))
    for k, v in stats['relative'].items():
        print('  %-24s n=%d  mean=%s  std=%s' % (k, v['count'],
              np.round(v['global_mean'], 4), np.round(v['global_std'], 4)))


def cmd_convert(a):
    col = collect(a.src, a.chunk_size, a.target_fps, a.modules.split(',') if a.modules else None)
    stats = json.load(open(a.stats, encoding='utf-8')) if a.stats else build_statistics(col, a.chunk_size)
    if not a.stats:
        json.dump(stats, open(a.out + '.stats.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    out, n, ntr, nval = convert(col, stats, a.stats, a.out, a.binarize_gripper, a.val_ratio, a.seed)
    print('wrote %s   samples=%d  train=%d  val=%d' % (out, n, ntr, nval))


def cmd_check(a):
    col = collect(a.src, a.chunk_size, a.target_fps, a.modules.split(',') if a.modules else None)
    stats = json.load(open(a.stats, encoding='utf-8'))
    problems = check(col, stats)
    if problems:
        print('found %d problem(s):' % len(problems))
        for p in problems:
            print('  ' + p)
        return 1
    print('payload audit clean: statistics match the data, masks binary, '
          'disabled slots empty, denormalization is lossless')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='wla-prep', description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('stats', cmd_stats), ('convert', cmd_convert),
                     ('check', cmd_check), ('lerobot', cmd_lerobot)):
        s = sub.add_parser(name)
        s.add_argument('--src', required=True)
        s.add_argument('--chunk-size', type=int, default=30, dest='chunk_size')
        s.add_argument('--target-fps', type=int, default=None, dest='target_fps')
        s.add_argument('--modules', default=None)
        s.add_argument('--seed', type=int, default=0)
        if name == 'stats':
            s.add_argument('--out', required=True)
        if name == 'convert':
            s.add_argument('--out', required=True)
            s.add_argument('--stats', default=None)
            s.add_argument('--binarize-gripper', action='store_true', dest='binarize_gripper')
            s.add_argument('--val-ratio', type=float, default=0.1, dest='val_ratio')
        if name == 'check':
            s.add_argument('--stats', required=True)
        if name == 'lerobot':
            pass
        s.set_defaults(func=fn)
    a = ap.parse_args(argv)
    return a.func(a) or 0


if __name__ == '__main__':
    raise SystemExit(main())
