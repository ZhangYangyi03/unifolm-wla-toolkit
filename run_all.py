# -*- coding: utf-8 -*-
"""run_all.py - every check in the toolkit, in order, one exit code.

    python run_all.py

No GPU, no network, no downloads: the real G1 episode in demo/real_g1 was fetched
once and is committed here, so this is repeatable offline.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
STEPS = [
    ('unit geometry + normalization identities', [PY, 'selftest.py']),
    ('URDF FK vs an independent implementation', [PY, 'selftest_urdf.py']),
    ('synthetic data through the whole pipeline', [PY, 'demo/make_demo_full.py', 'demo/data_full']),
    ('statistics', [PY, 'wla_prep.py', 'stats', '--src', 'demo/data_full', '--out', 'demo/data_full/stats.json']),
    ('convert', [PY, 'wla_prep.py', 'convert', '--src', 'demo/data_full', '--out', 'demo/data_full/payload.npz', '--stats', 'demo/data_full/stats.json']),
    ('audit', [PY, 'wla_prep.py', 'check', '--src', 'demo/data_full', '--stats', 'demo/data_full/stats.json']),
    ('training-side contract', [PY, 'demo/train_example.py', 'demo/data_full/payload.npz']),
    # the real episode: joint angles -> FK -> unified layout -> statistics -> payload
    # must exit 1 by design: the real dataset has no end-effector pose, and the
    # reader is supposed to say so instead of guessing. A 0 here is the failure.
    ('real G1: the le-robot reader refuses to guess (expected exit 1)',
     [PY, 'wla_prep.py', 'lerobot', '--src', 'demo/real_g1/dataset']),
    ('real G1: forward kinematics into the source layout',
     [PY, 'unitree_g1_to_unified.py', '--joints', 'demo/real_g1/g1_joints_sample.npz',
      '--urdf', 'demo/g1.urdf', '--out', 'demo/real_g1/src', '--fps', '30']),
    ('real G1: statistics', [PY, 'wla_prep.py', 'stats', '--src', 'demo/real_g1/src', '--out', 'demo/real_g1/src/stats.json']),
    ('real G1: convert', [PY, 'wla_prep.py', 'convert', '--src', 'demo/real_g1/src',
                          '--out', 'demo/real_g1/src/payload.npz', '--stats', 'demo/real_g1/src/stats.json']),
    ('real G1: audit', [PY, 'wla_prep.py', 'check', '--src', 'demo/real_g1/src', '--stats', 'demo/real_g1/src/stats.json']),
    ('real Unitree G1 episode, end to end', [PY, 'selftest_real.py']),
    ('ER-Flow released tokenizer vs the spec layout', [PY, 'erflow_tokens.py', '--meta', 'demo/er_flow_meta',
                                                       '--stats', 'demo/real_g1/src/stats.json']),
    ('ER-Flow token sequence from a real payload', [PY, 'erflow_prompt.py', '--meta', 'demo/er_flow_meta',
                                                    '--payload', 'demo/real_g1/src/payload.npz',
                                                    '--out', 'demo/real_g1/src/erflow_batch.json',
                                                    '--horizon', '2']),
]

EXPECTED_FAIL = ('refuses to guess',)

fails = 0
for name, cmd in STEPS:
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, errors='replace')
    expect_fail = any(k in name for k in EXPECTED_FAIL)
    ok = (p.returncode != 0) if expect_fail else (p.returncode == 0)
    fails += (not ok)
    print('%s  %s' % ('  OK  ' if ok else ' FAIL ', name))
    if not ok:
        print((p.stdout or '')[-1500:])
        print((p.stderr or '')[-1500:])
print('')
print('%d/%d steps passed' % (len(STEPS) - fails, len(STEPS)))
raise SystemExit(1 if fails else 0)
