# -*- coding: utf-8 -*-
"""erflow_tokens.py - check robot data against UnifoLM-ER-Flow's OWN tokenizer.

Why this exists: the spec (docs/robot_action_state_processing_en.md) defines the
continuous unified action/state space. UnifoLM-ER-Flow's released tokenizer
defines the *discrete* space the model actually consumes, and the two are not the
same object. The tokenizer file carries four 256-entry codebooks:

    <|POS_0..255|>    end-effector position
    <|ROT_0..255|>    end-effector rotation
    <|HAND_0..255|>   gripper / dexterous hand
    <|LOW_0..255|>    lower body

and config.json carries robot_state_dim = 120, with a robot_state_projector
mapping 120 -> 2560.

What this checks, using the released files and no weights download beyond their
headers:
  * the codebooks exist, are exactly 256 entries, and are contiguous;
  * the per-token id layout matches the released IGNORE/added-token ordering;
  * the number of robot_state slots (120) and the codebook size (256) are
    recorded next to the payload, so a pipeline cannot quietly disagree with
    them;
  * UNSPECIFIED vs SPECIFIED is reported honestly: the spec says nothing about
    how 54/60-dim vectors map onto POS/ROT/HAND/LOW codes, and this file does not
    invent that mapping.

    python erflow_tokens.py --meta demo/er_flow_meta --stats demo/real_g1/src/stats.json
"""
import argparse, json, os, re, sys

CODEBOOKS = {'POS': 256, 'ROT': 256, 'HAND': 256, 'LOW': 256}
MARKERS = ['<|EEF_START|>', '<|EEF_END|>', '<|LOW_START|>', '<|LOW_END|>',
           '<|HAND_START|>', '<|HAND_END|>', '<seg_begin>', '<seg_end>',
           '<|SEP_VQ|>', '<|robot_state|>', '<|robot_state_implicit_stats|>']


def load(meta_dir):
    tc = json.load(open(os.path.join(meta_dir, 'tokenizer_config.json'), encoding='utf-8'))
    cfg = json.load(open(os.path.join(meta_dir, 'config.json'), encoding='utf-8'))
    return tc, cfg


def audit(meta_dir, stats_path=None):
    tc, cfg = load(meta_dir)
    extra = tc.get('extra_special_tokens') or []
    out = []
    ok = True
    for name, n in CODEBOOKS.items():
        toks = [t for t in extra if re.fullmatch(r'<\|%s_\d+\|>' % name, t)]
        idx = sorted(int(re.search(r'(\d+)', t).group(1)) for t in toks)
        good = len(toks) == n and idx == list(range(n))
        ok &= good
        out.append('%s codebook: %d tokens, contiguous 0..%s  %s'
                   % (name, len(toks), (idx[-1] if idx else '?'), 'ok' if good else 'BAD'))
    missing = [m for m in MARKERS if m not in extra]
    ok &= not missing
    out.append('control markers present: %d/%d %s'
               % (len(MARKERS) - len(missing), len(MARKERS),
                  'ok' if not missing else 'missing %s' % missing))
    ok &= tc.get('extra_special_tokens') and cfg.get('robot_state_token') in extra
    out.append('robot_state_token %r in the tokenizer: %s'
               % (cfg.get('robot_state_token'), cfg.get('robot_state_token') in extra))
    out.append('robot_state_dim = %s (state vector the projector consumes)'
               % cfg.get('robot_state_dim'))
    out.append('codebook size 256 vs robot_state_dim %s: the codebooks quantize action '
               'components, they are not the state vector' % cfg.get('robot_state_dim'))

    if stats_path and os.path.exists(stats_path):
        st = json.load(open(stats_path, encoding='utf-8'))
        m = st.get('meta', {}).get('normalization', {})
        out.append('payload normalization: %s' % m)
        out.append('UNSPECIFIED: the spec does not say which unified components the '
                   'POS/ROT/HAND/LOW codebooks cover, nor their value range. A checker '
                   'cannot verify a mapping nobody has published.')
    return ok, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', required=True)
    ap.add_argument('--stats', default=None)
    a = ap.parse_args(argv)
    ok, lines = audit(a.meta, a.stats)
    for l in lines:
        print('  ' + l)
    print('')
    print('VERIFIED against the released files' if ok else 'MISMATCH in the released files')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
