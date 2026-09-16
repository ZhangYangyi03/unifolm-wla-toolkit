# -*- coding: utf-8 -*-
"""erflow_prompt.py - build an UnifoLM-ER-Flow input from a unifolm-wla-toolkit payload.

The docs spec stops at the continuous 54-dim action / 60-dim state space. ER-Flow
is the released half of the same project, and it consumes something else: a
<|robot_state|> vector of 120 floats, and discrete action codes drawn from four
256-entry codebooks (<|POS_N|>, <|ROT_N|>, <|HAND_N|>, <|LOW_N|>).

Nothing in the released material says how a 54-dim action chunk becomes those
codes. So this file does two separate things and never blurs them:

  1. it builds the parts that ARE determined by the released files -- the token
     ids, the control markers, the state vector width, the codebook sizes;
  2. for the parts that are NOT determined, it uses a named, replaceable default
     (uniform bins over the payload's own minmax_q range) and labels the output
     as a placeholder in the JSON it writes.

Run it and read `"unspecified"` in the output before you train anything on it.
Being wrong here is silent: the token ids will look perfect and the codes will be
meaningless.

    python erflow_prompt.py --meta demo/er_flow_meta --payload demo/real_g1/src/payload.npz \
        --out demo/real_g1/src/erflow_batch.json
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_repr as U  # the spec implementation; slot layout comes from there

CODE_BOOKS = ('POS', 'ROT', 'HAND', 'LOW')


def token_ids(meta_dir):
    """token -> id, straight from the released tokenizer.

    The full tokenizer.json is 11 MB and is not committed; `added_tokens.json` in
    this directory is the 1317-token subset extracted from it, which is all this
    script reads. Drop the real tokenizer.json in here and it is used instead.
    """
    full = os.path.join(meta_dir, 'tokenizer.json')
    if os.path.exists(full):
        tok = json.load(open(full, encoding='utf-8'))
        ids = {t['content']: t['id'] for t in tok.get('added_tokens', [])}
    else:
        ids = json.load(open(os.path.join(meta_dir, 'added_tokens.json'), encoding='utf-8'))
    if not ids:
        raise ValueError('no added tokens found in %s' % meta_dir)
    return ids


def code_tokens(ids, book, code):
    t = '<|%s_%d|>' % (book, int(code))
    if t not in ids:
        raise KeyError('%s is not in the released tokenizer' % t)
    return ids[t]


def uniform_codes(x, s, nbins=256):
    """PLACEHOLDER quantizer: uniform bins over the payload's own minmax_q range.

    Chosen because it is invertible and needs no training. It is NOT the RVQ the
    papers describe -- the released material contains no RVQ weights for the WLA
    action space, so this is the honest stand-in, flagged as such in the output.
    """
    if s['max'] - s['min'] < 1e-12:
        return np.zeros_like(x, dtype=np.int64)
    z = (x - s['min']) / (s['max'] - s['min'])
    return np.clip((z * (nbins - 1)).round(), 0, nbins - 1).astype(np.int64)


def build(meta_dir, payload_path, horizon=1, sample=0, nbins=256):
    ids = token_ids(meta_dir)
    cfg = json.load(open(os.path.join(meta_dir, 'config.json'), encoding='utf-8'))
    z = np.load(payload_path)
    action, mask, state = z['action'], z['action_mask'], z['state']
    t = int(sample)
    a = action[t, :horizon]                      # (H, 54)
    on = mask[t] > 0

    # state: the projector eats 120 floats. The toolkit produces 60. Until the
    # mapping is published, the first 60 slots are filled and the rest stay zero,
    # which is visible in `unspecified` below rather than hidden in the code.
    state_vec = np.zeros(int(cfg['robot_state_dim']), dtype=np.float32)
    state_vec[:min(len(state_vec), state.shape[-1])] = state[t]

    off = np.asarray(z['action_offset'], dtype=np.float64)
    sc = np.asarray(z['action_scale'], dtype=np.float64)

    def qrange(i):
        """PLACEHOLDER range for slot i: global_mean +/- 1 global_std.

        Used because the payload's own zscore statistics bound the value; the
        real training range of each codebook is not published. Labelled as such
        in `unspecified` rather than presented as the truth.
        """
        return float(off[i] - sc[i]), float(off[i] + sc[i])

    def codes_for(i):
        return int(uniform_codes(a[:, i:i + 1].ravel(), {'min': qrange(i)[0], 'max': qrange(i)[1]}, nbins)[0])

    slots = U.ACTION_SLICES
    seq = []
    for side in ('left', 'right'):
        i = slots[side + '_ee'][0]
        for kind, base in (('POS', i), ('ROT', i + 3)):
            for h in range(a.shape[0]):
                seq.append((kind, [int(uniform_codes(a[h, base + k], {'min': qrange(base + k)[0],
                                                                     'max': qrange(base + k)[1]}, nbins))
                                   for k in range(3)]))
    hand_idx = slots['left_hand'][0]
    low_idx = slots['left_leg'][0]

    tok_seq = []

    def add(name):
        tok_seq.append((name, ids[name]))

    add('<|EEF_START|>')
    for h in range(a.shape[0]):
        for k in range(3):
            i = slots['left_ee'][0] + k
            c = int(uniform_codes(a[h, i], {'min': qrange(i)[0], 'max': qrange(i)[1]}, nbins))
            tok_seq.append(('<|POS_%d|>' % c, code_tokens(ids, 'POS', c)))
    add('<|SEP_VQ|>')
    for h in range(a.shape[0]):
        for k in range(3):
            i = slots['left_ee'][0] + 3 + k
            c = int(uniform_codes(a[h, i], {'min': qrange(i)[0], 'max': qrange(i)[1]}, nbins))
            tok_seq.append(('<|ROT_%d|>' % c, code_tokens(ids, 'ROT', c)))
    add('<|EEF_END|>')
    add('<|HAND_START|>')
    for h in range(a.shape[0]):
        for k in range(6):
            i = hand_idx + k
            c = int(uniform_codes(a[h, i], {'min': qrange(i)[0], 'max': qrange(i)[1]}, nbins))
            tok_seq.append(('<|HAND_%d|>' % c, code_tokens(ids, 'HAND', c)))
    add('<|HAND_END|>')
    add('<|LOW_START|>')
    for h in range(a.shape[0]):
        for k in range(6):
            i = low_idx + k
            c = int(uniform_codes(a[h, i], {'min': qrange(i)[0], 'max': qrange(i)[1]}, nbins))
            tok_seq.append(('<|LOW_%d|>' % c, code_tokens(ids, 'LOW', c)))
    add('<|LOW_END|>')

    return {
        'tokenizer': os.path.abspath(meta_dir),
        'robot_state_token': cfg['robot_state_token'],
        'robot_state_token_id': cfg['robot_state_token_id'],
        'robot_state_dim': cfg['robot_state_dim'],
        'robot_state_vector': [float(v) for v in state_vec],
        'state_slots_filled': int(min(cfg['robot_state_dim'], state.shape[-1])),
        'codebook_size': nbins,
        'discrete_action_tokens': [{'token': n, 'id': i} for n, i in tok_seq],
        'slot_ranges': {str(i): qrange(i) for i in range(len(off))},
        'enabled_action_slots': [int(i) for i in np.nonzero(on)[0]],
        'unspecified': [
            'how a continuous 54-dim chunk maps onto POS/ROT/HAND/LOW codes (no released RVQ weights)',
            'the token order inside/among the EEF, HAND and LOW blocks (markers exist, layout not published)',
            'how a 60-dim unified state becomes the 120-dim robot_state vector',
            'the per-codebook value ranges the codes were trained on',
        ],
        'placeholder_policy': 'uniform bins over each slot\'s minmax_q range, derived from this payload',
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', required=True)
    ap.add_argument('--payload', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--sample', type=int, default=0)
    ap.add_argument('--horizon', type=int, default=1)
    a = ap.parse_args(argv)
    r = build(a.meta, a.payload, a.horizon, a.sample)
    txt = json.dumps(r, indent=1)
    if a.out:
        open(a.out, 'w', encoding='utf-8').write(txt)
        print('wrote %s  (%d discrete tokens, %d slots filled in the state vector)'
              % (a.out, len(r['discrete_action_tokens']), r['state_slots_filled']))
    else:
        print(txt)
    print('')
    print('UNSPECIFIED, do not train on this until the authors answer:')
    for u in r['unspecified']:
        print('  - ' + u)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
