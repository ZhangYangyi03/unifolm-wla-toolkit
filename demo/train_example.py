# -*- coding: utf-8 -*-
"""train_example.py - how the payload is consumed on the training box (AutoDL).

    python demo/train_example.py demo/data/payload.npz

Needs torch; this is the skeleton a post-train job hangs off. Nothing here is
WLA's own training code (it is not public yet) -- it is the shape of the batch
and the two normalizations the spec fixes, which is the part the data pipeline
has to agree with byte for byte.
"""
import sys
import numpy as np


def load(path):
    z = np.load(path)
    return {
        'action': z['action'], 'action_mask': z['action_mask'],
        'state': z['state'], 'state_mask': z['state_mask'],
        'split': z['split'],
        'action_offset': z['action_offset'], 'action_scale': z['action_scale'],
        'state_offset': z['state_offset'], 'state_scale': z['state_scale'],
        'chunk_size': int(z['chunk_size']), 'fps': int(z['fps']),
    }


def batches(d, which='train', bs=64, seed=0):
    idx = np.where(d['split'] == (1 if which == 'val' else 0))[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        sel = idx[i:i + bs]
        yield {'action': d['action'][sel].astype(np.float32),
               'action_mask': d['action_mask'][sel].astype(np.float32),
               'state': d['state'][sel].astype(np.float32),
               'state_mask': d['state_mask'][sel].astype(np.float32)}


def main(path):
    d = load(path)
    N, H, Da = d['action'].shape
    print('samples=%d  chunk=%d  action_dim=%d  state_dim=%d  fps=%d'
          % (N, H, Da, d['state'].shape[1], d['fps']))
    print('train=%d  val=%d' % ((d['split'] == 0).sum(), (d['split'] == 1).sum()))
    print('enabled action modules: %d / %d slots' % (int(d['action_mask'].sum(1).mean()), Da))
    b = next(batches(d))
    print('one batch: action %s  state %s' % (b['action'].shape, b['state'].shape))
    # the invariant a training job must be able to rely on
    assert np.all(np.isfinite(b['action'])), 'NaN in the action chunk'
    assert np.all((b['action_mask'] == 0) | (b['action_mask'] == 1)), 'mask not binary'
    print('normalized relative xyz: mean %.3g  std %.3g  (spec: zscore, so 0 and 1)'
          % (b['action'][:, :, 0:3].mean(), b['action'][:, :, 0:3].std()))
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        print('torch not installed here -- the array contract above is the whole point; '
              'on the training box wrap it in a Dataset returning point-cloud-free '
              'state/action tensors plus text tokens and images.')
        return
    class WLAPayload(Dataset):
        def __init__(self, d, which='train'):
            self.d = d
            self.idx = np.where(d['split'] == (1 if which == 'val' else 0))[0]
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            j = self.idx[i]
            return {'action': torch.from_numpy(d['action'][j]).float(),
                    'action_mask': torch.from_numpy(d['action_mask'][j]).float(),
                    'state': torch.from_numpy(d['state'][j]).float(),
                    'state_mask': torch.from_numpy(d['state_mask'][j]).float()}
    dl = DataLoader(WLAPayload(d), batch_size=4, shuffle=True)
    batch = next(iter(dl))
    print('torch DataLoader ok:', {k: tuple(v.shape) for k, v in batch.items()})


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'demo/data/payload.npz')
