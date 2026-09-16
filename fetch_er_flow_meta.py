# -*- coding: utf-8 -*-
"""fetch_er_flow_meta.py - re-download the released UnifoLM-ER-Flow metadata.

The full tokenizer.json is ~11 MB, so it is not committed; `added_tokens.json`
in demo/er_flow_meta is the 1317-token subset extracted from it, and that is what
the scripts read. Run this if you want the full file or a fresh copy.

    python fetch_er_flow_meta.py
"""
import json, os, urllib.request

BASE = 'https://huggingface.co/unitreerobotics/UnifoLM-ER-Flow/resolve/main/'
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo', 'er_flow_meta')
FILES = ['config.json', 'generation_config.json', 'tokenizer_config.json',
         'tokenizer.json', 'model.safetensors.index.json']


def main():
    os.makedirs(HERE, exist_ok=True)
    for f in FILES:
        req = urllib.request.Request(BASE + f, headers={'User-Agent': 'unifolm-wla-toolkit'})
        b = urllib.request.urlopen(req, timeout=120).read()
        open(os.path.join(HERE, f), 'wb').write(b)
        print('  %-32s %9d bytes' % (f, len(b)))
    tok = json.load(open(os.path.join(HERE, 'tokenizer.json'), encoding='utf-8'))
    at = {t['content']: t['id'] for t in tok['added_tokens']}
    json.dump(at, open(os.path.join(HERE, 'added_tokens.json'), 'w', encoding='utf-8'),
              indent=0, sort_keys=True)
    print('  extracted %d added tokens -> added_tokens.json' % len(at))


if __name__ == '__main__':
    main()
