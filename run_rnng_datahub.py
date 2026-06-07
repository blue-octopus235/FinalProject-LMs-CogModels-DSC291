#!/usr/bin/env python3
"""RNNG training + eval pipeline for datahub.ucsd.edu (1 GPU, ~16 GB RAM).

All expensive outputs are skipped if they already exist, so re-running after a
disconnect resumes where it left off. Run inside a tmux session.

Usage:
    python run_rnng_datahub.py --data_dir /path/to/tsv/files

    # optional flags:
    #   --work_dir ~/dsc291_rnng   (default)
    #   --rnng_dir ~/rnng-pytorch  (default; cloned here if missing)
    #   --seeds 1 2 3              (default)
    #   --conditions baseline low medium_low medium_high high  (default: all 5)
    #   --epochs 8                 (default)
"""

import argparse, csv, json, math, os, re, subprocess, sys
from pathlib import Path

# ── Repo root (script lives at repo root) ─────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent

# ── Constants that match the notebook ─────────────────────────────────────────
CONDITIONS_FILES = {
    'baseline':    'sva_corpus_baseline_rate_0.0.tsv',
    'low':         'sva_corpus_low_rate_0.004.tsv',
    'medium_low':  'sva_corpus_medium_low_rate_0.02.tsv',
    'medium_high': 'sva_corpus_medium_high_rate_0.1.tsv',
    'high':        'sva_corpus_high_rate_0.5.tsv',
}
RATES = {'baseline': 0.0, 'low': 0.004, 'medium_low': 0.02, 'medium_high': 0.10, 'high': 0.50}

csv.field_size_limit(2**31 - 1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(msg):
    print(f'\n{"="*60}\n  {msg}\n{"="*60}', flush=True)


def run_streaming(cmd, cwd, label):
    """Run a subprocess, stream output live, raise on non-zero exit."""
    print(f'[{label}] Running: {" ".join(str(c) for c in cmd)}', flush=True)
    tail = []
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in proc.stdout:
        print(line, end='', flush=True)
        tail.append(line)
        if len(tail) > 120:
            tail.pop(0)
    proc.wait()
    if proc.returncode != 0:
        print('\n--- last output ---\n' + ''.join(tail))
        raise RuntimeError(f'{label} failed (exit {proc.returncode})')


def patch_file(path, old, new, label):
    src = Path(path).read_text()
    if old in src:
        Path(path).write_text(src.replace(old, new, 1))
        print(f'  Patched: {label}')
    elif new in src:
        print(f'  Already patched: {label}')
    else:
        raise RuntimeError(f'Patch target not found: {label}\nLooking for: {repr(old[:60])}')


# ── Step 1: Clone + patch rnng-pytorch ────────────────────────────────────────

def setup_rnng(rnng_dir: Path):
    banner('Step 1 — Clone + patch aistairc/rnng-pytorch')
    if not rnng_dir.exists():
        subprocess.run(
            ['git', 'clone', 'https://github.com/aistairc/rnng-pytorch.git', str(rnng_dir)],
            check=True,
        )
        print('Cloned rnng-pytorch.')
    else:
        print(f'{rnng_dir} already exists.')

    # Fix 1: NumPy 2.x — permute indices instead of ragged list
    patch_file(
        rnng_dir / 'data.py',
        '        batches = np.random.permutation(batches)',
        '        perm = np.random.permutation(len(batches))\n'
        '        batches = [batches[k] for k in perm]',
        'NumPy 2.x ragged permutation (data.py)',
    )

    # Fix 2: PyTorch 2.6 weights_only — torch.load now defaults to True
    patch_file(
        rnng_dir / 'beam_search.py',
        'checkpoint = torch.load(args.model_file)',
        'checkpoint = torch.load(args.model_file, weights_only=False)',
        'PyTorch 2.6 weights_only (beam_search.py)',
    )
    patch_file(
        rnng_dir / 'train.py',
        'checkpoint = torch.load(args.train_from)',
        'checkpoint = torch.load(args.train_from, weights_only=False)',
        'PyTorch 2.6 weights_only (train.py)',
    )


# ── Step 2: Parse clean corpus with benepar (one-time) ───────────────────────

def parse_corpus(data_dir: Path, work_dir: Path, max_sentences: int, parse_batch: int):
    banner('Step 2 — Parse clean corpus with benepar (one-time)')

    parse_dir       = work_dir / 'parsed_trees'
    clean_trees_f   = parse_dir / 'clean_trees.txt'
    parse_meta_f    = parse_dir / 'parse_meta.json'
    parse_dir.mkdir(parents=True, exist_ok=True)

    if parse_meta_f.exists():
        print('Loading cached parse results...')
        parse_meta  = json.loads(parse_meta_f.read_text())
        clean_trees = clean_trees_f.read_text().splitlines()
        print(f'Loaded {len(clean_trees)} trees.')
        return parse_dir, parse_meta, clean_trees

    # Import data_utils from repo src/
    sys.path.insert(0, str(REPO_DIR / 'src'))
    from data_utils import get_test_indices, ORIGINAL_FILE

    import spacy, benepar, transformers
    from tqdm.auto import tqdm

    # Patch for benepar 0.2.x + transformers >= 4.40 (T5Tokenizer compatibility)
    if not hasattr(transformers.T5Tokenizer, 'build_inputs_with_special_tokens'):
        transformers.T5Tokenizer.build_inputs_with_special_tokens = (
            lambda self, token_ids_0, token_ids_1=None: token_ids_0
        )

    test_idx  = get_test_indices()
    orig_path = data_dir / ORIGINAL_FILE

    train_rows = []
    with open(orig_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if i in test_idx:
                continue
            s = row.get('orig_sentence', '').strip()
            if not s:
                continue
            try:
                vidx = int(row['verb_index']) - 1
            except (KeyError, ValueError):
                continue
            toks = s.split()
            if not (0 <= vidx < len(toks)):
                continue
            train_rows.append({
                'row_idx':   i,
                'sent':      s,
                'verb_idx':  vidx,
                'verb_form': toks[vidx],
                'verb_pos':  row.get('verb_pos', ''),
            })
            if len(train_rows) >= max_sentences:
                break

    print(f'Collected {len(train_rows)} training sentences.')

    nlp = spacy.load('en_core_web_md')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

    parse_meta  = []
    clean_trees = []
    fail_count  = 0

    for start in tqdm(range(0, len(train_rows), parse_batch), desc='Parsing'):
        batch = train_rows[start:start + parse_batch]
        try:
            docs = list(nlp.pipe([r['sent'] for r in batch]))
        except Exception:
            docs = [None] * len(batch)
        for row, doc in zip(batch, docs):
            if doc is None:
                fail_count += 1
                continue
            try:
                ps = list(doc.sents)[0]._.parse_string
                if ps.startswith('('):
                    parse_meta.append({
                        'row_idx':   row['row_idx'],
                        'verb_idx':  row['verb_idx'],
                        'verb_form': row['verb_form'],
                        'verb_pos':  row['verb_pos'],
                        'tree_line': len(clean_trees),
                    })
                    clean_trees.append(ps)
                else:
                    fail_count += 1
            except Exception:
                fail_count += 1

    print(f'Parsed {len(clean_trees)}/{len(train_rows)}, {fail_count} failures.')
    clean_trees_f.write_text('\n'.join(clean_trees) + '\n')
    parse_meta_f.write_text(json.dumps(parse_meta))
    print(f'Saved to {parse_dir}.')
    return parse_dir, parse_meta, clean_trees


# ── Step 3: Verb swap + preprocess.py per condition ──────────────────────────

def swap_verb_in_tree(tree_str, verb_idx_0based, noised_verb):
    terminals = list(re.finditer(r'\((\S+)\s+([^\s()]+)\)', tree_str))
    if verb_idx_0based >= len(terminals):
        return tree_str
    m = terminals[verb_idx_0based]
    old_tag = m.group(1)
    new_tag = 'VBZ' if old_tag == 'VBP' else ('VBP' if old_tag == 'VBZ' else old_tag)
    return tree_str[:m.start()] + f'({new_tag} {noised_verb})' + tree_str[m.end():]


def preprocess_condition(cond, data_dir, parse_dir, parse_meta, clean_trees, rnng_dir):
    banner(f'Step 3 — Preprocess condition: {cond}')
    cond_dir   = parse_dir / cond
    json_train = cond_dir / f'{cond}-train.json'

    if json_train.exists():
        print(f'[{cond}] Preprocessed JSON exists — skipping.')
        return

    cond_dir.mkdir(parents=True, exist_ok=True)

    row_to_meta       = {m['row_idx']: m for m in parse_meta}
    selected_row_idxs = set(row_to_meta.keys())

    cond_path = data_dir / CONDITIONS_FILES[cond]
    noised_verb_map = {}
    with open(cond_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if i in selected_row_idxs:
                toks = row.get('orig_sentence', '').split()
                vidx = row_to_meta[i]['verb_idx']
                if 0 <= vidx < len(toks):
                    noised_verb_map[i] = toks[vidx]

    cond_trees, swap_count = [], 0
    for meta in parse_meta:
        clean_tree  = clean_trees[meta['tree_line']]
        noised_verb = noised_verb_map.get(meta['row_idx'], meta['verb_form'])
        if noised_verb != meta['verb_form']:
            cond_trees.append(swap_verb_in_tree(clean_tree, meta['verb_idx'], noised_verb))
            swap_count += 1
        else:
            cond_trees.append(clean_tree)
    print(f'  {swap_count}/{len(cond_trees)} verb terminals swapped.')

    n_val   = max(200, int(len(cond_trees) * 0.1))
    n_test  = min(200, n_val)
    splits  = [
        (cond_dir / 'train.txt', cond_trees[n_val:]),
        (cond_dir / 'val.txt',   cond_trees[:n_val]),
        (cond_dir / 'test.txt',  cond_trees[:n_test]),
    ]
    for path, trees in splits:
        path.write_text('\n'.join(t.strip() for t in trees) + '\n')
    print(f'  train={len(cond_trees) - n_val}, val={n_val}, test(dummy)={n_test}')

    out_prefix = cond_dir / cond
    run_streaming(
        ['python', rnng_dir / 'preprocess.py',
         '--trainfile',    cond_dir / 'train.txt',
         '--valfile',      cond_dir / 'val.txt',
         '--testfile',     cond_dir / 'test.txt',
         '--outputfile',   out_prefix,
         '--vocabminfreq', '1',
         '--unkmethod',    'berkeleyrule'],
        cwd=rnng_dir,
        label=f'preprocess [{cond}]',
    )
    print(f'[{cond}] Preprocessing done.')


# ── Step 4: Train ─────────────────────────────────────────────────────────────

def train_condition_seed(cond, seed, parse_dir, work_dir, rnng_dir, epochs):
    banner(f'Step 4 — Train: {cond} seed{seed}')
    ckpt = work_dir / 'checkpoints' / f'rnng_{cond}_seed{seed}.pt'
    (work_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    if ckpt.exists():
        print(f'[{cond} seed{seed}] Checkpoint exists — skipping.')
        return ckpt

    cond_dir = parse_dir / cond
    run_streaming(
        ['python', rnng_dir / 'train.py',
         '--train_file',       cond_dir / f'{cond}-train.json',
         '--val_file',         cond_dir / f'{cond}-val.json',
         '--save_path',        ckpt,
         '--fixed_stack',
         '--strategy',         'top_down',
         '--w_dim',            '256',
         '--h_dim',            '256',
         '--num_layers',       '2',
         '--dropout',          '0.3',
         '--optimizer',        'adam',
         '--lr',               '0.001',
         '--batch_size',       '128',
         '--batch_token_size', '15000',
         '--num_epochs',       str(epochs),
         '--gpu',              '0',
         '--seed',             str(seed)],
        cwd=rnng_dir,
        label=f'train [{cond} seed{seed}]',
    )
    print(f'[{cond} seed{seed}] Done — saved to {ckpt}')
    return ckpt


# ── Step 5: Eval ──────────────────────────────────────────────────────────────

def parse_surprisal_file(surp_file, n_sents):
    sent_surps = [[] for _ in range(n_sents)]
    with open(surp_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('---') or line.startswith('perplexity'):
                continue
            parts = line.split('\t')
            if len(parts) < 5:
                continue
            try:
                si, ti, surp = int(parts[0]), int(parts[1]), float(parts[4])
            except ValueError:
                continue
            if si < n_sents:
                while len(sent_surps[si]) <= ti:
                    sent_surps[si].append(None)
                sent_surps[si][ti] = surp
    return sent_surps


def summarize(pairs, surp_correct, surp_incorrect):
    def acc(mask):
        idx = [k for k, m in enumerate(mask) if m]
        if not idx:
            return float('nan'), 0
        return sum(1 for k in idx if surp_correct[k] < surp_incorrect[k]) / len(idx), len(idx)

    attr   = [p['has_attractor'] for p in pairs]
    noattr = [not a for a in attr]
    acc_all,    n_all    = acc([True] * len(pairs))
    acc_attr,   n_attr   = acc(attr)
    acc_noattr, n_noattr = acc(noattr)
    mean_surp = sum(surp_correct) / max(len(surp_correct), 1) / math.log(2)
    gap = (acc_noattr - acc_attr) if (n_attr and n_noattr) else float('nan')
    return {
        'acc_all': acc_all, 'n_all': n_all,
        'acc_no_attractor': acc_noattr, 'n_no_attractor': n_noattr,
        'acc_attractor': acc_attr,      'n_attractor': n_attr,
        'attractor_gap': gap,
        'mean_surprisal_correct_bits': mean_surp,
    }


def eval_condition_seed(cond, seed, ckpt, all_pairs, work_dir, rnng_dir,
                        results_csv, write_header):
    banner(f'Step 5 — Eval: {cond} seed{seed}')

    eval_dir = work_dir / 'eval_tmp' / f'{cond}_seed{seed}'
    eval_dir.mkdir(parents=True, exist_ok=True)

    correct_txt   = eval_dir / 'correct.txt'
    incorrect_txt = eval_dir / 'incorrect.txt'

    if not correct_txt.exists():
        with open(correct_txt, 'w') as fc, open(incorrect_txt, 'w') as fi:
            for p in all_pairs:
                fc.write(' '.join(p['prefix'] + [p['correct']])   + '\n')
                fi.write(' '.join(p['prefix'] + [p['incorrect']]) + '\n')

    for txt_file, surp_file in [
        (correct_txt,   eval_dir / 'surp_correct.txt'),
        (incorrect_txt, eval_dir / 'surp_incorrect.txt'),
    ]:
        if surp_file.exists():
            print(f'  [{cond} seed{seed}] {surp_file.name} cached — skipping.')
            continue
        run_streaming(
            ['python', rnng_dir / 'beam_search.py',
             '--test_file',        txt_file,
             '--model_file',       ckpt,
             '--lm_output_file',   surp_file,
             '--beam_size',        '100',
             '--word_beam_size',   '20',
             '--shift_size',       '5',
             '--batch_size',       '50',
             '--stack_size_bound', '50',
             '--gpu',              '0',
             '--device',           'cuda'],
            cwd=rnng_dir,
            label=f'beam_search [{cond} seed{seed} {txt_file.name}]',
        )

    cs = parse_surprisal_file(eval_dir / 'surp_correct.txt',   len(all_pairs))
    is_ = parse_surprisal_file(eval_dir / 'surp_incorrect.txt', len(all_pairs))

    valid_pairs, surp_c, surp_i = [], [], []
    for k, p in enumerate(all_pairs):
        vpos = len(p['prefix'])
        c_row, i_row = cs[k], is_[k]
        if (c_row and vpos < len(c_row) and c_row[vpos] is not None and
                i_row and vpos < len(i_row) and i_row[vpos] is not None):
            valid_pairs.append(p)
            surp_c.append(c_row[vpos])
            surp_i.append(i_row[vpos])

    print(f'[{cond} seed{seed}] Valid pairs: {len(valid_pairs)}/{len(all_pairs)}')
    res = summarize(valid_pairs, surp_c, surp_i)
    res.update({'model': 'rnng', 'condition': cond, 'rate': RATES[cond],
                'seed': seed, 'checkpoint': ckpt.name})

    COLS = ['model', 'condition', 'rate', 'seed',
            'acc_all', 'acc_no_attractor', 'acc_attractor', 'attractor_gap',
            'mean_surprisal_correct_bits',
            'n_all', 'n_attractor', 'n_no_attractor', 'checkpoint']

    with open(results_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if write_header:
            w.writeheader()
        w.writerow({k: res.get(k) for k in COLS})

    print(f'  acc_all={res["acc_all"]:.3f}  attractor_gap={res["attractor_gap"]:.3f}')
    return False  # header written


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir',   required=True,
                   help='Directory containing the 5 .tsv corpora + agr_50_mostcommon_10K.tsv')
    p.add_argument('--work_dir',   default=os.path.expanduser('~/dsc291_rnng'),
                   help='Directory for parsed trees, checkpoints, eval output (default: ~/dsc291_rnng)')
    p.add_argument('--rnng_dir',   default=os.path.expanduser('~/rnng-pytorch'),
                   help='Where to clone aistairc/rnng-pytorch (default: ~/rnng-pytorch)')
    p.add_argument('--seeds',      nargs='+', type=int, default=[1, 2, 3])
    p.add_argument('--conditions', nargs='+',
                   default=['baseline', 'low', 'medium_low', 'medium_high', 'high'])
    p.add_argument('--epochs',     type=int, default=8)
    p.add_argument('--max_sentences', type=int, default=150_000)
    p.add_argument('--parse_batch',   type=int, default=32,
                   help='Benepar batch size (lower = less RAM; default 32 for 16 GB)')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    work_dir = Path(args.work_dir)
    rnng_dir = Path(args.rnng_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Add repo src/ to path for data_utils / minimal_pairs imports
    sys.path.insert(0, str(REPO_DIR / 'src'))

    # Step 1: clone + patch
    setup_rnng(rnng_dir)

    # Step 2: parse clean corpus (one-time)
    parse_dir, parse_meta, clean_trees = parse_corpus(
        data_dir, work_dir, args.max_sentences, args.parse_batch
    )

    # Step 3: preprocess each condition
    for cond in args.conditions:
        preprocess_condition(cond, data_dir, parse_dir, parse_meta, clean_trees, rnng_dir)

    # Build minimal-pair test set once (shared across all conditions)
    banner('Building minimal-pair test set')
    from data_utils import get_test_indices, ORIGINAL_FILE
    from minimal_pairs import build_minimal_pairs
    test_idx  = get_test_indices()
    all_pairs, skipped = build_minimal_pairs(data_dir / ORIGINAL_FILE, test_idx, vocab=None)
    print(f'Minimal pairs: {len(all_pairs)}, skipped: {skipped}')

    # Steps 4 + 5: train then eval each condition × seed
    results_csv  = REPO_DIR / 'results' / 'rnng_eval_results.csv'
    results_csv.parent.mkdir(exist_ok=True)
    write_header = not results_csv.exists()

    for seed in args.seeds:
        for cond in args.conditions:
            ckpt = train_condition_seed(cond, seed, parse_dir, work_dir, rnng_dir, args.epochs)
            write_header = eval_condition_seed(
                cond, seed, ckpt, all_pairs, work_dir, rnng_dir, results_csv, write_header
            )

    banner('All done')
    print(f'Results: {results_csv}')


if __name__ == '__main__':
    main()
