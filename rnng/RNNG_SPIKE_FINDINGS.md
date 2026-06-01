# RNNG Feasibility Spike — Findings

**Date:** 2026-05-31 (Jackson) · **Decision: GO** for RNNG-vs-LSTM in the paper.

The spike's job was the GO/NO-GO call + a reproducible path, **not** a finished RNNG
result. Both are in hand.

## TL;DR

- **Codebase works.** [aistairc/rnng-pytorch](https://github.com/aistairc/rnng-pytorch)
  (Noji & Oseki 2021, "Effective Batching for RNNGs") installs, preprocesses PTB-style
  trees, and **trains end-to-end** — verified on CPU locally (see numbers below).
- **Surprisal eval path exists.** `beam_search.py --lm_output_file surprisals.txt` emits
  per-word surprisals via word-synchronous beam search → this is how we score the
  minimal-pair test set for RNNG (analogous to the LSTM's verb-logprob comparison).
- **Two small fixes needed** to run on a fresh clone (captured as a patch — see below).
- **Real GPU training happens on Colab T4.** Local CPU is only for the smoke test.

## What was verified

1. **Install.** `pip install nltk tensorboard` (both missing from base). torch 2.11 already
   present. `sentencepiece` only needed if using subword vocab (we don't for the spike).
2. **Preprocess.** `preprocess.py` turns PTB bracketed trees → json with tokens, token_ids,
   oracle `actions` (NT(X)/SHIFT/REDUCE), action_ids, and an `in_order` variant. Ran clean
   on the repo's 20-sentence sample.
3. **Train (CPU smoke test).** `--device cpu`, w_dim=h_dim=64, 1 layer, batch 8, 2 epochs:
   - Epoch 1: PPL 43.32 (ActionPPL 18.37, WordPPL 380.57)
   - Epoch 2: PPL 41.74, **val PPL 42.92 → 40.95** — i.e. it learns. Model saves.
   - Throughput ~45 examples/sec on one M1 core with a toy model (irrelevant to T4 speed;
     the point was "does it step," not timing — time it on the T4 in the spike notebook).

## The two fixes (`rnng/rnng_pytorch_fixes.patch`)

Apply against a fresh `git clone` of aistairc/rnng-pytorch with `git apply`:

1. **NumPy 2.x ragged-array crash (BITES ON COLAB TOO if its numpy is 2.x).**
   `data.py`: `np.random.permutation(batches)` fails — newer NumPy refuses to build an
   inhomogeneous array from a list of variable-length batches. Patched to permute indices
   instead. This is the one to watch on Colab — re-clone hits it fresh.
2. **macOS-only:** `preprocess.py` calls `os.sched_getaffinity` (Linux-only) to default the
   job count. Workaround: pass `--jobs 1` (or any N) explicitly. **Linux/Colab unaffected**,
   so this is not in the patch — just remember it if anyone preprocesses on a Mac.

Reproducible command sequence (Linux/Colab):
```bash
git clone https://github.com/aistairc/rnng-pytorch.git && cd rnng-pytorch
git apply ../rnng/rnng_pytorch_fixes.patch
pip install -q nltk tensorboard
python preprocess.py --trainfile data/train.txt --valfile data/valid.txt \
  --testfile data/test.txt --outputfile data/ptb --vocabminfreq 1 --unkmethod berkeleyrule
python train.py --train_file data/ptb-train.json --val_file data/ptb-val.json \
  --save_path rnng.pt --batch_size 512 --fixed_stack --strategy top_down \
  --dropout 0.3 --optimizer adam --lr 0.001 --gpu 0   # T4: drop --device cpu, use --gpu 0
```

## KEY EFFICIENCY INSIGHT — parse the corpus ONCE, not 5×

The noise injection flips the verb's surface form (are↔is, VBP↔VBZ). This changes only the
**terminal token** at the verb position; it does **not** change the constituency bracketing,
so the SHIFT/REDUCE/NT oracle action sequence is **identical across all 5 noise conditions.**

→ Parse the *clean* corpus once, then swap the verb terminal per condition. Tree generation
is a one-time cost, not 5×. (For the verb tag, are/is are both finite verbs under VP; if we
want the tag to match the flipped form we can swap VBP↔VBZ on that one preterminal, but the
bracketing is untouched.)

## Tree generation for our corpus — the GO-path task (NOT done today, by design)

Our training text is the noised `orig_sentence` column: lowercased Linzen Wikipedia with
anonymized tokens (`a0`, `a1`, `a0p`). RNNG needs PTB bracketed trees, so we run a
constituency parser (benepar is the standard easy pick) over the **clean** corpus once,
emit PTB trees, then `preprocess.py`.

**Round-trip CONFIRMED (2026-05-31).** `rnng/parse_roundtrip_demo.py` pulls 40 real
`orig_sentence` lines → benepar (`spacy en_core_web_md` + `benepar_en3`) → **40/40 parsed,
0 failures** → PTB trees → `preprocess.py` (vocab 279, 0 dropped on length/unk filter) →
RNNG trains on our text: PPL 36.6 → 32.2, **val 35.2 → 30.8** over 3 CPU epochs. Sample
trees in `rnng/sample_parsed_trees.txt`. The anonymized tokens parse fine structurally
(e.g. `(VP (VBP are) (RB not) (NP (DT the) (NN issue)))` — verb correctly under VP), though
see the validity risk below re: tag quality. **The full our-text → trees → RNNG path works.**

Setup for the parser (one-time): `pip install benepar`,
`python -c "import benepar; benepar.download('benepar_en3')"`,
`python -m spacy download en_core_web_md`.

## OPEN VALIDITY RISK — log for the paper (6/11), do not solve now

Parser quality on **lowercased, anonymized** tokens (`a0`, `a0p`, no casing) may be weak.
RNNG's whole premise is that the *gold* syntactic supervision helps; if the trees are noisy,
RNNG gets weak/garbage supervision and the RNNG-vs-LSTM comparison is confounded
(we'd be testing "RNNG with bad trees" not "RNNG"). Mitigations to consider for the paper:
truecasing before parsing, sanity-checking a sample of trees by hand, and/or reporting parser
confidence. **Not a spike blocker** — the spike only had to prove the path exists.

## Fallback if RNNG ever stalls

ON-LSTM (Ordered Neurons) — syntactically-biased, trains like an LSTM with **no gold trees**,
so it sidesteps the parsing risk entirely. Weaker structural bias, but still a motivated
comparison if RNNG tree quality turns out to be a problem.
