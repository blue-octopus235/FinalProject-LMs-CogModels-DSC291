"""RNNG round-trip test: real Linzen orig_sentences -> benepar trees -> PTB .txt files."""
import sys, spacy, benepar

CORPUS = "/Users/jacksonwilke/Desktop/UCSD/DSC291-CogModels/FinalProject-LMs-CogModels-DSC291/data/sva_corpus_baseline_rate_0.0.tsv"

# pull ~40 real clean sentences (orig_sentence col), keep short-ish ones for the smoke test
sents = []
with open(CORPUS) as f:
    next(f)
    for line in f:
        s = line.split("\t")[0].strip()
        if 4 <= len(s.split()) <= 25:
            sents.append(s)
        if len(sents) >= 40:
            break

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})

trees = []
for s in sents:
    try:
        doc = nlp(s)
        sent = list(doc.sents)[0]
        ps = sent._.parse_string
        if ps.startswith("("):
            trees.append(ps)
    except Exception as e:
        print("PARSE FAIL:", s[:40], "->", e, file=sys.stderr)

print(f"parsed {len(trees)}/{len(sents)} sentences", file=sys.stderr)
print("=== sample tree ===", file=sys.stderr)
print(trees[0], file=sys.stderr)

# split 28/6/6
tr, va, te = trees[:28], trees[28:34], trees[34:40]
for name, ts in [("train", tr), ("valid", va), ("test", te)]:
    with open(f"/tmp/rnng-spike/ours-{name}.txt", "w") as f:
        f.write("\n".join(ts) + "\n")
print("wrote ours-{train,valid,test}.txt", file=sys.stderr)
