"""Generates the two Colab notebooks. Run once: python notebooks/_make_notebooks.py"""
import json
import os

HERE = os.path.dirname(__file__)


def nb(cells):
    return {
        "cells": cells,
        "metadata": {"accelerator": "GPU",
                     "colab": {"provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 0,
    }


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


# ---------------- colab_runner.ipynb ----------------
runner = nb([
    md("# SVA Noise Robustness — Colab Runner (LSTM)\n"
       "Trains + evaluates the LSTM on all 5 noise conditions on a T4.\n\n"
       "**Before running:** set `Runtime > Change runtime type > T4 GPU`.\n"
       "Put the 6 data TSVs in a Drive folder and point `DATA_DIR` at it."),
    code("# 1. Clone repo\n"
         "!git clone https://github.com/blue-octopus235/FinalProject-LMs-CogModels-DSC291.git\n"
         "%cd FinalProject-LMs-CogModels-DSC291\n"
         "!pip install -q lemminflect"),
    code("# 2. Mount Drive and point at the data folder (the 6 TSVs)\n"
         "from google.colab import drive\n"
         "drive.mount('/content/drive')\n"
         "DATA_DIR = '/content/drive/MyDrive/dsc291_data'  # <-- EDIT to your folder\n"
         "import os; print(sorted(os.listdir(DATA_DIR)))"),
    code("# 3. Config — KEEP max_sentences CONSTANT across conditions\n"
         "MAX_SENTENCES = 150000   # ~minutes/epoch on T4; raise for the paper\n"
         "EPOCHS = 5\n"
         "SEEDS = [1]              # add [1,2,3] for the paper-phase seed runs\n"
         "CONDITIONS = ['baseline','low','medium_low','medium_high','high']"),
    code("# 4. Train all conditions x seeds\n"
         "import subprocess\n"
         "for seed in SEEDS:\n"
         "    for cond in CONDITIONS:\n"
         "        print(f'==== train {cond} seed{seed} ====')\n"
         "        subprocess.run(['python','src/train.py','--condition',cond,\n"
         "            '--data_dir',DATA_DIR,'--max_sentences',str(MAX_SENTENCES),\n"
         "            '--epochs',str(EPOCHS),'--seed',str(seed)], check=True)"),
    code("# 5. Evaluate all checkpoints\n"
         "for seed in SEEDS:\n"
         "    for cond in CONDITIONS:\n"
         "        ckpt = f'checkpoints/lstm_{cond}_seed{seed}.pt'\n"
         "        subprocess.run(['python','src/evaluate.py','--checkpoint',ckpt,\n"
         "            '--condition',cond,'--data_dir',DATA_DIR,'--seed',str(seed)], check=True)"),
    code("# 6. Plots + table\n"
         "subprocess.run(['python','src/plots.py'], check=True)\n"
         "import pandas as pd\n"
         "from IPython.display import Image, display\n"
         "display(pd.read_csv('results/eval_results.csv'))\n"
         "for p in ['acc_vs_noise.png','attractor_gap.png','acc_by_attractor.png']:\n"
         "    display(Image(f'results/{p}'))"),
    code("# 7. (optional) copy results back to Drive so the team can see them\n"
         "!cp -r results $DATA_DIR/../dsc291_results 2>/dev/null; print('done')"),
])

# ---------------- rnng_spike.ipynb ----------------
spike = nb([
    md("# RNNG Spike — feasibility test (the #1 risk)\n"
       "Goal: get an RNNG **training end-to-end on a T4** and measure sec/epoch, so we know\n"
       "by Monday midday whether RNNG-vs-LSTM is on for the paper.\n\n"
       "We try Noji's `rnng-pytorch` (with effective batching). RNNG needs **bracketed trees**;\n"
       "for the spike we use the repo's sample data to confirm the model trains and to time it.\n"
       "Generating trees for our corpus (via benepar/spaCy) is the next step IF this works.\n\n"
       "**Decision gate:** if it trains and sec/epoch is reasonable on our data scale -> RNNG is on.\n"
       "If install/training is broken after ~2h -> LSTM-only for the talk, escalate to Warstadt for compute."),
    code("# 1. GPU check\n"
         "!nvidia-smi -L"),
    code("# 2. Clone candidate RNNG implementation(s)\n"
         "!git clone https://github.com/aistairc/rnng-pytorch.git\n"
         "%cd rnng-pytorch\n"
         "!cat README.md | head -60"),
    code("# 3. Install deps (adjust to the repo's requirements)\n"
         "!pip install -q torch numpy\n"
         "# many RNNG repos need a parsed/bracketed corpus + an action vocabulary;\n"
         "# follow the repo README's 'data preparation' to produce action sequences."),
    code("# 4. Toy train to confirm it runs + time it\n"
         "# Replace with the repo's actual train command from its README.\n"
         "# Time a few hundred steps and extrapolate sec/epoch for ~150k sentences.\n"
         "import time; t0=time.time()\n"
         "# !python train.py --train_file sample.train --val_file sample.val --batch_size 32 --num_epochs 1\n"
         "print('elapsed', time.time()-t0)"),
    md("## Notes to record here\n"
       "- Did it install cleanly? (Y/N + errors)\n"
       "- Did it train end-to-end on sample data? (Y/N)\n"
       "- sec/epoch at batch=32; extrapolated time for 150k sentences x 5 conditions?\n"
       "- How are trees produced for our corpus? (parser, format)\n"
       "- **GO / NO-GO for RNNG in the paper.**\n\n"
       "Fallback if NO-GO: ON-LSTM (Ordered Neurons) — a syntactically-biased model that\n"
       "trains like an LSTM with no gold trees; weaker bias but still a motivated comparison."),
])

with open(os.path.join(HERE, "colab_runner.ipynb"), "w") as f:
    json.dump(runner, f, indent=1)
with open(os.path.join(HERE, "rnng_spike.ipynb"), "w") as f:
    json.dump(spike, f, indent=1)
print("wrote colab_runner.ipynb and rnng_spike.ipynb")
