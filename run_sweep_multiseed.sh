#!/bin/bash
set -e
cd "$(dirname "$0")"
CONDS="baseline low medium_low medium_high high"
MAXSENT=150000
EPOCHS=5
for SEED in 1 2 3; do
  for c in $CONDS; do
    echo "########## TRAIN $c seed$SEED ##########"
    python src/train.py --condition $c --max_sentences $MAXSENT --epochs $EPOCHS --seed $SEED
    echo "########## EVAL $c seed$SEED ##########"
    python src/evaluate.py --checkpoint checkpoints/lstm_${c}_seed${SEED}.pt --condition $c --seed $SEED
  done
done
echo "########## PLOTS ##########"
python src/plots.py
echo "########## DONE ##########"
