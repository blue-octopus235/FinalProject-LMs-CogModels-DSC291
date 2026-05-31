#!/bin/bash
set -e
cd "$(dirname "$0")"
CONDS="baseline low medium_low medium_high high"
MAXSENT=50000
EPOCHS=5
SEED=1
for c in $CONDS; do
  echo "########## TRAIN $c ##########"
  python src/train.py --condition $c --max_sentences $MAXSENT --epochs $EPOCHS --seed $SEED
  echo "########## EVAL $c ##########"
  python src/evaluate.py --checkpoint checkpoints/lstm_${c}_seed${SEED}.pt --condition $c --seed $SEED
done
echo "########## PLOTS ##########"
python src/plots.py
echo "########## DONE ##########"
cat results/eval_results.csv
