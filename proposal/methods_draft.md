# Methods (draft — Jackson's section)

## Data and noise manipulation
We build on the subject–verb agreement (SVA) corpus of Linzen et al. (2016),
`agr_50_mostcommon_10K` (≈1.58M English sentences extracted from Wikipedia, with rare tokens
anonymized so that the vocabulary is capped at the 10K most frequent types). Each sentence is
annotated with the subject, the inflected verb, the verb's position, and the number of
*intervening* nouns that disagree in number with the subject (the "attractors" central to the
hierarchical-vs-linear distinction).

To study robustness to noise we construct five training corpora that are identical except for the
rate at which SVA agreement is corrupted. For each sentence we flip the verb's number with
probability *p*, producing an ungrammatical agreement relation while leaving every other lexical
and syntactic property unchanged. Verb flipping uses a deterministic inflection procedure
(`lemminflect`, with hand-specified irregulars for high-frequency copulas and auxiliaries) applied
at the annotated verb index. The five noise rates are *p* ∈ {0, 0.004, 0.02, 0.10, 0.50}
(baseline, low, medium-low, medium-high, high). We verified each corpus by confirming that the
empirical fraction of corrupted sentences matches *p* and that the baseline corpus is fully
grammatical.

## Held-out evaluation set
Because the five corpora are row-aligned with the source corpus, we reserve a single fixed,
randomly chosen set of 20,000 sentence indices as a held-out test set. These rows are excluded from
all training corpora, guaranteeing that no test sentence appears in training under any noise
condition. The test set is constructed from the *uncorrupted* source corpus, so all test items are
grammatical, and the same test set is used to evaluate every condition; only the training noise
varies. Training-set size is held constant across conditions so that any change in performance is
attributable to noise rather than to data quantity.

## Models
We compare two architectures trained from scratch on each noise condition. The primary contrast is
between a **syntactically-biased** model and a **sequential** model, which directly targets our
question of whether an explicit structural bias makes hierarchical agreement more robust to noisy
input. The sequential baseline is a word-level LSTM language model in the style of Linzen et al.
(2016) and Gulordava et al. (2018) (2 layers, tied input/output embeddings, dropout). The
syntactically-biased model is a Recurrent Neural Network Grammar (RNNG; Dyer et al. 2016), which
incorporates explicit hierarchical structure and has been shown to improve few-shot syntactic
generalization relative to sequential models (Wilcox et al. 2020). [If RNNG training proves
infeasible within the compute budget we substitute ON-LSTM (Shen et al. 2019), a model with a
softer structural inductive bias that requires no parse supervision; final choice noted in
Results.]

## Evaluation metrics
Our primary metric is **minimal-pair accuracy**: for each held-out sentence we form a minimal pair
by contrasting the grammatical verb form with its number-flipped, ungrammatical counterpart, and
score the model correct if it assigns higher conditional probability to the grammatical form given
the preceding context. As a graded secondary measure we report the **surprisal** (−log₂
probability) of the correct verb.

To probe whether models acquire the hierarchical or the linear rule, we partition the test set by
the presence of an attractor (an intervening noun mismatching the subject in number). We report
accuracy separately for attractor and no-attractor items and summarize reliance on the linear cue
as the **attractor gap**, the difference in accuracy between the two. A model that has learned the
hierarchical rule shows a small gap; a model relying on the nearest noun shows a large gap. Our
central analysis tracks how this gap changes as the training-noise rate increases.

## Compute
All models are trained on a single NVIDIA T4 GPU (Google Colab Pro). The LSTM runs across all five
noise conditions fit comfortably within this budget; the RNNG comparison, which is more expensive,
is run as compute permits and may use additional resources.
```
```
