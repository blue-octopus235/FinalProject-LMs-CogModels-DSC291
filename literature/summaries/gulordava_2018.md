# Gulordava et al. (2018) — Colorless green recurrent networks dream hierarchically

**Full citation:** Gulordava, K., Bojanowski, P., Grave, E., Linzen, T., & Baroni, M. (2018). Colorless green recurrent networks dream hierarchically. In *Proceedings of NAACL-HLT 2018* (pp. 1195–1205).

**PDF:** https://aclanthology.org/N18-1108.pdf  
**ACL Anthology:** https://aclanthology.org/N18-1108/

---

## What the paper does

Extends Linzen et al. (2016) in two important directions:

1. **Cross-lingual scope**: evaluates LSTMs on SVA (and related long-distance dependencies) in English, Hebrew, Russian, and Italian — showing the hierarchical-sensitivity finding is not English-specific.
2. **Nonce-word ("colorless green") test**: generates minimal pairs where content words are replaced with legal but meaningless nonce words (e.g. *"The colorless green ideas sleep furiously"*-style sentences). This ensures the LSTM cannot use lexical statistics or word-meaning shortcuts — only structural information — to get the agreement right. The models still succeed, providing strong evidence that LSTMs encode genuine structural representations.

Key results:
- LSTM LMs retain above-chance SVA accuracy even on nonce sentences, strongly arguing against a purely distributional-statistics explanation.
- Performance is robust across languages despite no multilingual training objective.
- Hierarchical sensitivity emerges from pure next-word language modeling without syntactic supervision.

## Relevance to our project

- **Architectural baseline**: Our 2-layer LSTM LM setup (tied embeddings, dropout) follows Gulordava et al. as closely as Linzen et al. Both are cited in the methods section.
- **Hierarchical rule claim**: The paper's central thesis — that LSTMs learn hierarchical agreement from distributional input alone — is the *baseline assumption* our noise experiments are designed to stress-test. We ask: does this hierarchical sensitivity survive noisy training data?
- **Related work**: The colorless green paradigm shows pure structural learning is possible, which motivates framing our noise manipulation as a challenge to that same learning signal.

## Key terms

- **Colorless green test**: nonce-word minimal pairs designed to strip lexical cues
- **Long-distance agreement**: dependencies spanning multiple intervening words

## Bibtex

```bibtex
@inproceedings{gulordava2018colorless,
  title={Colorless green recurrent networks dream hierarchically},
  author={Gulordava, Kristina and Bojanowski, Piotr and Grave, {\'E}douard and Linzen, Tal and Baroni, Marco},
  booktitle={Proceedings of NAACL-HLT},
  pages={1195--1205},
  year={2018}
}
```
