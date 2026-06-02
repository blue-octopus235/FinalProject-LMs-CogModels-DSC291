# Evanson, Lakretz & King (2023) — Language acquisition: do children and language models follow similar learning stages?

**Full citation:** Evanson, L., Lakretz, Y., & King, J.-R. (2023). Language acquisition: do children and language models follow similar learning stages? In *Findings of the Association for Computational Linguistics: ACL 2023*, 12205–12218.

**PDF:** https://aclanthology.org/2023.findings-acl.323.pdf  
**ACL Anthology:** https://aclanthology.org/2023.findings-acl.323/

---

## What the paper does

Studies the **learning trajectory** of language models — how their grammatical behavior changes *over the course of training* — and compares the ordering of acquired phenomena to the developmental order observed in children. By probing checkpoints throughout training (rather than only the final model), the authors find:
- LMs acquire syntactic/semantic phenomena in a **consistent, structured order** across training, not all at once.
- That order partially aligns with the sequence in which children acquire the same constructions.
- The trajectory, not just the endpoint, is the informative object for cognitive comparison.

## Relevance to our project

- **Methodological precedent for our trajectory analysis.** Our `src/trajectory.py` checkpoints every 500 steps and re-scores the minimal-pair set to chart accuracy and the attractor gap *over training* — exactly the checkpoint-trajectory methodology this paper argues for. It justifies treating the learning curve as a first-class result.
- **Frames our delay-vs-prevention finding.** Evanson et al. legitimize the question "*when* is the rule acquired?" as cognitively meaningful. Our contribution sharpens it for the noise setting: low noise shifts the acquisition curve later (delay), while 50% noise removes the acquisition stage entirely (prevention).
- **Acquisition-stage framing for the discussion.** Connects the project to the developmental-trajectory literature and supports reading the noise sweep as a manipulation of *whether and when* a developmental stage is reached.
- **Cite in results/discussion** where we introduce the trajectory curves.

## Key terms

- **Learning trajectory**: how a model's behavior evolves across training checkpoints, as opposed to its final-state performance
- **Acquisition order / stages**: the sequence in which grammatical phenomena become reliably handled
- **Age of acquisition (AoA)**: developmental analogue — when in learning a phenomenon is mastered

## Bibtex

```bibtex
@inproceedings{evanson2023language,
  title={Language acquisition: do children and language models follow similar learning stages?},
  author={Evanson, Linnea and Lakretz, Yair and King, Jean-R{\'e}mi},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  pages={12205--12218},
  year={2023}
}
```
