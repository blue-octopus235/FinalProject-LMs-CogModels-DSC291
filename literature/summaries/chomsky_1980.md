# Chomsky (1980) — Rules and Representations / Poverty of the Stimulus

**Core reference:** Chomsky, N. (1980). *Rules and Representations*. Columbia University Press.

**Poverty of Stimulus argument also in:**  
Chomsky, N. (1965). *Aspects of the Theory of Syntax*. MIT Press. (original POS framing)  
Chomsky, N. (1980). On cognitive structures and their development. In M. Piatelli-Palmarini (Ed.), *Language and Learning*. (the Piagetian debate version)

---

## What the argument is

The **Poverty of the Stimulus (POS)** argument holds that children acquire complex grammatical rules (like structure-dependent question formation) that could not be learned solely from the input they receive. Children:
1. Never hear enough relevant input to distinguish the correct hierarchical rule from a simpler linear one.
2. Never make the specific errors that a purely linear learner would make.
3. Therefore must have some prior grammatical knowledge built in (Universal Grammar / Language Acquisition Device).

The canonical example is **structure-dependent question formation**: to form a yes/no question from a complex sentence, children move the *main-clause* auxiliary, not the first auxiliary in the string. They never try the linear rule (move the first auxiliary), even though that rule would work for most sentences in their input and they have rarely if ever heard the distinguishing cases.

## Relevance to our project

- **The central theoretical stake**: Our project is designed as an empirical test of one dimension of the POS argument. If LMs trained on noisy input (which weakens the hierarchical signal in the data) still acquire the correct hierarchical SVA rule, it suggests hierarchical grammar *can* be learned from imperfect input — pushing back on the nativist conclusion.
- **Framing in the intro/discussion**: Chomsky's argument is the backdrop against which all of Linzen (2016), Gulordava (2018), and our own work should be contextualized. The question "can neural networks learn hierarchical rules from data?" is a direct machine-learning operationalization of the POS debate.
- **Caveat**: LMs are not children; the POS argument concerns *human* acquisition under specific constraints (no negative evidence, finite input, etc.). Our project does not refute POS but informs the debate by showing what distributional learning can and cannot do.

## Key terms

- **Poverty of the Stimulus (POS)**: the argument that input is insufficient for grammar learning without innate structure
- **Structure-dependence**: the property that grammatical rules are sensitive to hierarchical structure, not linear order
- **Universal Grammar (UG)**: Chomsky's proposed innate grammatical knowledge

## Citation

```bibtex
@book{chomsky1980rules,
  title={Rules and Representations},
  author={Chomsky, Noam},
  year={1980},
  publisher={Columbia University Press}
}

@book{chomsky1965aspects,
  title={Aspects of the Theory of Syntax},
  author={Chomsky, Noam},
  year={1965},
  publisher={MIT Press}
}
```
