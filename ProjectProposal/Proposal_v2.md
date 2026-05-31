DSC 291 Project Proposal: Robustness to Noise in Grammar Rule Learning

Authors: Jackson Wilke, Kelly Fu, Sanjana Garimella  
Course: DSC 291 — Language Models as Cognitive Models, UCSD  
Advisor: Alex Warstadt  
Date: May 7, 2026

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  
1\. RESEARCH QUESTION & MOTIVATION  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Children acquire robust grammatical rules despite noisy input, speech errors, non-native speech, and agreement violations in the environment. How resilient are language models to the same kind of noise?

We focus on subject-verb agreement (SVA), the most extensively studied grammatical construction in the LM probing literature. Our central questions are:

1. Can injecting SVA agreement errors into training data delay or prevent a model from acquiring the correct hierarchical agreement rule?  
2. Is the counterfactual linear agreement rule equally learnable under noise?  
3. Do RNNG (syntactically-biased) and LSTM (recurrent) architectures differ in their robustness to noise?

This connects directly to a foundational debate in language acquisition. Chomsky's poverty of the stimulus argument holds that linguistic input is too noisy and incomplete for grammar to be learned purely from data, implying some grammatical knowledge must be innate. This innate hypothesis is precisely motivated by the observation that children demonstrably acquire robust rules despite hearing speech errors, non-native input, and agreement violations. If LMs similarly succeed under noisy training conditions, it suggests grammar may be more learnable from input than traditionally assumed. If they fail, it raises the question of what children have that LMs lack, whether that is innate structure, embodied experience, or something else entirely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  
2\. RELATED WORK  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

* Linzen et al. (2016). Assessing the ability of LSTMs to learn syntax-sensitive dependencies. TACL. \[Foundational SVA evaluation; source of the rnn\_agreement dataset\]  
* Gulordava et al. (2018). Colorless green recurrent networks dream hierarchically. NAACL. \[Hierarchical vs. linear rule generalization in LSTMs\]  
* BLiMP (Warstadt et al., 2020). A challenge set for evaluating grammatical knowledge. TACL. \[Minimal pair evaluation paradigm\]  
* SyntaxGym (Gauthier et al., 2020). \[Targeted syntactic evaluation via surprisal\]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  
3\. EXPERIMENTAL DESIGN  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

\--- Data \---  
To construct our training data, we will stream a raw English Wikipedia corpus via the Hugging Face datasets package. After that, we will apply spacy's dependency parser to extract valid subject-verb structures. 

[https://www.dropbox.com/s/5axdu3q9jkxzjy8/agr\_50\_mostcommon\_10K.tsv.gz](https://www.dropbox.com/s/5axdu3q9jkxzjy8/agr_50_mostcommon_10K.tsv.gz)  
\-Linzen data

We will construct 5 training corpora by injecting SVA agreement errors at increasing rates:

 

| Condition | Error rate |
| :---- | :---- |
| Baseline | 0% |
| Low | 0.4% |
| Medium-low | 2% |
| Medium-high | 10% |
| High | 50% |

Error injection: for each sampled sentence, flip subject-verb agreement with probability p (the noise rate). This produces corrupted corpora that otherwise preserve all other syntactic and lexical properties. We will randomly select x% of our dataset or x sentences before and after the error insertion to check whether our constructed corpus contains any natural grammatical errors, and to verify if the syntactically corrupted sentences were modified correctly.

\--- Architectures \---  
Both architectures will be trained from scratch (not fine-tuned) on each noise condition:

1. RNNG (primary)  
   1. Syntactically biased model  
   2. The second paper the professor recommended compared RNNG and LSTM to show that syntactically biased models are better at few-shot learning, so it’s reasonable to choose them for comparison   
   3. Evaluated on next-token surprisal and minimal pair accuracy  
2. LSTM (secondary)  
   1. Recurrent LM matching Linzen et al. (2016) setup  
   2. Grounds our results in prior literature for direct comparison  
   3. Trains significantly faster; can run in parallel with GPT-2

Total model runs: 5 noise levels × 2 architectures × repeat 3 times using different random seeds \= 10 training runs

\--- Evaluation Metrics \---

1. Primary:   Minimal pair accuracy on SVA test sets   
   (does the model assign higher probability to the grammatical sentence?)  
2. Secondary: Surprisal scores for subject-verb agreement

           (how surprised is the model by the correct verb form?)

We will also probe whether models acquire the hierarchical rule (sensitive to the true subject, regardless of intervening nouns) vs. the linear rule (sensitive to the most recent noun before the verb). To distinguish the two, we evaluate on minimal pairs containing an attractor, an intervening noun that mismatches the true subject in number (e.g. "the keys to the cabinet are/is…"). A hierarchical model agrees with the true subject ("keys" → are) and ignores the attractor; a linear model agrees with the nearest noun ("cabinet" → is). The accuracy gap between attractor and no-attractor conditions measures reliance on linear cues, and we track how that gap widens with noise.

\--- Compute \---  
Primary:   UCSD teaching cluster  
Backup:    Alex Warstadt's lab cluster

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  
4\. TIMELINE & MILESTONES  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| Week | Milestone |
| ----- | :---- |
| 6 | Proposal due; set up compute environment and repo |
| 7 | Data pipeline: construct and verify 5 noise-level corpora; confirm training scripts run end-to-end on baseline |
| 8 | Train all 10 models (5 noise × 2 architectures); collect raw results |
| 9 | Evaluation runs; analysis (accuracy curves, rule probing); begin paper draft |
| 10 | Project presentations |
| 11 | Final paper due 6/11 |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  
5\. DIVISION OF LABOR  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Data & Training Infrastructure  
   1. Owns area. Sets up compute environment (teaching cluster access, job scripts); coordinates and monitors all training runs \[Jackson\]  
   2. Builds the noise injection pipeline: takes rnn\_agreement corpus, injects SVA errors at each of the 5 rates, validates output distributions \[Kelly\]  
   3. Implements training scripts for GPT-2-small and LSTM; manages model checkpointing and logging \[Sanjana\]   
2. Evaluation & Analysis  
   1. Owns area. Implements minimal pair accuracy evaluation across all 10 models; aggregates and organizes results \[Sanjana\]  
   2. Implements surprisal scoring and hierarchical vs. linear rule probing \[Kelly\]  
   3. Result visualization: accuracy curves by noise level, cross-architecture comparison plots, statistical analysis \[Jackson\]  
3. Writing & Presentation  
   1. Owns area. Coordinates paper structure and section assignments; writes introduction and conclusion \[Kelly\]  
   2. Writes methods and experimental setup sections (data, noise manipulation, architectures) \[Jackson\]  
   3. Writes results and discussion sections; creates presentation slides \[Sanjana\]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  
6\. EXPECTED CONTRIBUTIONS  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Empirical characterization of how noise rate affects SVA rule acquisition in GPT-2 and LSTMs — including whether there is a critical noise threshold beyond which acquisition breaks down.  
2. Comparison of hierarchical vs. linear generalization under varying noise conditions — does noise push models toward simpler (linear) strategies?  
3. Cross-architecture comparison (transformer vs. RNN) under identical noise conditions — do transformers and LSTMs differ in their robustness to noisy grammar input?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  
7\. DISCUSSION OF POSSIBLE OUTCOMES  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  
We anticipate three possible outcome patterns. First, models may remain robust at low noise rates (0.4–2%) but degrade sharply at higher rates (10–50%), suggesting a critical threshold for input quality beyond which grammar learning breaks down. Second, noise may exert a significant impact on the acquisition of hierarchical rules, presenting two competing hypotheses. On one hand, noise could disproportionately impair hierarchical rule learning while leaving linear rule performance intact, indicating that noise pushes models toward simpler generalization strategies. On the other hand, the models might maintain a robust preference for hierarchical generalization even under heavy noise, demonstrating the dominance of a global simplicity bias, where the hierarchical nature of the remaining grammar anchors the model's structural representations despite localized noise. Third, transformers and LSTMs may differ in robustness if transformers outperform LSTMs under noise, it implicates architectural differences; if both fail, it strengthens nativist arguments that input alone is insufficient for grammar acquisition.  
