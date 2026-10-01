<!--
INVENTED TEXT, labelled as an invented imitation of what the arXiv HTML (LaTeXML) to
markdown converter will produce (that converter is added later). Not from any real paper.
Test fixture for app/ingest/clean.py and chunk.py.
-->
# Calibrated Self-Critique for Small Language Models

Ana Ferreira, Kenji Sato

###### Abstract

Small language models often accept their own wrong answers. We add a calibration step to
self-critique and show that it reduces confident errors.

## 1 Introduction

Self-critique asks a model to review its answer before replying. For small models the
review is often too lenient.

## 2 Related Work

Earlier work studies critique with large models (Example et al., 2025).

## 3 Method

### 3.1 Calibration Score

For an answer $a$ we compute the score $s(a)=\log p(a\mid q)-\log p(a)$, and keep the
answer only if $s(a)>\tau$.

Figure 2: Accepted answers as a function of the threshold $\tau$.

### 3.2 Choosing the Threshold

The threshold $\tau$ is chosen on a held-out split.

## 4 Results

Calibrated critique cuts confident errors by a third.

## References

- Example, A. and Sample, B. (2025). An invented study of critique. Journal of Nothing.
- Placeholder, C. (2024). Fictional baselines. In Invented Workshop.

## Appendix A Proof of Lemma 1

The score is monotone in $p(a\mid q)$ for fixed $p(a)$.

### A.1 Hyperparameters

We use a learning rate of $10^{-4}$.
