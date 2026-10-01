<!--
INVENTED TEXT. Not from any real paper. It imitates the layout of a Hugging Face paper
.md (title, authors, abstract, numbered sections, figure, table, equation, references,
appendix). A real excerpt was planned, but its licence could not be verified, so this
file is invented instead. Test fixture for app/ingest/clean.py and chunk.py.
-->
# Sparse Routing Adapters for Long-Context Retrieval

Mira Okonkwo<sup>1</sup>, Tomasz Lindqvist<sup>2</sup>

<sup>1</sup>Invented University, <sup>2</sup>Example Lab

## Abstract

We study sparse routing adapters, a light way to extend retrieval models to long inputs.
Routing sends each passage to a small expert, which keeps memory use flat as inputs grow.

## 1 Introduction

Retrieval models are usually trained on short passages. When a document is long, the
model sees only a window of it, and evidence outside that window is lost.

We propose sparse routing adapters (SRA). Each adapter is a small feed-forward block, and
a router picks two adapters per passage.

![](figures/overview.png)

Figure 1: Overview of SRA. A router assigns each passage to two of eight adapters.

## 2 Method

### 2.1 Router

The router scores each adapter with a dot product and keeps the top two:

$$
r_i = \operatorname{softmax}(W h)_i
$$

### 2.2 Training

We train the adapters with a contrastive loss and freeze the base model.

## 3 Experiments

Table 1: Recall@10 on the invented LongDocs benchmark.

| Model | Recall@10 | Memory (GB) |
| --- | --- | --- |
| Base | 41.2 | 3.1 |
| SRA | 48.7 | 3.3 |

## 4 Conclusion

Sparse routing adapters add long-context ability at a small memory cost.

## References

[1] A. Example and B. Sample. Invented retrieval methods. In Proc. of Nowhere, 2025.
[2] C. Placeholder. A fictional survey of adapters. arXiv preprint, 2026.

## A Additional Results

### A.1 Ablation on the Number of Adapters

Using four adapters instead of eight lowers Recall@10 by 1.3 points.
