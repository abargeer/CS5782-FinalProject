# "Where Should I Look?" Reimplementing Visual Attention for Neural Image Captioning

**CS 4782 Final Project, Cornell University, Spring 2026**

**Authors:** James Tu (jt737), Evan Zhu (ejz26), Ayaan Bargeer (aab274)

---

## 1. Introduction

This repository contains our re-implementation of **"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"** by Xu et al. (ICML 2015) as part of the CS 4782 (Deep Learning) final project at Cornell University.

The paper's main contribution is an attention-based encoder-decoder architecture for image captioning. Instead of compressing an entire image into a single feature vector, the model extracts a grid of spatial features from a CNN and uses an attention mechanism to dynamically focus on relevant image regions while generating each word of the caption. The paper proposes two attention variants: **soft (deterministic)** attention trained via backpropagation, and **hard (stochastic)** attention trained via REINFORCE.

Beyond reproducing the paper's soft attention results, we designed two controlled attention experiments to investigate whether attention is truly learning spatial alignment or merely helping the decoder compute better features.

## 2. Chosen Result

We aimed to reproduce the **soft attention results from Table 1** of the original paper on the **Flickr30k** dataset:

| Model                  | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
| ---------------------- | ------ | ------ | ------ | ------ | ------ |
| Paper (Soft Attention) | 66.7   | 43.4   | 28.8   | 19.1   | 18.49  |

This result is significant because it demonstrates that attention over spatial CNN features can achieve state-of-the-art captioning performance while providing interpretable visualizations of where the model "looks" when generating each word.

## 3. GitHub Contents

```
├── README.md                    # This file
├── code/
│   ├── show_attend_tell.ipynb   # Full pipeline: training, attention ablation experiments, evaluation, visualization
├── data/
│   └── README.md                # Instructions for downloading
├── results/
│   ├── figures/      # Training/validation loss and BLEU-4 curves
│   ├── attention_visualizations/ # Sample attention maps per generated word
│   └── metrics.json             # Final BLEU and METEOR scores
├── poster/
│   └── poster.pdf               # Poster presented in class
├── report/
│   └── group_show_attend_tell_2page_report.pdf
├── LICENSE
└── .gitignore
```

## 4. Re-implementation Details

**Architecture** (matching the paper as closely as possible):

| Component    | Paper                             | Ours                              |
| ------------ | --------------------------------- | --------------------------------- |
| Encoder      | VGG-19 conv4, 14×14×512           | VGG-19 conv4, 14×14×512           |
| Decoder      | LSTM (hidden=512)                 | LSTM (hidden=512)                 |
| Attention    | Additive + doubly stochastic reg. | Additive + doubly stochastic reg. |
| β gate       | σ(f*β(h*{t-1}))                   | σ(f*β(h*{t-1}))                   |
| Output layer | Deep output (Eq. 7)               | Deep output (Eq. 7)               |
| Init h₀, c₀  | MLP(mean(annotations))            | MLP(mean(annotations))            |
| Vocab size   | 10,000                            | 10,000                            |
| Inference    | Beam search                       | Beam search (k=3)                 |

**Datasets:** Flickr8k (8K images, development/debug) and Flickr30k (31K images, primary evaluation), each with 5 captions per image. Downloaded via Kaggle (see `data/README.md`).

**Metrics:** BLEU-1 through BLEU-4 (without brevity penalty) and METEOR, computed using `pycocoevalcap`.

**Attention Ablation Experiments:** To test whether attention is truly understanding spatial layout or just helping the model compute better features, we trained two variants with attention as the only changing variable:

- **Shuffled Attention:** Same attention weights, but applied to random spatial locations
- **Entropy-Regularized:** Attention penalty encouraging more concentrated (sharper) attention maps

**Known Modifications:**

- We use Adam for both Flickr8k and Flickr30k (the paper uses RMSProp for Flickr8k)
- Images are resized to 256×256 then center-cropped to 224×224 (paper resizes shortest side to 256 with preserved aspect ratio)
- Random 85/10/5 train/val/test split (paper uses predefined Flickr8k splits and Karpathy splits for Flickr30k)

## 5. Reproduction Steps

### Requirements

- Python 3.8+
- PyTorch 1.12+
- Google Colab with GPU (T4 or better recommended)
- A [Kaggle account](https://www.kaggle.com/) with API key (`kaggle.json`) for dataset download

### Steps

1. **Clone this repository:**

   ```bash
   git clone https://github.com/<your-repo-url>.git
   cd show-attend-and-tell
   ```

2. **Download the dataset** following the instructions in `data/README.md`:

   ```bash
   kaggle datasets download -d adityajn105/flickr8k -p data/ --unzip    # for dev
   kaggle datasets download -d adityajn105/flickr30k -p data/ --unzip   # for primary eval
   ```

3. **Open the notebook in Google Colab** (or locally with a GPU):
   - `code/show_attend_tell.ipynb`: full pipeline including training, attention ablation experiments, evaluation, and visualization

4. **Set the dataset flag** in the first code cell:

   ```python
   DATASET = 'flickr8k'   # ~2 hours to train
   DATASET = 'flickr30k'  # ~8-15 hours to train
   ```

5. **Run all cells.** The notebook handles everything end-to-end: dataset download, vocabulary construction, training with early stopping, BLEU/METEOR evaluation, and attention visualization.

### Compute Requirements

| Task                                 | Estimated Time (T4 GPU) |
| ------------------------------------ | ----------------------- |
| Flickr8k full training (~30 epochs)  | ~1.5–2 hours            |
| Flickr30k full training (~30 epochs) | ~8–15 hours             |
| Evaluation + visualization           | ~1 hour                 |

Checkpoints are saved after each epoch so training can be resumed across Colab sessions.

## 6. Results / Insights

### Quantitative Results (Flickr30k)

| Model               | BLEU-1   | BLEU-2   | BLEU-3   | BLEU-4   | METEOR    |
| ------------------- | -------- | -------- | -------- | -------- | --------- |
| Paper (Soft)        | 66.7     | 43.4     | 28.8     | 19.1     | 18.49     |
| **Our Baseline**    | **62.1** | **41.9** | **29.0** | **20.1** | **20.85** |
| Shuffled Attention  | 58.1     | 37.3     | 24.6     | 16.5     | 19.05     |
| Entropy Regularized | 58.3     | 38.4     | 25.8     | 17.5     | 19.09     |

Our soft attention model achieves a BLEU-4 of 20.1 and METEOR of 20.85, exceeding the paper's METEOR (18.49) while falling slightly short on BLEU-1 (62.1 vs 66.7).

### Key Insights

- **Attention learns meaningful spatial alignment.** The shuffled-attention variant drops 3.6 BLEU-4 points compared to baseline, confirming that the spatial correspondence between attention weights and image regions is critical, not just the weighting distribution.
- **Content words vs. function words.** The model attends correctly to objects for content words (e.g., "man", "yellow", "shirt") but spreads attention diffusely for function words ("a", "the", "in", "is"). This matches the paper's qualitative findings.
- **Failure cases stem from the decoder, not attention.** When the model generates incorrect captions, attention typically points at the right region, but the decoder defaults to common training co-occurrences (e.g., predicting "smokes a cigarette" when attending to a face).
- **Entropy regularization doesn't help.** Sharpening attention via entropy penalty reduced performance, suggesting the model benefits from diffuse attention on some timesteps.

## 7. Conclusion

We successfully reimplemented the soft attention model from "Show, Attend and Tell" and achieved results competitive with the original paper on Flickr30k. Our attention ablation experiments provide evidence that the attention mechanism learns genuine spatial alignment rather than merely functioning as a feature-reweighting trick. The biggest lesson was the importance of doubly stochastic regularization since without it, attention collapses to a few spatial locations and caption quality degrades significantly. Given more time and compute, natural extensions include hard attention via REINFORCE, modern encoders (ResNet-101, ViT), and scaling to MS COCO (82K images).

## 8. References

- Xu, Kelvin, et al. "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention." _International Conference on Machine Learning (ICML)_. PMLR, 2015. [[arXiv]](https://arxiv.org/abs/1502.03044)
- Simonyan, K. and Zisserman, A. "Very Deep Convolutional Networks for Large-Scale Image Recognition." _ICLR_, 2015.
- Kingma, Diederik P. and Jimmy Ba. "Adam: A Method for Stochastic Optimization." _ICLR_, 2015.
- Flickr8k: [Kaggle (adityajn105)](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- Flickr30k: [Kaggle (adityajn105)](https://www.kaggle.com/datasets/adityajn105/flickr30k)
- Evaluation: [pycocoevalcap](https://github.com/salaniz/pycocoevalcap)
- Framework: [PyTorch](https://pytorch.org/)

## 9. Acknowledgements

This project was completed as part of **CS 4782: Introduction to Deep Learning** at Cornell University, Spring 2026, taught by the Cornell Bowers CIS faculty. We thank the course staff for their guidance and feedback throughout the project. Compute was provided via Google Colab.
