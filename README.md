# 🎭 Sentiment Classifier — Movie Reviews

> Learning project: PyTorch + Hugging Face finetuning  
> Based on concepts from *AI Engineering* by Chip Huyen (O'Reilly, 2024)

---

## 🎯 What this project does

Takes a movie review as input and predicts whether it is **positive** or **negative**.

```
Input:  "This movie was absolutely fantastic, I loved every minute."
Output: POSITIVE ✅

Input:  "Terrible plot, bad acting, complete waste of time."
Output: NEGATIVE ❌
```

---

## 🧠 The approach — Finetuning

We are **not training a model from scratch.** That would require billions of examples and weeks of compute.

Instead we use **finetuning**:

```
DistilBERT (already knows language)
        +
IMDB labeled reviews (our specific task)
        ↓
Finetuned model that classifies sentiment
```

> *"A small model, finetuned on a specific task, might outperform a much larger out-of-the-box model on that task."*  
> — Chip Huyen, AI Engineering

---

## 📦 The Dataset — IMDB

- **Source:** [Stanford NLP / IMDB on Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb)
- **Origin:** Real human reviews from IMDB.com (collected 2011)
- **Size:** 50,000 reviews total

| Split | Size | Purpose |
|-------|------|---------|
| `train` | 25,000 | Model learns from these |
| `test` | 25,000 | We evaluate on these — model never sees them during training |
| `unsupervised` | 50,000 | No labels — not used in this project |

### How labels were created

```
1-4 stars  on IMDB → label 0 (negative)
7-10 stars on IMDB → label 1 (positive)
5-6 stars           → dropped (ambiguous)
```

No LLM involved. Real humans, real opinions.

---

## 🤖 The Model — DistilBERT

- **Full name:** `distilbert-base-uncased`
- **From:** [Hugging Face Hub](https://huggingface.co/distilbert/distilbert-base-uncased)
- **What it is:** A smaller, faster version of BERT (66M parameters vs 110M)
- **Already knows:** Grammar, semantics, context, language structure
- **What we teach it:** Our specific task — positive vs negative sentiment

---

## 🛠️ Tech Stack

| Tool | Role |
|------|------|
| `PyTorch` | Deep learning framework — tensors, training loop, GPU |
| `transformers` | Hugging Face library — load DistilBERT |
| `datasets` | Hugging Face library — load IMDB dataset |
| `scikit-learn` | Evaluation metrics (accuracy, F1) |
| Apple MPS | Apple Silicon GPU backend for PyTorch |

---

## 📁 Project Structure

```
sentiment-project/
├── README.md            ← you are here
├── sentiment.ipynb      ← main notebook
├── .gitignore
└── sentiment-env/       ← virtual environment (not committed)
```

---

## 🗺️ Learning Roadmap

- [x] Step 1 — Environment setup & GPU check
- [x] Step 2 — Load IMDB dataset from Hugging Face
- [x] Step 3 — Tokenization (converting text → numbers)
- [x] Step 4 — Load DistilBERT + add classification head
- [x] Step 5 — Training loop
- [x] Step 6 — Evaluate on test set
- [x] Step 7 — Run inference on custom sentences

---

## 📚 Further Reading

- [AI Engineering — Chip Huyen](https://www.oreilly.com/library/view/ai-engineering/9781098166304/) — Chapter on Finetuning
- [Hugging Face — Fine-tuning a pretrained model](https://huggingface.co/docs/transformers/training)
- [DistilBERT paper](https://arxiv.org/abs/1910.01108)
- [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
- [PyTorch MPS backend](https://pytorch.org/docs/stable/notes/mps.html)



---

## 📊 Results

| Epoch | Train Loss | Val Loss | Accuracy | F1 |
|-------|-----------|----------|----------|-----|
| 1 | 0.2169 | 0.2007 | 92.3% | 92.1% |
| 2 | 0.1403 | 0.2298 | 93.3% | 93.3% |

**Final accuracy: 93.3% on 25,000 unseen reviews**

### Key observations
- Epoch 2 val loss higher than epoch 1 → early overfitting signal
- Model has no neutral class (IMDB removed 5-6 star reviews)
- Sarcasm and irony are edge cases the model handles by luck
- Training time: ~167 minutes on Apple Silicon MPS

### Inference examples
| Text | Prediction | Confidence |
|------|-----------|------------|
| "absolutely fantastic" | POSITIVE ✅ | 99.3% |
| "terrible, waste of time" | NEGATIVE ❌ | 99.4% |
| "it was okay I guess" | NEGATIVE ❌ | 92.8% |
| "so bad it was actually good" | POSITIVE ✅ | 93.0% |
