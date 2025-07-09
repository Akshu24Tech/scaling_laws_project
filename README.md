# 📏 Scaling Laws Project: Discovering the Rules of AI Building

This project is a hands-on implementation of the core ideas from the groundbreaking paper
**“Scaling Laws for Neural Language Models” by Kaplan et al. (2020)**.

## 🧠 The Big Idea: Learning the Rules of Building

Think of it like this:
The researchers at OpenAI built a giant, city-sized LEGO skyscraper. We don’t have enough bricks or space for that.

Instead, we’re building a series of **smaller LEGO models**, from a tiny house to a tall tower, to see if we can uncover the same architectural rules they did.

Our goal isn't to build the *biggest* model — it's to understand the **rules of building**.
We aim to **personally verify the paper’s key insight**:

> A model's performance improves **predictably** as you give it more "bricks" (increase its size).

---

## 🔬 What We're Testing

We focus on verifying the **first and most fundamental scaling law** from the paper:

> **Performance vs. Model Size (N)**
> As the number of parameters (**N**) increases, the loss decreases in a **predictable, power-law** fashion — visible as a **straight line on a log-log plot**.

---

## 🗂️ Project Structure

```
scaling_laws_project/
├── configs/              # Configs for different model sizes (our blueprints)
│   ├── small_model.yaml
│   └── medium_model.yaml
├── data/                 # Script to download the dataset (our instruction manuals)
│   └── prepare_data.py
├── models/               # Transformer model architecture (our LEGO bricks)
│   └── transformer.py
├── analysis/             # Notebook to visualize and analyze the results
│   └── plot_results.ipynb
├── train.py              # Main script to train all models
└── requirements.txt      # Python dependencies
```

---

## 🚀 Getting Started

### 1. 📦 Install Requirements

Install the necessary libraries:

```bash
pip install -r requirements.txt
```

---

### 2. 📚 Prepare the Dataset

The `train.py` script will **automatically download and prepare the WikiText-2 dataset** the first time you run it.
No manual steps needed.

---

### 3. 🧪 Run the Experiments

Start training the models:

```bash
python train.py
```

This will:

* Train all models defined in the `configs/` directory.
* Save results to `analysis/results.csv`.

> ⚠️ **Note:** This process may take time depending on your hardware. Be patient!

---

## 📊 Analyzing the Results

1. Open the notebook:

```bash
analysis/plot_results.ipynb
```

2. Run all cells to generate a **log-log plot** of:

> `Validation Loss` vs. `Model Parameters`

---

## ✅ What to Expect

If successful, you’ll see:

* A **downward-sloping straight line** on the log-log plot.
* This confirms the **performance improves predictably** with model size — matching the original findings.

🎉 Congratulations — you've just rediscovered one of the **core laws of AI scaling**!

---

## 📄 Reference

* **Paper**: [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)
