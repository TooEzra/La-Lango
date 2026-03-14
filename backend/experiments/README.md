# Experiments

This folder contains Jupyter notebooks that walk through each phase of the project.

They are meant to be **read, run, and learned from** — not just executed blindly.
Think of them as interactive tutorials that complement the source code.

---

## Notebooks

| Notebook | Phase | What it covers |
|----------|-------|----------------|
| `01_baseline_lstm.ipynb` | Phase 1 | Train a Seq2Seq LSTM on a toy dataset, visualize the loss |
| `02_attention_visualization.ipynb` | Phase 2 | *(coming soon)* Visualize attention weights as a heatmap |
| `03_bpe_tokenizer.ipynb` | Phase 3 | *(coming soon)* Walk through BPE vocabulary building step by step |
| `04_transformer_walkthrough.ipynb` | Phase 4 | *(coming soon)* Build and train a Transformer from scratch |

---

## How to run a notebook

```bash
# Make sure you have installed the dev dependencies
pip install -r requirements-dev.txt

# Start Jupyter
jupyter notebook experiments/

# Or Jupyter Lab if you prefer
jupyter lab experiments/
```

---

## Contributing a notebook

Have an interesting experiment or visualization? Add it here!

Good notebook contributions:
- Walk through a concept step by step with explanation
- Show something surprising or counterintuitive
- Visualize something that is hard to understand from code alone
- Benchmark two approaches side by side

Label your PR with `good-first-issue` or `research` depending on complexity.
