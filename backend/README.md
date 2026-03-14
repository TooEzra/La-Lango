# Backend

This folder contains everything related to the translation engine and API.

---

## What is in here

```
backend/
├── lalango/          # The core Python package
│   ├── models/       # Translation model implementations
│   ├── tokenizers/   # Character and BPE tokenizers
│   ├── data/         # Data loading, cleaning, splitting
│   ├── evaluation/   # BLEU, chrF metrics
│   └── api/          # FastAPI REST API
│
├── scripts/          # Command-line tools (train, evaluate, preprocess)
├── tests/            # Automated tests (pytest)
├── experiments/      # Jupyter notebooks — start here if you are new
├── requirements.txt
└── requirements-dev.txt
```

---

## Setup

```bash
cd backend/
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Running the API

```bash
# Run from the project root
uvicorn backend.lalango.api.main:app --reload

# Or run from inside backend/
cd backend/
uvicorn lalango.api.main:app --reload
```

## Running the tests

```bash
# From the project root
PYTHONPATH=backend pytest backend/tests/ -v

# From inside backend/
cd backend/
PYTHONPATH=. pytest tests/ -v
```

## Running the scripts

```bash
# From the project root
PYTHONPATH=backend python backend/scripts/preprocess.py --help
PYTHONPATH=backend python backend/scripts/train.py --help
PYTHONPATH=backend python backend/scripts/evaluate.py --help

# From inside backend/
cd backend/
python scripts/preprocess.py --help
```

---

## Where to start as a contributor

1. Open `experiments/01_understanding_the_data.ipynb` in Jupyter
2. Read through the notebooks in order (01 → 02 → 03)
3. Pick a TODO in `lalango/models/seq2seq_lstm.py` (Phase 1)
4. Check [CONTRIBUTING.md](../CONTRIBUTING.md) for the PR process

---

## Import structure

All Python imports use the `lalango` package name:

```python
from lalango.tokenizers.char_tokenizer import CharTokenizer
from lalango.models.seq2seq_lstm import Seq2SeqLSTM
from lalango.evaluation.bleu import bleu_score
```

When running from the project root, prefix with `PYTHONPATH=backend`.
When running from inside `backend/`, imports work directly.
