# Data Directory

This folder holds your datasets. **It is not tracked by git** (see `.gitignore`).
Every contributor manages their own data locally.

---

## Expected structure

```
data/
├── raw/
│   └── konkani-english/
│       ├── all.src          ← (optional) combined source file before splitting
│       ├── all.tgt          ← (optional) combined target file before splitting
│       ├── train.src
│       ├── train.tgt
│       ├── val.src
│       ├── val.tgt
│       ├── test.src
│       └── test.tgt
│
└── processed/
    └── konkani-english/
        ├── train.src        ← cleaned, ready for training
        ├── train.tgt
        ├── val.src
        ├── val.tgt
        ├── test.src
        └── test.tgt
```

---

## How to prepare your data

1. Read the [data format guide](../docs/data_format.md)
2. Place your raw files in `data/raw/<your-lang-pair>/`
3. Run the preprocessing script:

```bash
python scripts/preprocess.py \
  --src data/raw/konkani-english/all.src \
  --tgt data/raw/konkani-english/all.tgt \
  --output data/processed/konkani-english/
```

---

## Where to find data

Looking for a parallel corpus for your language? Try:

- **OPUS** — http://opus.nlpl.eu (large collection of multilingual corpora)
- **AI4Bharat** — https://ai4bharat.org (Indian language datasets)
- **Masakhane** — https://www.masakhane.io (African language NLP)
- **Common Voice** — https://commonvoice.mozilla.org (community-collected speech data)
- University linguistics departments in your region
- Local NGOs, government translation offices, or religious organizations
