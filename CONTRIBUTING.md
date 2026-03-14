# Contributing to La Lango AI

First off — thank you for being here. Every contribution matters, whether it is fixing a typo,
adding a new language, or implementing a new model architecture.

This guide will walk you through everything you need to know.

---

## Table of contents

1. [Code of conduct](#code-of-conduct)
2. [How to find something to work on](#how-to-find-something-to-work-on)
3. [Setting up your development environment](#setting-up-your-development-environment)
4. [Making a contribution](#making-a-contribution)
5. [Pull request checklist](#pull-request-checklist)
6. [Coding style](#coding-style)
7. [Adding a new language](#adding-a-new-language)
8. [Questions?](#questions)

---

## Code of conduct

We follow the [Contributor Covenant](CODE_OF_CONDUCT.md). Be kind, be patient, be respectful.
We are all learning here.

---

## How to find something to work on

Go to the [Issues](https://github.com/Wecncode/la-lango-ai/issues) tab and filter by label:

| Label                 | What it means                                               |
|-----------------------|-------------------------------------------------------------|
| 🟢 `good-first-issue` | Small, well-defined tasks. Perfect if this is your first PR |
| 🔵 `data`             | Data cleaning, tokenizers, preprocessing scripts            |
| 🟠 `model`            | Model architecture, training loop improvements              |
| 🔴 `research`         | Evaluation metrics, benchmarking, experiments               |
| `help wanted`         | We are stuck and need fresh eyes                            |
| `documentation`       | Improving guides, docstrings, comments                      |

> **Tip:** Comment on an issue before starting work so two people do not do the same thing.

---

## Setting up your development environment

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/la-lango-ai.git
cd la-lango-ai

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Install dev dependencies (linting, testing)
pip install -r backend/requirements-dev.txt

# 5. Run the tests to make sure everything is working
PYTHONPATH=backend pytest backend/tests/
```

---

## Making a contribution

```bash
# 1. Create a new branch for your work
#    Use a descriptive name
git checkout -b feature/add-konkani-tokenizer

# 2. Make your changes

# 3. Run the tests
PYTHONPATH=backend pytest backend/tests/

# 4. Commit your changes
#    Write a clear commit message (see examples below)
git commit -m "feat: add character tokenizer for Konkani"

# 5. Push to your fork
git push origin feature/add-konkani-tokenizer

# 6. Open a Pull Request on GitHub
```

### Commit message examples

```
feat: add Bahdanau attention to LSTM encoder
fix: handle empty string input in char tokenizer
docs: add guide for adding a new language
test: add unit tests for BLEU score calculation
data: add preprocessing script for Yoruba corpus
```

---

## Pull request checklist

Before submitting your PR, make sure you can check all of these:

- [ ] My code runs without errors
- [ ] I have added or updated tests for my changes
- [ ] All existing tests still pass (`PYTHONPATH=backend pytest backend/tests/`)
- [ ] I have added comments explaining what my code does
- [ ] If I added a new language, I followed the [adding a language guide](docs/adding_a_language.md)
- [ ] My branch is up to date with `main`

---

## Coding style

- Use **plain functions** where possible. Classes when necessary.
- **Comment your code** — explain *why*, not just *what*.
- Variable names should be readable: `source_sentence` not `ss`.
- Keep functions short. If a function is longer than 40 lines, consider splitting it.
- We use `flake8` for linting. Run `flake8 backend/lalango/` before submitting.

---

## Adding a new language

Adding a language is one of the most impactful contributions you can make.

See the full guide here: [docs/adding_a_language.md](docs/adding_a_language.md)

The short version:
1. Create a folder under `languages/your-language-name/`
2. Add a `config.json` with language metadata
3. Add a `README.md` with notes about the language
4. Open a PR with the label `data`

---

## Questions?

- Open a [GitHub Discussion](https://github.com/Wecncode/la-lango-ai/discussions)
- Or comment on the relevant issue

We respond to all questions. No question is too basic.
