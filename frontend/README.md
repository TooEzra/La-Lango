# La Lango AI — Frontend

A simple, single-file web UI for the La Lango AI translation API.

Built with plain HTML, CSS, and JavaScript — no framework, no build step.
Any student who knows basic web development can read, run, and contribute to it.

---

## How to run it

**Step 1: Start the backend**

```bash
# From the project root
uvicorn lalango.api.main:app --reload
```

The API will be available at `http://localhost:8000`.

**Step 2: Open the frontend**

You have two options:

Option A — Open directly in a browser (simplest):
```
Just double-click frontend/index.html
```
> Note: Some browsers block `fetch()` calls to localhost when opening a local file.
> If translations do not work, use Option B.

Option B — Use VS Code Live Server (recommended):
1. Install the [Live Server extension](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer)
2. Right-click `frontend/index.html` → "Open with Live Server"
3. The page opens at `http://127.0.0.1:5500/frontend/index.html`

Option C — Use Python's built-in server:
```bash
cd frontend/
python -m http.server 5500
# Open http://localhost:5500
```

---

## File structure

```
frontend/
├── index.html    ← The entire UI: HTML + CSS + JavaScript in one file
└── README.md     ← This file
```

Everything is in `index.html` by design — it is easier to read and contribute to
than a project split across dozens of files.

---

## How the frontend connects to the backend

The JavaScript at the bottom of `index.html` calls three API endpoints:

| Endpoint          | When it is called                                      |
|-------------------|--------------------------------------------------------|
| `GET /health`     | On page load and every 30 seconds (API status dot)    |
| `GET /languages`  | On page load (to populate the language dropdowns)     |
| `POST /translate` | When the user clicks Translate or presses Ctrl+Enter  |

The base URL is set at the top of the script section:
```javascript
const API_BASE_URL = 'http://localhost:8000';
```

Change this if your backend runs on a different address.

---

## Contributing to the frontend

The `index.html` file is heavily commented — every section, every function is explained.

Good first issues for frontend contributors:

| Task | Difficulty |
|------|-----------|
| Add a "swap languages" button | 🟢 Easy |
| Add a translation history panel (last 5 translations) | 🔵 Medium |
| Add a character-by-character typing animation for the output | 🔵 Medium |
| Make the UI work offline with a helpful message | 🟠 Harder |
| Add dark/light mode toggle | 🟢 Easy |
| Add keyboard accessibility improvements | 🟢 Easy |
| Add confidence score display (when the model supports it) | 🟠 Harder |

Open a GitHub issue with the label `good-first-issue` before you start so no one duplicates your work.

---

## Design decisions

- **Single file** — easier to read, contribute to, and share. No build toolchain.
- **No framework** — students learn vanilla JS, which transfers to any framework later.
- **CSS variables** — the entire colour scheme is defined in `:root {}` at the top of the `<style>` block. Change four lines to retheme the whole UI.
- **Commented JS** — every function has a comment explaining what it does and why.
- **Accessible** — `aria-label` on interactive elements, `aria-live` on the output area.
