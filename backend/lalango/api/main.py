# lalango/api/main.py
#
# The FastAPI application entrypoint.
#
# Run the server with:
#   uvicorn lalango.api.main:app --reload
#
# Then open http://localhost:8000/docs in your browser.
# FastAPI automatically generates an interactive API explorer there.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lalango.api.routes import router

# ---------------------------------------------------------------------------
# Create the FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="La Lango AI",
    description=(
        "A community-driven NLP platform for translating regional dialects and "
        "low-resource languages. Built from scratch by students, for the world.\n\n"
        "**GitHub:** https://github.com/Wecncode/la-lango-ai"
    ),
    version="0.1.0",
    # The /docs route will show the interactive Swagger UI
    docs_url="/docs",
    # The /redoc route shows an alternative documentation style
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing) controls which websites can call this API.
# During development, we allow all origins.
# In production, restrict this to your actual frontend domain.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow requests from any origin (development only)
    allow_methods=["*"],       # Allow GET, POST, etc.
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register routes
# ---------------------------------------------------------------------------
# All the actual endpoints (/translate, /languages, /health) are defined
# in routes.py and registered here.

app.include_router(router)

# ---------------------------------------------------------------------------
# Model registry (to be populated when models are trained)
# ---------------------------------------------------------------------------
# Store loaded models and tokenizers here so routes.py can access them.
# Key: (source_lang, target_lang) tuple
# Value: the loaded model object

loaded_models = {}
tokenizers = {}


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------
# Code here runs once when the server starts.
# This is where we load trained models into memory.

@app.on_event("startup")
def load_models():
    """
    Load all trained models into memory when the server starts.

    TODO (Phase 1 — final step):
        Once you have a trained model checkpoint, load it here.

        Example:
            import torch
            from lalango.models.seq2seq_lstm import Seq2SeqLSTM
            from lalango.tokenizers.char_tokenizer import CharTokenizer

            # Load tokenizers
            src_tokenizer = CharTokenizer()
            src_tokenizer.load("checkpoints/konkani-english/src_tokenizer.json")

            tgt_tokenizer = CharTokenizer()
            tgt_tokenizer.load("checkpoints/konkani-english/tgt_tokenizer.json")

            # Load model
            model = Seq2SeqLSTM(src_vocab_size=..., tgt_vocab_size=...)
            model.load_state_dict(torch.load("checkpoints/konkani-english/model.pt"))
            model.eval()

            # Register in the global dictionaries
            loaded_models[("konkani", "english")] = model
            tokenizers[("konkani", "english")] = (src_tokenizer, tgt_tokenizer)
    """
    print("La Lango AI server started.")
    print("No models loaded yet. Train a model using scripts/train.py first.")
    print("Open http://localhost:8000/docs to explore the API.")
