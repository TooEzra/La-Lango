# lalango/api/schemas.py
#
# Pydantic schemas define the shape of request and response data.
#
# When a user sends a POST request to /translate, FastAPI automatically
# validates that the request body matches the TranslationRequest schema.
# If it does not match, FastAPI returns a helpful error message automatically.

from pydantic import BaseModel, Field
from typing import Optional


class TranslationRequest(BaseModel):
    """
    The body of a POST /translate request.

    Example JSON:
        {
            "text": "Koso asa?",
            "source_lang": "konkani",
            "target_lang": "english"
        }
    """
    text: str = Field(
        ...,
        description="The text to translate.",
        min_length=1,
        max_length=500,
        example="Koso asa?"
    )
    source_lang: str = Field(
        ...,
        description="The source language name (e.g. 'konkani', 'yoruba').",
        example="konkani"
    )
    target_lang: str = Field(
        ...,
        description="The target language name (e.g. 'english').",
        example="english"
    )


class TranslationResponse(BaseModel):
    """
    The response body from POST /translate.

    Example JSON:
        {
            "translation": "How are you?",
            "source_lang": "konkani",
            "target_lang": "english",
            "model": "seq2seq_lstm"
        }
    """
    translation: str = Field(..., description="The translated text.")
    source_lang: str = Field(..., description="The source language.")
    target_lang: str = Field(..., description="The target language.")
    model: str = Field(..., description="Which model produced this translation.")


class LanguageInfo(BaseModel):
    """
    Information about one supported language pair.
    """
    source_lang: str
    target_lang: str
    model: str
    description: Optional[str] = None


class HealthResponse(BaseModel):
    """
    Response from GET /health.
    """
    status: str = "ok"
    message: str = "La Lango AI is running."
