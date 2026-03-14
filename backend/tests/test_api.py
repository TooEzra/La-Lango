# tests/test_api.py
#
# Tests for the FastAPI endpoints.
#
# We use FastAPI's built-in TestClient, which lets us make HTTP requests
# to the app without running a real server.

from fastapi.testclient import TestClient
from lalango.api.main import app

# Create a test client — this simulates sending HTTP requests to the API
client = TestClient(app)


class TestHealthEndpoint:

    def test_health_returns_200(self):
        """The /health endpoint should always return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status_ok(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_response_has_message(self):
        response = client.get("/health")
        data = response.json()
        assert "message" in data


class TestLanguagesEndpoint:

    def test_languages_returns_200(self):
        response = client.get("/languages")
        assert response.status_code == 200

    def test_languages_returns_a_list(self):
        response = client.get("/languages")
        data = response.json()
        assert isinstance(data, list)

    def test_each_language_has_required_fields(self):
        response = client.get("/languages")
        for lang in response.json():
            assert "source_lang" in lang
            assert "target_lang" in lang
            assert "model" in lang


class TestTranslateEndpoint:

    def test_translate_unsupported_pair_returns_404(self):
        """Requesting an unsupported language pair should return HTTP 404."""
        response = client.post("/translate", json={
            "text": "Hello world",
            "source_lang": "klingon",
            "target_lang": "elvish",
        })
        assert response.status_code == 404

    def test_translate_empty_text_returns_422(self):
        """An empty text field should fail Pydantic validation with HTTP 422."""
        response = client.post("/translate", json={
            "text": "",
            "source_lang": "english",
            "target_lang": "french",
        })
        assert response.status_code == 422

    def test_translate_missing_fields_returns_422(self):
        """Missing required fields should return HTTP 422."""
        response = client.post("/translate", json={
            "text": "Hello",
            # Missing source_lang and target_lang
        })
        assert response.status_code == 422

    def test_translate_text_too_long_returns_422(self):
        """Text longer than 500 characters should fail validation."""
        response = client.post("/translate", json={
            "text": "A" * 501,
            "source_lang": "english",
            "target_lang": "french",
        })
        assert response.status_code == 422

    def test_translate_valid_request_structure(self):
        """
        This test documents the expected request and response structure.
        It will pass once a language pair is added to SUPPORTED_PAIRS in routes.py.

        To make this test pass:
            1. Train a model for a language pair
            2. Add it to SUPPORTED_PAIRS in lalango/api/routes.py
            3. Connect the model in lalango/api/main.py

        This test is intentionally skipped until a language pair exists.
        """
        # Once you add a language pair, replace the lines below:
        #
        # response = client.post("/translate", json={
        #     "text": "Hello",
        #     "source_lang": "english",
        #     "target_lang": "your_language",
        # })
        # assert response.status_code == 200
        # data = response.json()
        # assert "translation" in data
        # assert "source_lang" in data
        # assert "target_lang" in data
        # assert "model" in data
        pass
