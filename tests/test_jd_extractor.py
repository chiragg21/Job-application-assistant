"""Tests for app/core/jd_extractor.py"""
import pytest
import httpx
from unittest.mock import patch, MagicMock

from app.core.jd_extractor import extract_jd_from_url


def _mock_ok_response(text: str = "Senior Python Engineer at Acme Corp.") -> MagicMock:
    resp = MagicMock()
    resp.text = text
    resp.raise_for_status = MagicMock()   # no-op: 200 OK
    return resp


class TestExtractJdFromUrl:

    def test_returns_extracted_text(self):
        expected = "Senior ML Engineer\nRequirements: Python, PyTorch"
        with patch("httpx.get", return_value=_mock_ok_response(expected)):
            assert extract_jd_from_url("https://example.com/job") == expected

    def test_prepends_https_when_no_scheme(self):
        with patch("httpx.get", return_value=_mock_ok_response()) as mock_get:
            extract_jd_from_url("linkedin.com/jobs/123")
            called_url = mock_get.call_args[0][0]
            assert called_url.startswith("https://r.jina.ai/https://")

    def test_does_not_double_prepend_https(self):
        with patch("httpx.get", return_value=_mock_ok_response()) as mock_get:
            extract_jd_from_url("https://linkedin.com/jobs/123")
            called_url = mock_get.call_args[0][0]
            assert called_url.count("https://") == 2   # one for jina, one for original
            assert "r.jina.ai/https://linkedin.com" in called_url

    def test_http_error_raises_value_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        exc = httpx.HTTPStatusError("Not found", request=MagicMock(), response=mock_resp)
        with patch("httpx.get", side_effect=exc):
            with pytest.raises(ValueError, match="404"):
                extract_jd_from_url("https://example.com/missing-job")

    def test_timeout_raises_value_error(self):
        with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
            with pytest.raises(ValueError, match="timed out"):
                extract_jd_from_url("https://example.com/slow-job")

    def test_network_error_raises_value_error(self):
        with patch("httpx.get", side_effect=httpx.RequestError("connection refused")):
            with pytest.raises(ValueError, match="connection refused"):
                extract_jd_from_url("https://example.com/job")

    def test_empty_response_raises_value_error(self):
        with patch("httpx.get", return_value=_mock_ok_response("")):
            with pytest.raises(ValueError, match="empty content"):
                extract_jd_from_url("https://example.com/empty-job")

    def test_whitespace_only_response_raises_value_error(self):
        with patch("httpx.get", return_value=_mock_ok_response("   \n\t  ")):
            with pytest.raises(ValueError, match="empty content"):
                extract_jd_from_url("https://example.com/whitespace")

    def test_passes_plain_text_accept_header(self):
        with patch("httpx.get", return_value=_mock_ok_response()) as mock_get:
            extract_jd_from_url("https://example.com/job")
            _, kwargs = mock_get.call_args
            assert kwargs["headers"]["Accept"] == "text/plain"

    def test_passes_timeout(self):
        with patch("httpx.get", return_value=_mock_ok_response()) as mock_get:
            extract_jd_from_url("https://example.com/job")
            _, kwargs = mock_get.call_args
            assert kwargs["timeout"] == 30
