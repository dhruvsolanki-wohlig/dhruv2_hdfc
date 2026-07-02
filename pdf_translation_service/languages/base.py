"""Shared base class for language-specific handlers.

Every supported language subclasses ``LanguageHandler`` and overrides only
the hooks it needs.  The base class provides sensible defaults that match
the original (pre-isolation) behaviour, so languages that don't need
special handling get it automatically.

Design principles
------------------
* A fix for one language must never impact another.
* Language-specific logic lives inside the language's folder.
* Shared utilities (PyMuPDF, Gemini, fonts) remain in the parent package.
* The handler is the *single integration point* — the pipeline calls
  ``handler.get_config()``, ``handler.get_glossary()``, etc. instead of
  reading global config dicts.
"""

from __future__ import annotations

from typing import Any, Optional

from ..config import LANGUAGE_CONFIG, INSURANCE_GLOSSARY, COMPLEX_SCRIPTS


class LanguageHandler:
    """Base handler — default behaviour matches the original pipeline.

    Subclass and override methods to customise a language.  All methods
    return ``None`` or empty values by default, signalling "use the standard
    pipeline behaviour".
    """

    # ── Identity ──────────────────────────────────────────────────────────

    #: Human-readable language name (e.g. "Hindi", "Tamil").
    name: str = ""

    #: ISO 639-1 code (e.g. "hi", "ta").
    code: str = ""

    #: Script name as used in Unicode / Noto fonts (e.g. "Devanagari", "Tamil").
    script: str = ""

    #: Whether this script requires complex OpenType shaping (conjuncts,
    #: ligatures, reordering).  If ``True`` the pipeline uses the vector
    #: renderer (uharfbuzz) or PIL raster fallback instead of htmlbox.
    is_complex_script: bool = False

    # ── Configuration ───────────────────────────────────────────────────────

    def get_config(self) -> dict[str, Any]:
        """Return the ``LANGUAGE_CONFIG`` entry for this language.

        Override to add or modify fields (e.g. custom font file names).
        """
        return LANGUAGE_CONFIG.get(self.name, {})

    def get_font_family(self) -> str:
        """Noto font family name (e.g. 'Noto Sans Devanagari')."""
        return self.get_config().get("font_family", "Noto Sans")

    def get_font_files(self) -> dict[str, str]:
        """Return ``{"regular": "...ttf", "bold": "...ttf"}``."""
        cfg = self.get_config()
        return {
            "regular": cfg.get("regular", "NotoSans-Regular.ttf"),
            "bold": cfg.get("bold", "NotoSans-Bold.ttf"),
        }

    # ── Glossary ───────────────────────────────────────────────────────────

    def get_glossary(self) -> dict[str, str]:
        """Locked terminology map (English → target language).

        Override per-language to add insurance/domain terms.
        """
        return INSURANCE_GLOSSARY.get(self.name, {})

    # ── Translation prompt ─────────────────────────────────────────────────

    def get_prompt_extras(self) -> str:
        """Extra instructions appended to the Gemini translation prompt.

        Use this for language-specific transliteration rules, tone, or
        domain-specific guidance.  Return ``""`` for none.
        """
        return ""

    def get_fit_multiplier(self) -> float:
        """Character budget multiplier for the fit constraint.

        The pipeline sets ``budget = len(source) * multiplier``.
        Languages with longer translations (Tamil, Malayalam) should
        override this to return a higher value (e.g. 1.3 or 1.4).
        """
        return 1.1

    # ── Rendering ──────────────────────────────────────────────────────────

    def get_render_mode(self) -> str:
        """Preferred render mode: "vector", "raster", or "auto".

        "auto" (default) uses vector if uharfbuzz is available, else raster.
        """
        return "auto"

    def should_use_vector_renderer(self) -> bool:
        """Whether to attempt the vector (uharfbuzz) renderer for this language."""
        if self.get_render_mode() == "raster":
            return False
        if self.get_render_mode() == "vector":
            return True
        # auto: use vector for complex scripts
        return self.is_complex_script

    def get_line_height_multiplier(self) -> float:
        """Multiplier for line height when wrapping text.

        Some scripts (Malayalam, Tamil) need taller line spacing.
        """
        return 1.0

    def get_min_font_scale(self) -> float:
        """Minimum font shrinkage scale (0.0-1.0) before giving up.

        Scripts with complex conjuncts may need a lower floor.
        """
        return 0.55

    # ── Pre / post processing ──────────────────────────────────────────────

    def preprocess_text(self, text: str) -> str:
        """Transform source text *before* sending to Gemini.

        Override for language-specific normalisation (e.g. normalising
        Unicode forms, stripping certain characters).
        """
        return text

    def postprocess_translation(self, original: str, translated: str) -> str:
        """Transform a translation *after* Gemini returns it.

        Override for language-specific cleanup (e.g. fixing punctuation,
        removing transliteration artifacts, normalising spacing).
        """
        return translated

    def is_residual(self, source_text: str, translated_text: str) -> bool:
        """Check if a translation is residual (untranslated English).

        Default uses the shared ``is_residual_english()`` from extraction.
        Override for language-specific residual detection (e.g. legal
        text with many kept English entity names).
        """
        from ..extraction import is_residual_english
        return is_residual_english(translated_text)

    # ── Validation ─────────────────────────────────────────────────────────

    def verify_glyph_coverage(self) -> list[str]:
        """Return a list of missing-glyph issues for this language's font.

        Uses the shared ``verify_font_glyph_coverage()`` by default.
        """
        from ..fonts import verify_font_glyph_coverage
        # The shared function checks all fonts; we could filter to just
        # this language's script but the overhead is negligible.
        return verify_font_glyph_coverage()


# ── Registry & dispatcher ─────────────────────────────────────────────────

_REGISTRY: dict[str, type[LanguageHandler]] = {}


def register_language(name: str):
    """Decorator: register a ``LanguageHandler`` subclass for ``name``."""
    def decorator(cls: type[LanguageHandler]):
        cls.name = name
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_language_handler(language: str) -> LanguageHandler:
    """Return a handler instance for ``language``.

    Falls back to the base ``LanguageHandler`` (generic behaviour) if no
    registered handler exists for the language.
    """
    cls = _REGISTRY.get(language.lower())
    if cls is not None:
        handler = cls()
        # Populate is_complex_script from COMPLEX_SCRIPTS if not set
        if not handler.is_complex_script:
            handler.is_complex_script = handler.script in COMPLEX_SCRIPTS
        return handler

    # Fallback: build a generic handler from LANGUAGE_CONFIG
    handler = LanguageHandler()
    handler.name = language
    cfg = LANGUAGE_CONFIG.get(language, {})
    handler.code = cfg.get("code", "")
    handler.script = cfg.get("script", "")
    handler.is_complex_script = handler.script in COMPLEX_SCRIPTS
    return handler