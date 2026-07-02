"""Hindi language handler.

Uses Devanagari script with Noto Sans Devanagari fonts.
Contains the locked insurance glossary and Hindi-specific settings.
"""

from ..base import LanguageHandler, register_language


@register_language("Hindi")
class HindiHandler(LanguageHandler):
    """Hindi (Devanagari) — the primary language with full glossary support."""

    code = "hi"
    script = "Devanagari"
    is_complex_script = True

    def get_glossary(self) -> dict[str, str]:
        return {
            "Sum Assured": "बीमा राशि",
            "Death Benefit": "मृत्यु लाभ",
            "Maturity Benefit": "परिपक्वता लाभ",
            "Survival Benefit": "उत्तरजीविता लाभ",
            "Income Benefit": "आय लाभ",
            "Guaranteed": "गारंटीड",
            "Premium": "प्रीमियम",
            "Premium Payment Term": "प्रीमियम भुगतान अवधि",
            "Policy Term": "पॉलिसी अवधि",
            "Policyholder": "पॉलिसीधारक",
            "Life Insurance": "जीवन बीमा",
            "Maturity": "परिपक्वता",
            "Nominee": "नामिती",
            "Rider": "राइडर",
            "Lumpsum": "एकमुश्त",
            "Annual": "वार्षिक",
            "Half-Yearly": "अर्ध-वार्षिक",
            "Monthly": "मासिक",
            "years": "वर्ष",
            "Eligibility": "पात्रता",
            "Minimum": "न्यूनतम",
            "Maximum": "अधिकतम",
            "Waiver of Premium": "प्रीमियम की छूट",
        }

    def get_prompt_extras(self) -> str:
        return (
            "\nHINDI-SPECIFIC NOTES:\n"
            "  - Transliterate 'Sampoorna Jeevan' as 'सम्पूर्ण जीवन' (not 'संपूर्ण').\n"
            "  - Use 'वर्ष' for 'years' (not 'साल' in formal/insurance context).\n"
            "  - 'Guaranteed' should be transliterated as 'गारंटीड' (commonly used "
            "in Indian insurance documents, not translated as 'गारंटीकृत').\n"
        )

    def get_fit_multiplier(self) -> float:
        return 1.1

    def get_min_font_scale(self) -> float:
        return 0.55

    def is_residual(self, source_text: str, translated_text: str) -> bool:
        """Hindi-specific residual check.

        Legal/disclaimer blocks correctly keep English entity names per
        Rule 7.  If the block is substantially Hindi, remaining English
        words are kept entities, not untranslated text.
        """
        # Count Devanagari vs Latin characters
        deva_count = sum(
            1 for c in translated_text
            if 0x0900 <= ord(c) <= 0x097F
        )
        latin_count = sum(
            1 for c in translated_text
            if c.isalpha() and ord(c) < 0x0900
        )
        total_alpha = deva_count + latin_count
        if total_alpha == 0:
            return False
        # If >40% of alpha chars are Devanagari, the block IS translated —
        # remaining Latin is kept entities (company names, CIN, addresses).
        if deva_count / total_alpha > 0.40:
            return False
        # Otherwise use the default check
        return super().is_residual(source_text, translated_text)