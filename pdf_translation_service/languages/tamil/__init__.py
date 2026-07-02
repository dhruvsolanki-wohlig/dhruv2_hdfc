"""Tamil language handler.

Uses Tamil script with Noto Sans Tamil fonts.
Contains the locked insurance glossary and Tamil-specific settings.
"""

from ..base import LanguageHandler, register_language


@register_language("Tamil")
class TamilHandler(LanguageHandler):
    """Tamil (Tamil) — isolated language handler."""

    code = "ta"
    script = "Tamil"
    is_complex_script = True

    def get_glossary(self) -> dict[str, str]:
        return {
            "Sum Assured": "காப்பீட்டுத் தொகை",
            "Death Benefit": "மரண ஆதாயம்",
            "Maturity Benefit": "அதிகாரத்துவ ஆதாயம்",
            "Survival Benefit": "உயிர்வாழ ஆதாயம்",
            "Income Benefit": "வருமான ஆதாயம்",
            "Guaranteed": "உத்தரவாத",
            "Premium": "காப்பீடு",
            "Premium Payment Term": "காப்பீட்டு பணம்செலுத்தல் காலம்",
            "Policy Term": "கொள்கை காலம்",
            "Policyholder": "கொள்கைதாரர்",
            "Life Insurance": "வாழ்க்கை காப்பீடு",
            "Maturity": "அதிகாரத்துவம்",
            "Nominee": "நியமப்பட்டவர்",
            "Rider": "ரைடர்",
            "Lumpsum": "ஒரே தவணை",
            "Annual": "ஆண்டு",
            "Half-Yearly": "அரை-ஆண்டு",
            "Monthly": "மாதாந்திர",
            "years": "ஆண்டுகள்",
            "Eligibility": "தகுதி",
            "Minimum": "குறைந்தபட்ச",
            "Maximum": "அதிகபட்ச",
            "Waiver of Premium": "காப்பீட்டு விலக்கு",
        }

    def get_prompt_extras(self) -> str:
        return (
            "\n\nTAMIL-SPECIFIC NOTES:\n- Tamil translations are significantly longer — use fit multiplier 1.3.\n- Tamil has complex chill letters (க், ச், ட்) — ensure proper shaping.\n- 'Premium' = 'காப்பீடு' (translated), not transliterated.\n- 'Guaranteed' = 'உத்தரவாத' (translated, not transliterated).\n"
        )

    def get_fit_multiplier(self) -> float:
        return 1.3

    def is_residual(self, source_text: str, translated_text: str) -> bool:
        """Tamil-specific residual check.

        Legal/disclaimer blocks correctly keep English entity names per
        Rule 7.  If the block is substantially Tamil, remaining English
        words are kept entities, not untranslated text.
        """
        script_count = sum(1 for c in translated_text if 0xb80 <= ord(c) <= 0xbff)
        latin_count = sum(1 for c in translated_text if c.isalpha() and ord(c) < 0x0900)
        total_alpha = script_count + latin_count
        if total_alpha == 0:
            return False
        if script_count / total_alpha > 0.40:
            return False
        return super().is_residual(source_text, translated_text)
