"""Punjabi language handler.

Uses Gurmukhi script with Noto Sans Gurmukhi fonts.
Contains the locked insurance glossary and Punjabi-specific settings.
"""

from ..base import LanguageHandler, register_language


@register_language("Punjabi")
class PunjabiHandler(LanguageHandler):
    """Punjabi (Gurmukhi) — isolated language handler."""

    code = "pa"
    script = "Gurmukhi"
    is_complex_script = True

    def get_glossary(self) -> dict[str, str]:
        return {
            "Sum Assured": "ਬੀਮਾ ਰਕਮ",
            "Death Benefit": "ਮੌਤ ਲਾਭ",
            "Maturity Benefit": "ਪਰਿਪੱਕਤਾ ਲਾਭ",
            "Survival Benefit": "ਜੀਵਤ ਲਾਭ",
            "Income Benefit": "ਆਮਦਨ ਲਾਭ",
            "Guaranteed": "ਗਰੰਟੀਡ",
            "Premium": "ਪ੍ਰੀਮੀਅਮ",
            "Premium Payment Term": "ਪ੍ਰੀਮੀਅਮ ਭੁਗਤਾਨ ਮਿਆਦ",
            "Policy Term": "ਪਾਲਿਸੀ ਮਿਆਦ",
            "Policyholder": "ਪਾਲਿਸੀਧਾਰਕ",
            "Life Insurance": "ਜੀਵਨ ਬੀਮਾ",
            "Maturity": "ਪਰਿਪੱਕਤਾ",
            "Nominee": "ਨਾਮਜ਼ਦ",
            "Rider": "ਰਾਈਡਰ",
            "Lumpsum": "ਇਕਤਰਾ",
            "Annual": "ਸਾਲਾਨਾ",
            "Half-Yearly": "ਅਰਧ-ਸਾਲਾਨਾ",
            "Monthly": "ਮਾਸਿਕ",
            "years": "ਸਾਲ",
            "Eligibility": "ਯੋਗਤਾ",
            "Minimum": "ਘੱਟੋ-ਘੱਟ",
            "Maximum": "ਵੱਧੋ-ਵੱਧ",
            "Waiver of Premium": "ਪ੍ਰੀਮੀਅਮ ਮਾਫੀ",
        }

    def get_prompt_extras(self) -> str:
        return (
            "\n\nPUNJABI-SPECIFIC NOTES:\n- Use Gurmukhi script (not Shahmukhi).\n- 'Guaranteed' transliterated as 'ਗਰੰਟੀਡ' (commonly used).\n"
        )

    def get_fit_multiplier(self) -> float:
        return 1.15

    def is_residual(self, source_text: str, translated_text: str) -> bool:
        """Punjabi-specific residual check.

        Legal/disclaimer blocks correctly keep English entity names per
        Rule 7.  If the block is substantially Gurmukhi, remaining English
        words are kept entities, not untranslated text.
        """
        script_count = sum(1 for c in translated_text if 0xa00 <= ord(c) <= 0xa7f)
        latin_count = sum(1 for c in translated_text if c.isalpha() and ord(c) < 0x0900)
        total_alpha = script_count + latin_count
        if total_alpha == 0:
            return False
        if script_count / total_alpha > 0.40:
            return False
        return super().is_residual(source_text, translated_text)
