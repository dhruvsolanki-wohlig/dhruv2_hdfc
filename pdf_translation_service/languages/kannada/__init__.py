"""Kannada language handler.

Uses Kannada script with Noto Sans Kannada fonts.
Contains the locked insurance glossary and Kannada-specific settings.
"""

from ..base import LanguageHandler, register_language


@register_language("Kannada")
class KannadaHandler(LanguageHandler):
    """Kannada (Kannada) — isolated language handler."""

    code = "kn"
    script = "Kannada"
    is_complex_script = True

    def get_glossary(self) -> dict[str, str]:
        return {
            "Sum Assured": "ವಿಮಾ ಮೊತ್ತ",
            "Death Benefit": "ಮರಣ ಲಾಭ",
            "Maturity Benefit": "ಪ್ರೌಢತೆ ಲಾಭ",
            "Survival Benefit": "ಬದುಕುಳಿಯುವಿಕೆ ಲಾಭ",
            "Income Benefit": "ಆದಾಯ ಲಾಭ",
            "Guaranteed": "ಖಾತರಿ",
            "Premium": "ಪ್ರೀಮಿಯಂ",
            "Premium Payment Term": "ಪ್ರೀಮಿಯಂ ಪಾವತಿ ಅವಧಿ",
            "Policy Term": "ಪಾಲಿಸಿ ಅವಧಿ",
            "Policyholder": "ಪಾಲಿಸಿದಾರ",
            "Life Insurance": "ಜೀವ ವಿಮೆ",
            "Maturity": "ಪ್ರೌಢತೆ",
            "Nominee": "ನಾಮನಿರ್ದೇಶಿತ",
            "Rider": "ರೈಡರ್",
            "Lumpsum": "ಒಟ್ಟು",
            "Annual": "ವಾರ್ಷಿಕ",
            "Half-Yearly": "ಅರ್ಧ-ವಾರ್ಷಿಕ",
            "Monthly": "ಮಾಸಿಕ",
            "years": "ವರ್ಷಗಳು",
            "Eligibility": "ಅರ್ಹತೆ",
            "Minimum": "ಕನಿಷ್ಠ",
            "Maximum": "ಗರಿಷ್ಠ",
            "Waiver of Premium": "ಪ್ರೀಮಿಯಂ ಮನ್ನಾ",
        }

    def get_prompt_extras(self) -> str:
        return (
            "\n\nKANNADA-SPECIFIC NOTES:\n- Use formal Kannada. 'ವಿಮೆ' for insurance (not transliteration of 'insurance').\n- 'Guaranteed' = 'ಖಾತರಿ' (translated, not transliterated).\n"
        )

    def get_fit_multiplier(self) -> float:
        return 1.2

    def is_residual(self, source_text: str, translated_text: str) -> bool:
        """Kannada-specific residual check.

        Legal/disclaimer blocks correctly keep English entity names per
        Rule 7.  If the block is substantially Kannada, remaining English
        words are kept entities, not untranslated text.
        """
        script_count = sum(1 for c in translated_text if 0xc80 <= ord(c) <= 0xcff)
        latin_count = sum(1 for c in translated_text if c.isalpha() and ord(c) < 0x0900)
        total_alpha = script_count + latin_count
        if total_alpha == 0:
            return False
        if script_count / total_alpha > 0.40:
            return False
        return super().is_residual(source_text, translated_text)
