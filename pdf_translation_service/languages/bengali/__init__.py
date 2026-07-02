"""Bengali language handler.

Uses Bengali script with Noto Sans Bengali fonts.
Contains the locked insurance glossary and Bengali-specific settings.
"""

from ..base import LanguageHandler, register_language


@register_language("Bengali")
class BengaliHandler(LanguageHandler):
    """Bengali (Bengali) — isolated language handler."""

    code = "bn"
    script = "Bengali"
    is_complex_script = True

    def get_glossary(self) -> dict[str, str]:
        return {
            "Sum Assured": "বীমা অঙ্ক",
            "Death Benefit": "মৃত্যু সুবিধা",
            "Maturity Benefit": "পরিপক্বতা সুবিধা",
            "Survival Benefit": "জীবিত সুবিধা",
            "Income Benefit": "আয় সুবিধা",
            "Guaranteed": "নিশ্চিত",
            "Premium": "প্রিমিয়াম",
            "Premium Payment Term": "প্রিমিয়াম পরিশোধের মেয়াদ",
            "Policy Term": "পলিসি মেয়াদ",
            "Policyholder": "পলিসিধারক",
            "Life Insurance": "জীবন বীমা",
            "Maturity": "পরিপক্বতা",
            "Nominee": "মনোনীত",
            "Rider": "রাইডার",
            "Lumpsum": "এককালীন",
            "Annual": "বার্ষিক",
            "Half-Yearly": "অর্ধ-বার্ষিক",
            "Monthly": "মাসিক",
            "years": "বছর",
            "Eligibility": "যোগ্যতা",
            "Minimum": "ন্যূনতম",
            "Maximum": "সর্বোচ্চ",
            "Waiver of Premium": "প্রিমিয়াম মওকুফ",
        }

    def get_prompt_extras(self) -> str:
        return (
            "\n\nBENGALI-SPECIFIC NOTES:\n- Use standard Bengali (not Assamese). Bengali 'র' vs Assamese 'ৰ'.\n- 'Policy' = 'পলিসি' (transliteration commonly used).\n"
        )

    def get_fit_multiplier(self) -> float:
        return 1.15

    def is_residual(self, source_text: str, translated_text: str) -> bool:
        """Bengali-specific residual check.

        Legal/disclaimer blocks correctly keep English entity names per
        Rule 7.  If the block is substantially Bengali, remaining English
        words are kept entities, not untranslated text.
        """
        script_count = sum(1 for c in translated_text if 0x980 <= ord(c) <= 0x9ff)
        latin_count = sum(1 for c in translated_text if c.isalpha() and ord(c) < 0x0900)
        total_alpha = script_count + latin_count
        if total_alpha == 0:
            return False
        if script_count / total_alpha > 0.40:
            return False
        return super().is_residual(source_text, translated_text)
