"""Assamese language handler.

Uses Bengali script with Noto Sans Bengali fonts.
Contains the locked insurance glossary and Assamese-specific settings.
"""

from ..base import LanguageHandler, register_language


@register_language("Assamese")
class AssameseHandler(LanguageHandler):
    """Assamese (Bengali) — isolated language handler."""

    code = "as"
    script = "Bengali"
    is_complex_script = True

    def get_glossary(self) -> dict[str, str]:
        return {
            "Sum Assured": "বীমা সমতা",
            "Death Benefit": "মৃত্যু সুবিধা",
            "Maturity Benefit": "পৰিপক্বতা সুবিধা",
            "Survival Benefit": "জীৱিত সুবিধা",
            "Income Benefit": "আয় সুবিধা",
            "Guaranteed": "নিশ্চিত",
            "Premium": "প্ৰিমিয়াম",
            "Premium Payment Term": "প্ৰিমিয়াম পৰিশোধৰ কাল",
            "Policy Term": "নীতিৰ মিয়াদ",
            "Policyholder": "নীতিধাৰী",
            "Life Insurance": "জীৱন বীমা",
            "Maturity": "পৰিপক্বতা",
            "Nominee": "মনোনীত",
            "Rider": "সংযোজন",
            "Lumpsum": "একমুঠ",
            "Annual": "বাৰ্ষিক",
            "Half-Yearly": "অৰ্ধ-বাৰ্ষিক",
            "Monthly": "মাহিক",
            "years": "বছৰ",
            "Eligibility": "যোগ্যতা",
            "Minimum": "ন্যূনতম",
            "Maximum": "সৰ্বোচ্চ",
            "Waiver of Premium": "প্ৰিমিয়াম মকুবল",
        }

    def get_prompt_extras(self) -> str:
        return (
            "\n\nASSAMESE-SPECIFIC NOTES:\n- Assamese uses Bengali script but has distinct vocabulary.\n- 'ৰ' (with vertical stroke) is the Assamese ra, distinct from Bengali 'র'.\n- Use Assamese numerals when appropriate, but Western numerals are acceptable.\n"
        )

    def get_fit_multiplier(self) -> float:
        return 1.15

    def is_residual(self, source_text: str, translated_text: str) -> bool:
        """Assamese-specific residual check.

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
