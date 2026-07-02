"""Gujarati language handler.

Uses Gujarati script with Noto Sans Gujarati fonts.
Contains the locked insurance glossary and Gujarati-specific settings.
"""

from ..base import LanguageHandler, register_language


@register_language("Gujarati")
class GujaratiHandler(LanguageHandler):
    """Gujarati (Gujarati) — isolated language handler."""

    code = "gu"
    script = "Gujarati"
    is_complex_script = True

    def get_glossary(self) -> dict[str, str]:
        return {
            "Sum Assured": "વીમા રકમ",
            "Death Benefit": "મૃત્યુ લાભ",
            "Maturity Benefit": "પરિપક્વતા લાભ",
            "Survival Benefit": "જીવિત લાભ",
            "Income Benefit": "આવક લાભ",
            "Guaranteed": "ગેરંટીડ",
            "Premium": "પ્રીમિયમ",
            "Premium Payment Term": "પ્રીમિયમ ચુકવણી સમયગાળો",
            "Policy Term": "પોલિસી સમયગાળો",
            "Policyholder": "પોલિસીધારક",
            "Life Insurance": "જીવન વીમો",
            "Maturity": "પરિપક્વતા",
            "Nominee": "નામાંકિત",
            "Rider": "રાઇડર",
            "Lumpsum": "એકમુઠ્ઠી",
            "Annual": "વાર્ષિક",
            "Half-Yearly": "અર્ધ-વાર્ષિક",
            "Monthly": "માસિક",
            "years": "વર્ષ",
            "Eligibility": "પાત્રતા",
            "Minimum": "ન્યૂનતમ",
            "Maximum": "મહત્તમ",
            "Waiver of Premium": "પ્રીમિયમ માફી",
        }

    def get_prompt_extras(self) -> str:
        return (
            "\n\nGUJARATI-SPECIFIC NOTES:\n- Use standard Gujarati terminology.\n- 'Guaranteed' transliterated as 'ગેરંટીડ' (commonly used in Indian insurance).\n"
        )

    def get_fit_multiplier(self) -> float:
        return 1.15

    def is_residual(self, source_text: str, translated_text: str) -> bool:
        """Gujarati-specific residual check.

        Legal/disclaimer blocks correctly keep English entity names per
        Rule 7.  If the block is substantially Gujarati, remaining English
        words are kept entities, not untranslated text.
        """
        script_count = sum(1 for c in translated_text if 0xa80 <= ord(c) <= 0xaff)
        latin_count = sum(1 for c in translated_text if c.isalpha() and ord(c) < 0x0900)
        total_alpha = script_count + latin_count
        if total_alpha == 0:
            return False
        if script_count / total_alpha > 0.40:
            return False
        return super().is_residual(source_text, translated_text)
