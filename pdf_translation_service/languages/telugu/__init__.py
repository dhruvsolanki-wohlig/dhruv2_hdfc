"""Telugu language handler.

Uses Telugu script with Noto Sans Telugu fonts.
Contains the locked insurance glossary and Telugu-specific settings.
"""

from ..base import LanguageHandler, register_language


@register_language("Telugu")
class TeluguHandler(LanguageHandler):
    """Telugu (Telugu) — isolated language handler."""

    code = "te"
    script = "Telugu"
    is_complex_script = True

    def get_glossary(self) -> dict[str, str]:
        return {
            "Sum Assured": "భీమా మొత్తం",
            "Death Benefit": "మరణ ప్రయోజనం",
            "Maturity Benefit": "పరిపక్వత ప్రయోజనం",
            "Survival Benefit": "జీవిత ప్రయోజనం",
            "Income Benefit": "ఆదాయ ప్రయోజనం",
            "Guaranteed": "హామీ",
            "Premium": "ప్రీమియం",
            "Premium Payment Term": "ప్రీమియం చెల్లింపు కాలం",
            "Policy Term": "పాలసీ కాలం",
            "Policyholder": "పాలసీదారు",
            "Life Insurance": "జీవిత భీమా",
            "Maturity": "పరిపక్వత",
            "Nominee": "నామినీ",
            "Rider": "రైడర్",
            "Lumpsum": "మొత్తం",
            "Annual": "వార్షిక",
            "Half-Yearly": "అర్ధ-వార్షిక",
            "Monthly": "నెలవారీ",
            "years": "సంవత్సరాలు",
            "Eligibility": "అర్హత",
            "Minimum": "కనీసం",
            "Maximum": "గరిష్ట",
            "Waiver of Premium": "ప్రీమియం మినహాయింపు",
        }

    def get_prompt_extras(self) -> str:
        return (
            "\n\nTELUGU-SPECIFIC NOTES:\n- Telugu has complex conjunct consonants — ensure proper shaping.\n- 'Guaranteed' = 'హామీ' (translated, not transliterated).\n- 'Premium' transliterated as 'ప్రీమియం' (commonly used).\n"
        )

    def get_fit_multiplier(self) -> float:
        return 1.2

    def is_residual(self, source_text: str, translated_text: str) -> bool:
        """Telugu-specific residual check.

        Legal/disclaimer blocks correctly keep English entity names per
        Rule 7.  If the block is substantially Telugu, remaining English
        words are kept entities, not untranslated text.
        """
        script_count = sum(1 for c in translated_text if 0xc00 <= ord(c) <= 0xc7f)
        latin_count = sum(1 for c in translated_text if c.isalpha() and ord(c) < 0x0900)
        total_alpha = script_count + latin_count
        if total_alpha == 0:
            return False
        if script_count / total_alpha > 0.40:
            return False
        return super().is_residual(source_text, translated_text)
