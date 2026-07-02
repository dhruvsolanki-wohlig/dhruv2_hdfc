"""Odia language handler.

Uses Odia script with Noto Sans Oriya fonts.
Contains the locked insurance glossary and Odia-specific settings.
"""

from ..base import LanguageHandler, register_language


@register_language("Odia")
class OdiaHandler(LanguageHandler):
    """Odia (Odia) — isolated language handler."""

    code = "or"
    script = "Odia"
    is_complex_script = True

    def get_glossary(self) -> dict[str, str]:
        return {
            "Sum Assured": "ବୀମା ରାଶି",
            "Death Benefit": "ମୃତ୍ୟୁ ସୁବିଧା",
            "Maturity Benefit": "ପରିପକ୍ବତା ସୁବିଧା",
            "Survival Benefit": "ଜୀବିତ ସୁବିଧା",
            "Income Benefit": "ଆୟ ସୁବିଧା",
            "Guaranteed": "ନିଶ୍ଚିତ",
            "Premium": "ପ୍ରିମିୟମ୍",
            "Premium Payment Term": "ପ୍ରିମିୟମ୍ ଦେୟ ଅବଧି",
            "Policy Term": "ପଲିସି ଅବଧି",
            "Policyholder": "ପଲିସିଧାରୀ",
            "Life Insurance": "ଜୀବନ ବୀମା",
            "Maturity": "ପରିପକ୍ବତା",
            "Nominee": "ନାମାଙ୍କିତ",
            "Rider": "ରାଇଡର୍",
            "Lumpsum": "ଏକଦା",
            "Annual": "ବାର୍ଷିକ",
            "Half-Yearly": "ଅର୍ଦ୍ଧ-ବାର୍ଷିକ",
            "Monthly": "ମାସିକ",
            "years": "ବର୍ଷ",
            "Eligibility": "ଯୋଗ୍ୟତା",
            "Minimum": "ନ୍ୟୂନତମ",
            "Maximum": "ସର୍ବୋଚ୍ଚ",
            "Waiver of Premium": "ପ୍ରିମିୟମ୍ ଛାଡ଼",
        }

    def get_prompt_extras(self) -> str:
        return (
            "\n\nODIA-SPECIFIC NOTES:\n- Use standard Odia (Oriya) script.\n- 'Guaranteed' = 'ନିଶ୍ଚିତ' (translated, not transliterated).\n"
        )

    def get_fit_multiplier(self) -> float:
        return 1.15

    def is_residual(self, source_text: str, translated_text: str) -> bool:
        """Odia-specific residual check.

        Legal/disclaimer blocks correctly keep English entity names per
        Rule 7.  If the block is substantially Odia, remaining English
        words are kept entities, not untranslated text.
        """
        script_count = sum(1 for c in translated_text if 0xb00 <= ord(c) <= 0xb7f)
        latin_count = sum(1 for c in translated_text if c.isalpha() and ord(c) < 0x0900)
        total_alpha = script_count + latin_count
        if total_alpha == 0:
            return False
        if script_count / total_alpha > 0.40:
            return False
        return super().is_residual(source_text, translated_text)
