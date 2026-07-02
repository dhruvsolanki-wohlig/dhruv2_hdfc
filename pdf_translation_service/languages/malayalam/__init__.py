"""Malayalam language handler.

Uses Malayalam script with Noto Sans Malayalam fonts.
Contains the locked insurance glossary and Malayalam-specific settings.
"""

from ..base import LanguageHandler, register_language


@register_language("Malayalam")
class MalayalamHandler(LanguageHandler):
    """Malayalam (Malayalam) — isolated language handler."""

    code = "ml"
    script = "Malayalam"
    is_complex_script = True

    def get_glossary(self) -> dict[str, str]:
        return {
            "Sum Assured": "ഇൻഷുറൻസ് തുക",
            "Death Benefit": "മരണ ആനുകൂല്യം",
            "Maturity Benefit": "മെച്യൂരിറ്റി ആനുകൂല്യം",
            "Survival Benefit": "അതിജീവന ആനുകൂല്യം",
            "Income Benefit": "വരുമാന ആനുകൂല്യം",
            "Guaranteed": "ഉറപ്പാക്കിയ",
            "Premium": "പ്രീമിയം",
            "Premium Payment Term": "പ്രീമിയം അടയ്ക്കൽ കാലയളവ്",
            "Policy Term": "പോളിസി കാലയളവ്",
            "Policyholder": "പോളിസി ഉടമ",
            "Life Insurance": "ജീവൻ ഇൻഷുറൻസ്",
            "Maturity": "മെച്യൂരിറ്റി",
            "Nominee": "നാമനിർദ്ദേശം ചെയ്യപ്പെട്ടയാൾ",
            "Rider": "റൈഡർ",
            "Lumpsum": "ഒറ്റത്തവണ",
            "Annual": "വാർഷിക",
            "Half-Yearly": "അർധ-വാർഷിക",
            "Monthly": "മാസിക",
            "years": "വർഷം",
            "Eligibility": "യോഗ്യത",
            "Minimum": "ഏറ്റവും കുറഞ്ഞ",
            "Maximum": "പരമാവധി",
            "Waiver of Premium": "പ്രീമിയം ഒഴിവാക്കൽ",
        }

    def get_prompt_extras(self) -> str:
        return (
            "\n\nMALAYALAM-SPECIFIC NOTES:\n- Malayalam translations are longer than source — use fit multiplier 1.3.\n- Malayalam has complex chill letters — ensure proper shaping.\n- Use 'ഉറപ്പാക്കിയ' for 'Guaranteed' (translated, not transliterated).\n"
        )

    def get_fit_multiplier(self) -> float:
        return 1.3

    def is_residual(self, source_text: str, translated_text: str) -> bool:
        """Malayalam-specific residual check.

        Legal/disclaimer blocks correctly keep English entity names per
        Rule 7.  If the block is substantially Malayalam, remaining English
        words are kept entities, not untranslated text.
        """
        script_count = sum(1 for c in translated_text if 0xd00 <= ord(c) <= 0xd7f)
        latin_count = sum(1 for c in translated_text if c.isalpha() and ord(c) < 0x0900)
        total_alpha = script_count + latin_count
        if total_alpha == 0:
            return False
        if script_count / total_alpha > 0.40:
            return False
        return super().is_residual(source_text, translated_text)
