"""Marathi language handler.

Uses Devanagari script (shared with Hindi) but has its own glossary
and Marathi-specific terminology.
"""

from ..base import LanguageHandler, register_language


@register_language("Marathi")
class MarathiHandler(LanguageHandler):
    """Marathi (Devanagari) — distinct glossary from Hindi."""

    code = "mr"
    script = "Devanagari"
    is_complex_script = True

    def get_glossary(self) -> dict[str, str]:
        return {
            "Sum Assured": "विमा रक्कम",
            "Death Benefit": "मृत्यू लाभ",
            "Maturity Benefit": "परिपक्वता लाभ",
            "Survival Benefit": "जगण्याचा लाभ",
            "Income Benefit": "उत्पन्न लाभ",
            "Guaranteed": "गारंटीड",
            "Premium": "प्रीमियम",
            "Premium Payment Term": "प्रीमियम भरता कालावधी",
            "Policy Term": "धोरण कालावधी",
            "Policyholder": "धोरणधारक",
            "Life Insurance": "जीवन विमा",
            "Maturity": "परिपक्वता",
            "Nominee": "नामनिर्देशित",
            "Rider": "रायडर",
            "Lumpsum": "एकरकमी",
            "Annual": "वार्षिक",
            "Half-Yearly": "अर्ध-वार्षिक",
            "Monthly": "मासिक",
            "years": "वर्षे",
            "Eligibility": "पात्रता",
            "Minimum": "किमान",
            "Maximum": "कमाल",
            "Waiver of Premium": "प्रीमियम सूट",
        }

    def get_prompt_extras(self) -> str:
        return (
            "\nMARATHI-SPECIFIC NOTES:\n"
            "  - Use formal Marathi, not Hindi.  Marathi has distinct vocabulary "
            "(e.g. 'वर्षे' not 'वर्ष', 'किमान' not 'न्यूनतम').\n"
            "  - 'Policy' = 'धोरण' in Marathi (not 'पॉलिसी' transliteration).\n"
            "  - 'Sum Assured' = 'विमा रक्कम' (not 'बीमा राशि' which is Hindi).\n"
        )

    def get_fit_multiplier(self) -> float:
        return 1.15

    def is_residual(self, source_text: str, translated_text: str) -> bool:
        """Same Devanagari ratio check as Hindi."""
        deva_count = sum(1 for c in translated_text if 0x0900 <= ord(c) <= 0x097F)
        latin_count = sum(1 for c in translated_text if c.isalpha() and ord(c) < 0x0900)
        total_alpha = deva_count + latin_count
        if total_alpha == 0:
            return False
        if deva_count / total_alpha > 0.40:
            return False
        return super().is_residual(source_text, translated_text)