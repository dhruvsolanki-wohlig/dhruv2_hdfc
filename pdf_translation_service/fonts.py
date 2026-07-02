from pathlib import Path
from typing import Optional

import httpx
import pymupdf
from PIL import ImageFont

from .config import FONTS_DIR, LANGUAGE_CONFIG, NOTO_FONTS_BASE, logger

_pil_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

# Latin fallback fonts (used when the script-specific font lacks Latin glyphs).
# NotoSans-Regular/Bold cover Basic Latin + many scripts, making them safe
# fallbacks for mixed-script text (e.g. "7x मृत्यू लाभ").
_LATIN_FALLBACK_REGULAR = "NotoSans-Regular.ttf"
_LATIN_FALLBACK_BOLD = "NotoSans-Bold.ttf"
# macOS / Linux system fonts tried in order when the Noto file is absent.
_SYSTEM_LATIN_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]


def _get_latin_pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Get a PIL font that covers Basic Latin — used as fallback for mixed-script text."""
    fname = _LATIN_FALLBACK_BOLD if bold else _LATIN_FALLBACK_REGULAR
    fpath = Path(FONTS_DIR) / fname
    if fpath.exists():
        try:
            return ImageFont.truetype(str(fpath), size)
        except Exception:
            pass
    for candidate in _SYSTEM_LATIN_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                continue
    return ImageFont.load_default()


async def ensure_fonts_available():
    """Download Noto fonts if not already present in FONTS_DIR."""
    fonts_dir = Path(FONTS_DIR)
    fonts_dir.mkdir(parents=True, exist_ok=True)

    needed_fonts = set()
    for lang_config in LANGUAGE_CONFIG.values():
        needed_fonts.add(lang_config["regular"])
        needed_fonts.add(lang_config["bold"])
    # Latin fallback fonts for mixed-script PIL rendering
    needed_fonts.add(_LATIN_FALLBACK_REGULAR)
    needed_fonts.add(_LATIN_FALLBACK_BOLD)

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        for font_file in sorted(needed_fonts):
            font_path = fonts_dir / font_file
            if font_path.exists():
                continue
            font_family_name = font_file.rsplit("-", 1)[0]
            url = f"{NOTO_FONTS_BASE}/{font_family_name}/hinted/ttf/{font_file}"
            logger.info(f"Downloading font: {font_file}")
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    font_path.write_bytes(response.content)
                    logger.info(f"  -> Downloaded: {font_file}")
                else:
                    logger.warning(f"  -> Failed to download {font_file}: HTTP {response.status_code}")
            except Exception as e:
                logger.warning(f"  -> Error downloading {font_file}: {e}")

    verify_font_glyph_coverage()


# Representative base codepoints per script — if the font can render these, it
# covers the script block. (Conjuncts are shaped from these, so per-codepoint
# coverage of the bases is the right granularity.)
_SCRIPT_PROBES = {
    "Devanagari": "अक",
    "Gujarati": "અક",
    "Kannada": "ಅಕ",
    "Tamil": "அக",
    "Telugu": "అక",
    "Bengali": "অক",
    "Malayalam": "അക",
    "Gurmukhi": "ਅਕ",
    "Odia": "ଅକ",
}


def verify_font_glyph_coverage() -> list[str]:
    """Fail-fast safety check: confirm each configured font actually has glyphs
    for its script. Logs an ERROR per gap (so missing/corrupt fonts surface at
    startup instead of silently producing tofu in output). Returns the issues."""
    fonts_dir = Path(FONTS_DIR)
    issues: list[str] = []
    checked: set[str] = set()
    for lang, cfg in LANGUAGE_CONFIG.items():
        probe = _SCRIPT_PROBES.get(cfg["script"])
        if not probe:
            continue
        font_file = cfg["regular"]
        if font_file in checked:
            continue
        checked.add(font_file)
        font_path = fonts_dir / font_file
        if not font_path.exists():
            issues.append(f"{cfg['script']}: font file missing ({font_file})")
            continue
        try:
            f = pymupdf.Font(fontfile=str(font_path))
            missing = [ch for ch in probe if not f.has_glyph(ord(ch))]
            if missing:
                issues.append(f"{cfg['script']} ({font_file}): missing glyphs for {missing}")
        except Exception as e:
            issues.append(f"{cfg['script']} ({font_file}): load error {e}")
    if issues:
        for i in issues:
            logger.error(f"FONT COVERAGE: {i}")
    else:
        logger.info(f"Font glyph coverage verified for {len(checked)} font file(s) across all scripts")
    return issues


def get_pil_font(language: str, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Get a PIL ImageFont for the given language, size, and weight. Cached."""
    config = LANGUAGE_CONFIG.get(language)
    if not config:
        return ImageFont.load_default()
    font_file = config["bold"] if bold else config["regular"]
    cache_key = (font_file, size)
    if cache_key in _pil_font_cache:
        return _pil_font_cache[cache_key]
    font_path = Path(FONTS_DIR) / font_file
    if font_path.exists():
        try:
            font = ImageFont.truetype(str(font_path), size)
            _pil_font_cache[cache_key] = font
            return font
        except Exception:
            pass
    return ImageFont.load_default()


def get_font_css(language: str) -> str:
    """Generate @font-face CSS for PyMuPDF's insert_htmlbox (text mode only)."""
    config = LANGUAGE_CONFIG.get(language)
    if not config:
        return "* { font-family: sans-serif; }"
    regular_file = config["regular"]
    bold_file = config["bold"]
    family = config["font_family"]
    fonts_dir = Path(FONTS_DIR).resolve()
    css = ""
    if (fonts_dir / regular_file).exists():
        css += (
            f"@font-face {{"
            f" font-family: '{family}';"
            f" src: url('{regular_file}');"
            f" font-weight: normal;"
            f"}}\n"
        )
    if (fonts_dir / bold_file).exists():
        css += (
            f"@font-face {{"
            f" font-family: '{family}';"
            f" src: url('{bold_file}');"
            f" font-weight: bold;"
            f"}}\n"
        )
    css += f"* {{ font-family: '{family}', sans-serif; }}\n"
    return css


_pymupdf_font_cache: dict[str, "pymupdf.Font"] = {}


def get_pymupdf_font(language: str, bold: bool = False) -> Optional["pymupdf.Font"]:
    """Get a cached pymupdf.Font for the language's script (for the invisible
    selectable-text layer written via TextWriter). Returns None if unavailable."""
    config = LANGUAGE_CONFIG.get(language)
    if not config:
        return None
    fname = config["bold"] if bold else config["regular"]
    if fname in _pymupdf_font_cache:
        return _pymupdf_font_cache[fname]
    path = Path(FONTS_DIR) / fname
    if not path.exists():
        return None
    try:
        f = pymupdf.Font(fontfile=str(path))
        _pymupdf_font_cache[fname] = f
        return f
    except Exception:
        return None


def get_latin_pymupdf_font(bold: bool = False) -> Optional["pymupdf.Font"]:
    """Cached pymupdf.Font covering Latin (for non-Indic words in the invisible
    text layer — kept English, numerics, hashtags)."""
    fname = _LATIN_FALLBACK_BOLD if bold else _LATIN_FALLBACK_REGULAR
    if fname in _pymupdf_font_cache:
        return _pymupdf_font_cache[fname]
    path = Path(FONTS_DIR) / fname
    if path.exists():
        try:
            f = pymupdf.Font(fontfile=str(path))
            _pymupdf_font_cache[fname] = f
            return f
        except Exception:
            pass
    # Fall back to a built-in font (Helvetica covers Latin) so the invisible
    # layer still works when the Noto Latin file is absent.
    try:
        f = pymupdf.Font("helv")
        _pymupdf_font_cache[fname] = f
        return f
    except Exception:
        return None


def get_font_archive() -> Optional[pymupdf.Archive]:
    """Get a PyMuPDF Archive containing the font directory (text mode only)."""
    fonts_dir = Path(FONTS_DIR)
    if fonts_dir.exists():
        try:
            return pymupdf.Archive(str(fonts_dir))
        except Exception as e:
            logger.error(f"Failed to create font Archive from {fonts_dir}: {e}")
            return None
    logger.warning(f"Fonts directory not found: {fonts_dir}")
    return None
