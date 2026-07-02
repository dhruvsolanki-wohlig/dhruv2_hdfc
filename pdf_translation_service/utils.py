import uuid
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import CDN_URL, UPLOAD_URL, logger  # noqa: F401 — CDN_URL re-exported for api.py

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}


def is_image_file(filename: str) -> bool:
    """Check if filename has an image extension."""
    return Path(filename.lower()).suffix in IMAGE_EXTENSIONS


def hex_from_int(color_int: int) -> str:
    r = (color_int >> 16) & 0xFF
    g = (color_int >> 8) & 0xFF
    b = color_int & 0xFF
    return f"#{r:02x}{g:02x}{b:02x}"


def rgb_tuple_from_hex(hex_color: str) -> tuple[int, int, int]:
    """Convert '#rrggbb' to (r, g, b) ints 0-255."""
    h = hex_color.lstrip('#')
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_from_int(color_int: int) -> tuple:
    r = ((color_int >> 16) & 0xFF) / 255.0
    g = ((color_int >> 8) & 0xFF) / 255.0
    b = (color_int & 0xFF) / 255.0
    return (r, g, b)


def is_bold(font_name: str) -> bool:
    name = font_name.lower()
    return any(w in name for w in ("bold", "extrabold", "heavy", "black", "semibold", "demibold"))


def is_italic(font_name: str) -> bool:
    name = font_name.lower()
    return any(w in name for w in ("italic", "oblique", "inclined"))


def rect_to_list(rect) -> list[float]:
    return [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)]


def clean_symbol_text(text: str) -> str:
    """Map symbol-font (Wingdings/Symbol) glyphs to renderable Unicode.

    Source PDFs encode dingbat bullets in the Private Use Area — e.g. a Wingdings
    bullet is U+F0A7 (= 0xF000 + 0xA7) and the symbol-font space is U+F020. Noto
    fonts have NO glyph for the PUA, so these survive translation and render as
    tofu boxes (▯) in both the vector and raster paths (the vector renderer draws
    the .notdef box; the invisible TextWriter layer silently drops them, which is
    why they don't show up in get_text). Normalise them: known bullet codepoints →
    '•' (NotoSans has U+2022), symbol-font space → a real space, and any other
    F0xx dingbat (arrows/checks with no Noto glyph) is dropped.
    """
    if not text:
        return text
    out = []
    for ch in text:
        cp = ord(ch)
        if cp in (0xF0A7, 0xF0B7):
            out.append("•")          # Wingdings/Symbol bullet → •
        elif cp == 0xF020:
            out.append(" ")               # symbol-font space → real space
        elif 0xF000 <= cp <= 0xF0FF:
            continue                      # other symbol-font dingbats: no Noto glyph, drop
        else:
            out.append(ch)
    return "".join(out)


def should_translate(text: str) -> bool:
    """Check if text should be translated (skip pure numbers, single chars, symbols).

    Conservative: only reject if the text is ENTIRELY numeric/symbolic after
    removing whitespace.  Keeping parentheses, slashes, colons, currency symbols
    etc. in the check ensures that strings like 'Premium Payment Term (Years)'
    or '18 years' are NOT rejected — they contain alpha chars and should be
    translated.
    """
    if not text or not text.strip():
        return False
    stripped = text.strip()
    if len(stripped) <= 1:
        return False
    # Only remove whitespace for the digit check — a date like "18 years" has
    # digits + letters; only pure "60" should be skipped.
    no_space = stripped.replace(" ", "")
    if no_space.isdigit():
        return False
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if alpha_count < 2:
        return False
    return True


async def upload_to_gcp(
    file_bytes: bytes,
    original_filename: str,
    mime_type: str = "application/pdf",
) -> str:
    new_filename = f"{uuid.uuid4()}{Path(original_filename).suffix}"
    async with httpx.AsyncClient(timeout=120.0) as client:
        files = {"file": (new_filename, file_bytes, mime_type)}
        response = await client.post(f"{UPLOAD_URL}/api/upload", files=files)
        if response.status_code != 200:
            raise Exception(f"GCP upload failed: HTTP {response.status_code} - {response.text}")
        cdn_url = f"https://storage.googleapis.com/pocketstudio/{new_filename}"
        logger.info(f"Uploaded to GCP: {cdn_url}")
        return cdn_url


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def upload_to_gcp_with_retry(
    file_bytes: bytes,
    original_filename: str,
    mime_type: str = "application/pdf",
) -> str:
    return await upload_to_gcp(file_bytes, original_filename, mime_type)
