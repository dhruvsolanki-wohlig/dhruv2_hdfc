import asyncio
import json
import os
from typing import Optional

from google import genai

from .config import (
    GEMINI_BATCH_SIZE,
    GEMINI_MODEL,
    GEMINI_SEMAPHORE,
    GOOGLE_CLOUD_LOCATION,
    GOOGLE_CLOUD_PROJECT,
    get_glossary,
    logger,
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

gemini_client: Optional[genai.Client] = None


def init_gemini_client():
    """Initialize Gemini client — Vertex AI when credentials available, else API key fallback."""
    global gemini_client
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    sa_exists = bool(sa_path and os.path.isfile(sa_path))

    if GOOGLE_CLOUD_PROJECT and sa_exists:
        gemini_client = genai.Client(
            vertexai=True,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
        )
        mode = f"Vertex AI (project={GOOGLE_CLOUD_PROJECT})"
    elif GOOGLE_API_KEY:
        gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
        mode = "API key (GOOGLE_API_KEY)"
    else:
        raise RuntimeError("No GCP credentials: set GOOGLE_CLOUD_PROJECT + service account, or GOOGLE_API_KEY")

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"Gemini client initialized — {mode}")
    logger.info(f"  Model:       {GEMINI_MODEL}")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def build_error_feedback_section(error_context: str) -> str:
    """Build the prompt section informing the model about a prior failed attempt."""
    if not error_context:
        return ""
    return (
        f"\nIMPORTANT — A PREVIOUS ATTEMPT AT THIS EXACT REQUEST FAILED with the "
        f"following error:\n  {error_context}\n"
        f"Do not repeat that failure. Return ONLY valid JSON matching the required "
        f"schema, with the complete result.\n"
    )


def build_translation_prompt(
    text_blocks: list[dict],
    target_language: str,
    target_script: str,
    page_context: str = "",
    error_context: str = "",
) -> str:
    """Build a batch translation prompt. No span markers — translate full block text.

    Fit-aware: each segment carries a character budget derived from its source
    length (the source already fits its box, so its length reflects how much room
    is available). The model is told to translate completely but concisely so the
    result fits the same space at a readable size — avoiding the heavy shrink that
    made translated text tiny.
    """
    segments = []
    for i, block in enumerate(text_blocks):
        src = block["text"]
        # Budget ≈ source length (+ small allowance). Indic text renders wider per
        # glyph, so staying near the source char count keeps the rendered width
        # close enough to fit without shrinking the font down to an unreadable size.
        budget = max(10, round(len(src) * 1.1))
        segments.append(f'[{i}] (fit in ~{budget} characters): "{src}"')
    segments_text = "\n".join(segments)

    glossary = get_glossary(target_language)
    glossary_section = ""
    if glossary:
        terms = "\n".join(f'  "{en}" → "{tr}"' for en, tr in glossary.items())
        glossary_section = (
            "\nTERMINOLOGY GLOSSARY — use these exact translations whenever the "
            f"English term appears (match case-insensitively):\n{terms}\n"
        )

    context_section = ""
    if page_context:
        context_section = (
            f"\nPAGE CONTEXT (for reference only — do NOT translate these lines, "
            f"use only to understand the meaning of each segment):\n{page_context}\n"
        )

    error_section = build_error_feedback_section(error_context)

    return f"""You are a professional document translator specializing in Indian languages. Translate the following text segments from English to {target_language} ({target_script} script).
{glossary_section}{context_section}{error_section}
CRITICAL RULES:
1. TRANSLITERATE brand names, product names, company names, and proper nouns in headings and body text — write them phonetically in {target_script}. Example: "HDFC Life" → "एचडीएफसी लाइफ", "Google" → "गूगल", "Sampoorna Jeevan" → "सम्पूर्ण जीवन". (EXCEPTION: in legal/disclaimer/footnote text — see rule 7 — the registered company name and product/plan names are kept in English, NOT transliterated.)
2. Preserve the NUMBERS and symbols themselves (digits, %, ₹, $, dates) exactly — but TRANSLATE the words around them, including unit words like "years"→वर्ष, "months", "days", and terms like "Annual", "Monthly", "Half-Yearly", "Minimum", "Maximum", "Option", "Benefit", "variant". Example: "28 years" → "28 वर्ष". The ONLY text that stays in English: brand names, registration codes (CIN, UIN, IRDAI, ARN), URLs, emails, phone numbers, and the digits themselves.
3. If a segment contains ONLY numbers, special characters, or whitespace, return it unchanged.
4. PRESERVE ALL LINE BREAKS exactly. If the original text has \\n, keep \\n in the translation at the same logical positions.
5. Preserve all spaces between numbers or words. Do NOT concatenate separate items.
6. FIT CONSTRAINT (important — each translation is placed back into the SAME box as its source): keep every translation within roughly its "fit in ~N characters" budget, and never much longer than its source. Use natural, standard, CONCISE {target_language} terminology and common short forms. You MUST preserve the complete meaning — do not drop information — but avoid verbose, redundant, or over-literal phrasing; when a faithful translation would run long, pick the shortest equivalent that keeps the meaning. Translations that overflow their budget get shrunk to an unreadable size, so concision matters.
7. LEGAL / DISCLAIMER / footnote text (e.g. the fine print at the end, terms & conditions, statutory notices): fully TRANSLATE the explanatory prose into {target_language}, but keep the following EXACTLY as written in the source English — do NOT translate, transliterate, or alter them: registration & license numbers (CIN, UIN, IRDAI/ARN registration numbers), the registered company/entity name and the product/plan names (e.g. "HDFC Life Insurance Company Limited", "HDFC Life Click 2 Protect Supreme Plus", rider/option names), URLs, email addresses, phone numbers, and postal/registered addresses. Translate everything else around them. The disclaimer must NOT be left in English.

Translate each of the {len(text_blocks)} numbered text segments. Return exactly {len(text_blocks)} translations in the "translations" array:

{segments_text}"""


def safe_parse_gemini_json(raw_text: Optional[str]) -> Optional[dict]:
    """Safely parse Gemini response, handling common quirks."""
    if not raw_text:
        return None
    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = text.find(start_char)
        end_idx = text.rfind(end_char)
        if start_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(text[start_idx: end_idx + 1])
            except json.JSONDecodeError:
                continue

    repaired = text
    quote_count = repaired.count('"') - repaired.count('\\"')
    if quote_count % 2 != 0:
        repaired += '"'
    open_brackets = repaired.count('[') - repaired.count(']')
    open_braces = repaired.count('{') - repaired.count('}')
    repaired = repaired.rstrip().rstrip(',')
    repaired += ']' * max(0, open_brackets)
    repaired += '}' * max(0, open_braces)
    try:
        result = json.loads(repaired)
        logger.warning("Parsed truncated Gemini JSON after repair")
        return result
    except json.JSONDecodeError:
        pass

    return None


async def translate_batch_with_gemini(
    blocks_with_text: list[dict],
    target_language: str,
    target_script: str,
    page_context: str = "",
) -> list[str]:
    """Translate blocks, auto-chunking into batches of GEMINI_BATCH_SIZE."""
    if not blocks_with_text:
        return []
    if len(blocks_with_text) <= GEMINI_BATCH_SIZE:
        return await _translate_batch_with_split(
            blocks_with_text, target_language, target_script, page_context=page_context
        )

    all_translations: list[str] = []
    total = len(blocks_with_text)
    # rolling_context starts as page_context; after each chunk we append the
    # translated output so the next chunk knows how earlier segments were phrased.
    rolling_context = page_context

    for start in range(0, total, GEMINI_BATCH_SIZE):
        chunk = blocks_with_text[start: start + GEMINI_BATCH_SIZE]
        chunk_num = start // GEMINI_BATCH_SIZE + 1
        total_chunks = (total + GEMINI_BATCH_SIZE - 1) // GEMINI_BATCH_SIZE
        logger.info(f"  [Gemini] Chunk {chunk_num}/{total_chunks}: {len(chunk)} block(s) to {target_language}")
        chunk_result = await _translate_batch_with_split(
            chunk, target_language, target_script, page_context=rolling_context
        )
        all_translations.extend(chunk_result)
        # Carry translated output as context for the next chunk [H5 fix].
        if chunk_result and start + GEMINI_BATCH_SIZE < total:
            prior = "\n".join(t for t in chunk_result if t)
            rolling_context = (
                page_context
                + "\n\nAlready translated earlier on this page:\n"
                + prior
            )
    return all_translations


async def _translate_batch_with_split(
    blocks_with_text: list[dict],
    target_language: str,
    target_script: str,
    page_context: str = "",
) -> list[str]:
    """Translate a batch; if it fails (e.g. token-heavy Indic output overflows
    max_output_tokens → empty response, even after retries), split it in half and
    retry each half. Token-heavy scripts (Tamil/Telugu/Malayalam) produce far more
    output tokens than English, so a batch that fits for one language may overflow
    for another — halving recovers without dropping content. A single block that
    still fails keeps its original text (the completeness loop will flag it)."""
    try:
        return await _translate_single_batch(
            blocks_with_text, target_language, target_script, page_context=page_context
        )
    except Exception as e:
        if len(blocks_with_text) <= 1:
            logger.warning(f"  [Gemini] Single block failed ({str(e)[:80]}) — keeping original")
            return [b["text"] for b in blocks_with_text]
        mid = len(blocks_with_text) // 2
        logger.warning(
            f"  [Gemini] Batch of {len(blocks_with_text)} failed ({str(e)[:50]}) — "
            f"splitting into {mid}+{len(blocks_with_text) - mid}"
        )
        left = await _translate_batch_with_split(
            blocks_with_text[:mid], target_language, target_script, page_context=page_context
        )
        right = await _translate_batch_with_split(
            blocks_with_text[mid:], target_language, target_script, page_context=page_context
        )
        return left + right


async def _translate_single_batch(
    blocks_with_text: list[dict],
    target_language: str,
    target_script: str,
    page_context: str = "",
) -> list[str]:
    """Translate a single batch of text blocks using Gemini.

    Retries up to 3 attempts; from attempt 2 onward the prompt includes the
    error from the previous attempt so the model can correct it (e.g. truncated
    or malformed JSON). Raises after the last attempt so the caller's
    batch-splitting recovery can kick in.
    """
    originals = [b["text"] for b in blocks_with_text]
    last_error = ""

    for attempt in range(3):
        logger.info(
            f"  [Gemini] Translating {len(blocks_with_text)} block(s) to {target_language}"
            + (f" (attempt {attempt + 1}/3, prev error fed back)" if attempt else "...")
        )
        prompt = build_translation_prompt(
            blocks_with_text, target_language, target_script,
            page_context=page_context, error_context=last_error,
        )
        try:
            async with GEMINI_SEMAPHORE:
                import time as _time
                _t0 = _time.monotonic()
                response = await asyncio.wait_for(
                    gemini_client.aio.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt,
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": {
                                "type": "OBJECT",
                                "properties": {
                                    "translations": {
                                        "type": "ARRAY",
                                        "items": {"type": "STRING"},
                                    }
                                },
                                "required": ["translations"],
                            },
                            "temperature": 0.1,
                            "max_output_tokens": 32768,
                            "thinking_config": {"thinking_level": "LOW"},
                        },
                    ),
                    timeout=120,
                )
                _elapsed = _time.monotonic() - _t0
                logger.info(f"  [Gemini] Response in {_elapsed:.1f}s for {len(blocks_with_text)} block(s)")

            if not response or not response.text:
                # Raise (not return originals) so the retry loop re-sends the chunk.
                # Silently returning originals would leave the block in English.
                raise RuntimeError("Gemini returned empty response")

            result = safe_parse_gemini_json(response.text)
            if result is None:
                raise ValueError(
                    f"Could not parse Gemini JSON response (first 200 chars): {response.text[:200]}"
                )

            translations = []
            if isinstance(result, dict):
                translations = result.get("translations", [])
            elif isinstance(result, list):
                translations = result

            validated = []
            for i, t in enumerate(translations):
                if isinstance(t, str) and t.strip():
                    validated.append(t)
                else:
                    validated.append(originals[i] if i < len(originals) else "")

            while len(validated) < len(blocks_with_text):
                idx = len(validated)
                validated.append(originals[idx] if idx < len(originals) else "")

            return validated

        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)[:300]}"
            logger.error(
                f"  [Gemini] Translation attempt {attempt + 1}/3 failed: {last_error}"
            )
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)

    raise RuntimeError(f"Gemini translation failed after 3 attempts — last error: {last_error}")
    