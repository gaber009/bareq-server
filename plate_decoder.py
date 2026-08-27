"""
plate_decoder.py — Arabic License Plate Decoder & Validator
============================================================
Converts spoken Arabic (Egyptian / Saudi dialect) to standardized plate format.

DESIGN RULES (per user requirements):
  1. This module does NORMALIZATION and VALIDATION only — never guesses missing data.
  2. Letter/number maps are derived from the existing server.py SPATIAL_LETTER_WORDS
     and NUM_WORDS tables, with additional dialect variants added.
  3. Confidence is composite, not a single ASR number.
  4. Partial results are for display only — never saved.
  5. Unicode normalization handles أ/ا/إ/آ and diacritics without losing meaningful info.

Plate format (existing project convention):
  3 Arabic letters (space-separated) + 1–4 digits
  Example: "أ ب ج 1234"
"""

import re
import unicodedata
from typing import Optional

# ═══════════════════════════════════════════════════════════
# Unicode / Arabic Text Normalization
# ═══════════════════════════════════════════════════════════

def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text for plate processing.
    - Strip diacritics (tashkeel) — they are noise for plate recognition.
    - Normalize Alef forms: أ إ آ ٱ → أ  (keep hamza-above form, matching project convention).
    - Normalize ى → ى (keep as-is, project uses ى for Ya in plates).
    - Normalize ة → ه (ta-marbuta to ha, matches frontend normPlate).
    - Collapse multiple whitespace.
    - Strip zero-width chars.
    Does NOT merge أ/ا — the project's SPATIAL_LETTER_WORDS maps to 'أ' specifically.
    """
    if not text:
        return ""
    t = text

    # Strip zero-width characters
    t = re.sub(r'[\u200b\u200c\u200d\ufeff\u200e\u200f]', '', t)

    # Strip Arabic diacritics / tashkeel (U+064B–U+065F, U+0670)
    t = re.sub(r'[\u064b-\u065f\u0670]', '', t)

    # Normalize Alef variants → أ  (project convention: plates use أ)
    t = t.replace('إ', 'أ').replace('آ', 'أ').replace('ٱ', 'أ')
    # Keep bare Alef (ا) as-is — it's distinct from أ in plate matching.
    # The letter map below handles both 'ألف' → 'أ' and bare ا if ASR outputs it.

    # Normalize ta-marbuta → ه (matches frontend normPlate)
    t = t.replace('ة', 'ه')

    # Collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()

    return t


# ═══════════════════════════════════════════════════════════
# Letter & Number Maps
# Derived from server.py SPATIAL_LETTER_WORDS (lines 247-276)
# and NUM_WORDS (lines 278-288), with additional dialect variants.
# ═══════════════════════════════════════════════════════════

# Original project letters (from server.py line 247-276):
#   ('ألف','أ'), ('الف','أ'), ('إلف','أ'), ('إليف','أ'),
#   ('باء','ب'), ('با','ب'),
#   ('تاء','ت'), ('تا','ت'),
#   ('ثاء','ث'), ('ثا','ث'),
#   ('جيم','ج'), ('جم','ج'),
#   ('حاء','ح'), ('حا','ح'),
#   ('خاء','خ'), ('خا','خ'),
#   ('دال','د'), ('دا','د'),
#   ('ذال','ذ'), ('ذا','ذ'),
#   ('راء','ر'), ('را','ر'), ('ري','ر'),
#   ('زين','ز'), ('زاي','ز'), ('زا','ز'),
#   ('سين','س'), ('سا','س'),
#   ('شين','ش'), ('شا','ش'),
#   ('صاد','ص'), ('صا','ص'),
#   ('ضاد','ض'), ('ضا','ض'),
#   ('طاء','ط'), ('طا','ط'),
#   ('ظاء','ظ'), ('ظا','ظ'),
#   ('عين','ع'), ('عا','ع'),
#   ('غين','غ'), ('غا','غ'),
#   ('فاء','ف'), ('فا','ف'),
#   ('قاف','ق'), ('قا','ق'),
#   ('كاف','ك'), ('كا','ك'),
#   ('لام','ل'), ('لا','ل'),
#   ('ميم','م'), ('ما','م'),
#   ('نون','ن'), ('نا','ن'),
#   ('هاء','هـ'), ('ها','هـ'),
#   ('واو','و'),
#   ('ياء','ى'), ('يا','ى')

LETTER_MAP = [
    # ── Alef ──
    ('ألف', 'أ'), ('الف', 'أ'), ('إلف', 'أ'), ('إليف', 'أ'), ('أليف', 'أ'), ('آلف', 'أ'), ('اليف', 'أ'), ('آليف', 'أ'),
    # ── Ba ──
    ('باء', 'ب'), ('با', 'ب'), ('بيه', 'ب'), ('به', 'ب'), ('بي', 'ب'), ('بى', 'ب'),
    # ── Ta ──
    ('تاء', 'ت'), ('تا', 'ت'), ('تيه', 'ت'), ('ته', 'ت'), ('تي', 'ت'), ('تى', 'ت'),
    # ── Tha ──
    ('ثاء', 'ث'), ('ثا', 'ث'), ('ثيه', 'ث'), ('ثه', 'ث'), ('ثي', 'ث'), ('ثى', 'ث'),
    # ── Jeem ──
    ('جيم', 'ج'), ('جم', 'ج'), ('جيه', 'ج'), ('جه', 'ج'), ('جي', 'ج'), ('جى', 'ج'),
    # ── Ha (ح) ──
    ('حاء', 'ح'), ('حا', 'ح'), ('حيه', 'ح'), ('حه', 'ح'), ('حي', 'ح'), ('حى', 'ح'),
    # ── Kha ──
    ('خاء', 'خ'), ('خا', 'خ'), ('خيه', 'خ'), ('خه', 'خ'), ('خي', 'خ'), ('خى', 'خ'),
    # ── Dal ──
    ('دال', 'د'), ('دا', 'د'), ('ديه', 'د'), ('ده', 'د'), ('دي', 'د'), ('دى', 'د'),
    # ── Thal ──
    ('ذال', 'ذ'), ('ذا', 'ذ'), ('ذيه', 'ذ'), ('ذه', 'ذ'), ('ذي', 'ذ'), ('ذى', 'ذ'),
    # ── Ra ──
    ('راء', 'ر'), ('را', 'ر'), ('ريه', 'ر'), ('ره', 'ر'), ('ري', 'ر'), ('رى', 'ر'),
    # ── Zay ──
    ('زين', 'ز'), ('زاي', 'ز'), ('زا', 'ز'), ('زيه', 'ز'), ('زه', 'ز'), ('زي', 'ز'), ('زى', 'ز'),
    # ── Seen ──
    ('سين', 'س'), ('سا', 'س'), ('سيه', 'س'), ('سه', 'س'), ('سي', 'س'), ('سى', 'س'),
    # ── Sheen ──
    ('شين', 'ش'), ('شا', 'ش'), ('شيه', 'ش'), ('شه', 'ش'), ('شي', 'ش'), ('شى', 'ش'),
    # ── Sad ──
    ('صاد', 'ص'), ('صا', 'ص'), ('صيه', 'ص'), ('صه', 'ص'), ('صي', 'ص'), ('صى', 'ص'),
    # ── Dad ──
    ('ضاد', 'ض'), ('ضا', 'ض'), ('ضيه', 'ض'), ('ضه', 'ض'), ('ضي', 'ض'), ('ضى', 'ض'),
    # ── Taa ──
    ('طاء', 'ط'), ('طا', 'ط'), ('طيه', 'ط'), ('طه', 'ط'), ('طي', 'ط'), ('طى', 'ط'),
    # ── Dhaa ──
    ('ظاء', 'ظ'), ('ظا', 'ظ'), ('ظيه', 'ظ'), ('ظه', 'ظ'), ('ظي', 'ظ'), ('ظى', 'ظ'),
    # ── Ain ──
    ('عين', 'ع'), ('عا', 'ع'), ('عيه', 'ع'), ('عه', 'ع'), ('عي', 'ع'), ('عى', 'ع'),
    # ── Ghain ──
    ('غين', 'غ'), ('غا', 'غ'), ('غيه', 'غ'), ('غه', 'غ'), ('غي', 'غ'), ('غى', 'غ'),
    # ── Fa ──
    ('فاء', 'ف'), ('فا', 'ف'), ('فيه', 'ف'), ('فه', 'ف'), ('في', 'ف'), ('فى', 'ف'),
    # ── Qaf — قاف (G-sound in Gulf/Saudi dialects maps to ق) ──
    ('قاف', 'ق'), ('قيف', 'ق'), ('قا', 'ق'), ('قيه', 'ق'), ('قه', 'ق'), ('قي', 'ق'), ('قى', 'ق'),
    # ── Kaf — كاف (K-sound, never confused with ق) ──
    ('كاف', 'ك'), ('كيف', 'ك'), ('كا', 'ك'), ('كيه', 'ك'), ('كه', 'ك'), ('كي', 'ك'), ('كى', 'ك'),
    # ── Lam ──
    ('لام', 'ل'), ('لا', 'ل'), ('ليه', 'ل'), ('له', 'ل'), ('لي', 'ل'), ('لى', 'ل'),
    # ── Meem ──
    ('ميم', 'م'), ('ما', 'م'), ('ميه', 'م'), ('مه', 'م'), ('مي', 'م'), ('مى', 'م'),
    # ── Noon ──
    ('نون', 'ن'), ('نا', 'ن'), ('نيه', 'ن'), ('نه', 'ن'), ('ني', 'ن'), ('نى', 'ن'),
    # ── Ha (هـ) ──
    ('هاء', 'هـ'), ('ها', 'هـ'), ('هيه', 'هـ'), ('هه', 'هـ'), ('هي', 'هـ'), ('هى', 'هـ'),
    # ── Waw ──
    ('واو', 'و'), ('وا', 'و'), ('ويه', 'و'), ('وي', 'و'),
    # ── Ya ──
    ('ياء', 'ى'), ('يا', 'ى'), ('ييه', 'ى'), ('يه', 'ى'), ('يي', 'ى'),
    # ── Common Whisper Merged Words (ASR merging 3 letter names into dictionary words) ──
    ('أبداً', 'أ ب د'), ('أبدا', 'أ ب د'), ('ابدا', 'أ ب د'),
    ('أبوة', 'أ ب و'), ('ابوة', 'أ ب و'), ('أبوه', 'أ ب هـ'), ('ابوه', 'أ ب هـ'),
    ('أبوك', 'أ ب ك'), ('ابوك', 'أ ب ك'), ('أبوكي', 'أ ب ك'), ('ابوكي', 'أ ب ك'),
    ('أبوم', 'أ ب م'), ('ابوم', 'أ ب م'),
    ('أمي', 'أ م ى'), ('امي', 'أ م ى'),
]

# Original project numbers (from server.py lines 278-288) + dialect variants:
NUM_MAP = [
    ('واحد', '1'), ('واحدة', '1'), ('واحده', '1'),
    ('اثنين', '2'), ('إثنين', '2'), ('أثنين', '2'), ('اتنين', '2'), ('إتنين', '2'), ('أتنين', '2'),
    ('تنين', '2'), ('اثنان', '2'), ('إثنان', '2'), ('أثنان', '2'),
    ('ثلاثه', '3'), ('تلاتة', '3'), ('ثلاثة', '3'), ('تلاته', '3'), ('ثلاث', '3'), ('تلات', '3'),
    ('أربعه', '4'), ('اربعة', '4'), ('أربعة', '4'), ('اربعه', '4'), ('أربع', '4'), ('اربع', '4'),
    ('خمسه', '5'), ('خمسة', '5'), ('خمس', '5'),
    ('سته', '6'), ('ستة', '6'), ('ستّة', '6'), ('ست', '6'),
    ('سبعه', '7'), ('سبعة', '7'), ('سبع', '7'),
    ('ثمانيه', '8'), ('ثمانية', '8'), ('تمانية', '8'), ('تمانيه', '8'),
    ('تمنية', '8'), ('تمنيه', '8'), ('ثماني', '8'), ('تماني', '8'), ('ثمان', '8'), ('تمان', '8'),
    ('تسعه', '9'), ('تسعة', '9'), ('تسع', '9'),
    ('صفر', '0'), ('زيرو', '0'),
    # Eastern Arabic numerals (from server.py line 287)
    ('٠', '0'), ('١', '1'), ('٢', '2'), ('٣', '3'), ('٤', '4'),
    ('٥', '5'), ('٦', '6'), ('٧', '7'), ('٨', '8'), ('٩', '9'),
]

# Valid plate letters — the regex range [أ-يى] from the existing project
# covers all standard Arabic letters. We use a set for O(1) lookup.
VALID_PLATE_LETTERS = set('أابتثجحخدذرزسشصضطظعغفقكلمنهوىي')
VALID_PLATE_LETTERS_EXT = VALID_PLATE_LETTERS | {'هـ'}

# Sort maps longest-first to prevent partial matches
LETTER_MAP_SORTED = sorted(LETTER_MAP, key=lambda x: len(x[0]), reverse=True)
NUM_MAP_SORTED = sorted(NUM_MAP, key=lambda x: len(x[0]), reverse=True)

# Noise / filler words ASR might output
_NOISE_WORDS = [
    'يعني', 'اه', 'آه', 'ممم', 'اللوحه', 'اللوحة', 'لوحه', 'لوحة',
    'رقم', 'نمره', 'نمرة', 'السياره', 'السيارة', 'العربيه', 'العربية',
    'تمام', 'خلاص', 'اوك', 'اوكي', 'كده', 'بقى', 'شكرا', 'جزيلا',
    'صباح', 'الخير', 'مرحبا', 'يا فندم', 'الجو', 'النهارده', 'النهاردة', 'جميل'
]
_NOISE_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(w) for w in sorted(_NOISE_WORDS, key=len, reverse=True)) + r')\b',
    re.UNICODE
)


class PlateDecoder:
    """
    Decodes spoken Arabic text into standardized license plate format.

    Plate format (from existing project regex patterns):
      Pattern 1: [أ-يى] SP [أ-يى] SP [أ-يى] SP \\d{1,4}   e.g. "أ ب د 1234"
      Pattern 2: [أ-يى]{3} \\d{1,4}                         e.g. "أبد1234" → "أ ب د 1234"

    This class ONLY normalizes and validates. It NEVER:
      - Guesses missing letters or digits.
      - Completes a partial plate.
      - Invents data not present in the ASR output.
    """

    def normalize_speech(self, text: str) -> str:
        """Convert spoken Arabic text to plate components via phonetic rules."""
        if not text:
            return ""

        t = normalize_arabic_text(text)

        # Strip noise/filler words
        t = _NOISE_RE.sub(' ', t)

        # Replace number words → digits (longest first)
        for word, digit in NUM_MAP_SORTED:
            t = re.sub(r'\b' + re.escape(word) + r'\b', ' ' + digit + ' ', t)
        # Plain replace for remaining (handles concatenated cases)
        for word, digit in NUM_MAP_SORTED:
            if word in t:
                t = t.replace(word, ' ' + digit + ' ')

        # Replace letter names → single characters (longest first)
        for word, letter in LETTER_MAP_SORTED:
            t = re.sub(r'\b' + re.escape(word) + r'\b', ' ' + letter + ' ', t)

        # Convert any remaining Eastern Arabic digits
        for ea, wd in [('٠','0'),('١','1'),('٢','2'),('٣','3'),('٤','4'),
                       ('٥','5'),('٦','6'),('٧','7'),('٨','8'),('٩','9')]:
            t = t.replace(ea, wd)

        # Collapse whitespace
        t = re.sub(r'\s+', ' ', t).strip()

        # Merge isolated digits: "1 2 3 4" → "1234"
        prev = ""
        while prev != t:
            prev = t
            t = re.sub(r'(\d)\s+(\d)', r'\1\2', t)

        return t.strip()

    def extract_components(self, normalized: str) -> dict:
        """Extract letters and digits from normalized text."""
        if not normalized:
            return {"letters": [], "digits": "", "total_letters_found": 0, "total_digits_found": 0, "raw": normalized}

        tokens = normalized.split()
        all_letters = []
        all_digits_parts = []

        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if re.match(r'^\d+$', token):
                all_digits_parts.append(token)
            elif token == 'هـ':
                all_letters.append('هـ')
            elif all(c in VALID_PLATE_LETTERS for c in token):
                # Handles single letters ('أ', 'ب'), 2-letter tokens ('اب'), 3-letter tokens ('ابم'), etc.
                for c in token:
                    c_norm = 'أ' if c in ('ا', 'إ', 'آ', 'ٱ') else ('هـ' if c == 'ه' else c)
                    all_letters.append(c_norm)

        digits_str = ''.join(all_digits_parts)

        return {
            "letters": all_letters[:3] if len(all_letters) == 3 else all_letters,
            "digits": digits_str,
            "total_letters_found": len(all_letters),
            "total_digits_found": len(digits_str),
            "raw": normalized,
        }

    def format_plate(self, letters: list, digits: str) -> str:
        """Format as 'أ ب ج 1234' — matches project convention."""
        if not letters:
            return ""
        plate = " ".join(letters)
        if digits:
            plate += " " + digits
        return plate

    def validate_plate(self, letters: list, digits: str, total_letters: int = None, total_digits: int = None) -> bool:
        """
        Validate against existing project rules:
          - Exactly 3 Arabic letters (no more, no less)
          - 1–4 digits (no more, no less)
        """
        tot_l = total_letters if total_letters is not None else len(letters)
        tot_d = total_digits if total_digits is not None else len(digits)

        if tot_l != 3 or len(letters) != 3:
            return False
        if not digits or not (1 <= tot_d <= 4) or not (1 <= len(digits) <= 4):
            return False
        for ch in letters:
            if ch != 'هـ' and ch not in VALID_PLATE_LETTERS:
                return False
        if not digits.isdigit():
            return False
        return True

    # ─────────────────────────────────────────────────────
    # Partial Decode — for display only, NEVER saved
    # ─────────────────────────────────────────────────────

    def decode_partial(self, raw_text: str) -> dict:
        """
        Decode partial speech for display during speaking.
        Returns whatever letters/digits recognized so far.
        This result is NEVER saved — display only.
        """
        normalized = self.normalize_speech(raw_text)
        comp = self.extract_components(normalized)
        partial = self.format_plate(comp["letters"], comp["digits"])

        return {
            "partial_plate": partial if comp["letters"] else "",
            "letters_count": len(comp["letters"]),
            "digits_count": len(comp["digits"]),
            "is_complete": len(comp["letters"]) == 3 and len(comp["digits"]) >= 1,
            "raw_text": raw_text,
            "normalized": normalized,
        }

    # ─────────────────────────────────────────────────────
    # Final Decode — validated, for saving
    # ─────────────────────────────────────────────────────

    def decode_final(self, raw_text: str, asr_segment_confidences: list = None,
                     partial_history: list = None) -> dict:
        """
        Decode final speech after end_of_turn.

        Composite confidence is built from multiple signals (per user req #3):
          1. ASR segment avg_logprob (if available from faster-whisper)
          2. Plate grammar match (3 letters + 1-4 digits)
          3. Completeness (all components present)
          4. Stability (does final match last stable partial?)

        Returns dict with plate, valid, confidence, and all signals.

        NEVER guesses missing data (req #4). If plate is incomplete,
        valid=False and it should NOT be saved.
        """
        normalized = self.normalize_speech(raw_text)
        comp = self.extract_components(normalized)

        letters = comp["letters"]
        digits = comp["digits"]
        tot_l = comp.get("total_letters_found", len(letters))
        tot_d = comp.get("total_digits_found", len(digits))
        is_valid = self.validate_plate(letters, digits, total_letters=tot_l, total_digits=tot_d)
        plate = self.format_plate(letters, digits) if is_valid or (len(letters) == 3 and 1 <= len(digits) <= 4) else ""

        # ── Composite confidence signals ──
        signals = {}

        # Signal 1: ASR segment confidence
        if asr_segment_confidences and len(asr_segment_confidences) > 0:
            # avg_logprob from faster-whisper: negative, closer to 0 = better
            # Typical range: -0.1 (excellent) to -1.5 (poor)
            avg_logprob = sum(asr_segment_confidences) / len(asr_segment_confidences)
            # Convert to 0-1 scale (NOT calibrated probability — see note below)
            asr_conf = max(0.0, min(1.0, 1.0 + avg_logprob))
            signals["asr_logprob"] = round(avg_logprob, 4)
            signals["asr_confidence"] = round(asr_conf, 3)
        else:
            signals["asr_logprob"] = None
            signals["asr_confidence"] = None

        # Signal 2: Grammar match
        signals["grammar_match"] = is_valid

        # Signal 3: Completeness
        signals["letters_found"] = len(letters)
        signals["digits_found"] = len(digits)
        signals["complete"] = len(letters) == 3 and len(digits) >= 1

        # Signal 4: Stability — does final match the last partial?
        if partial_history and len(partial_history) > 0:
            last_partial = partial_history[-1]
            signals["matches_last_partial"] = (plate == last_partial)
            # Check if at least the letters part was stable across recent partials
            if len(partial_history) >= 2:
                recent = partial_history[-2:]
                letters_part = " ".join(letters) if letters else ""
                stable = all(p.startswith(letters_part) for p in recent if p)
                signals["letters_stable"] = stable
            else:
                signals["letters_stable"] = None
        else:
            signals["matches_last_partial"] = None
            signals["letters_stable"] = None

        # ── Composite confidence score ──
        # NOTE: This is NOT a calibrated probability. It is a heuristic
        # composite score from multiple signals. Do not treat 0.8 as "80%
        # probability of being correct." It is a ranking/threshold signal.
        composite = 0.0
        weights_sum = 0.0

        # ASR signal (weight 0.4 if available)
        if signals["asr_confidence"] is not None:
            composite += signals["asr_confidence"] * 0.4
            weights_sum += 0.4

        # Grammar signal (weight 0.3)
        composite += (1.0 if is_valid else 0.0) * 0.3
        weights_sum += 0.3

        # Completeness signal (weight 0.2)
        comp_score = (min(len(letters), 3) / 3.0) * 0.5 + (min(len(digits), 1) / 1.0) * 0.5
        composite += comp_score * 0.2
        weights_sum += 0.2

        # Stability signal (weight 0.1 if available)
        if signals["matches_last_partial"] is not None:
            composite += (1.0 if signals["matches_last_partial"] else 0.3) * 0.1
            weights_sum += 0.1

        # Normalize
        if weights_sum > 0:
            composite = composite / weights_sum
        else:
            composite = 0.0

        return {
            "plate": plate,
            "letters": letters,
            "digits": digits,
            "valid": is_valid,
            "confidence": round(composite, 3),
            "confidence_note": "Heuristic composite score, NOT calibrated probability",
            "signals": signals,
            "raw_text": raw_text,
            "normalized": normalized,
        }


# ═══════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════

_decoder_instance = PlateDecoder()

def get_decoder() -> PlateDecoder:
    return _decoder_instance
