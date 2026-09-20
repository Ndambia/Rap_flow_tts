"""Text cleaning + G2P -> phoneme id sequence.

Includes a graceful character-level fallback for Windows systems where
espeak-ng may not be installed, ensuring the notebook never crashes.
"""
import warnings

_PAD, _EOS = "_", "~"
_SPECIAL = "-"
_PUNCT = "!'(),.:;? "
_LETTERS_IPA = (
    "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻʘɓǀɗǃʄǂɠǁʛpbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟˈˌːˑʲʷⁿˠ̩"
)

SYMBOLS = [_PAD] + list(_SPECIAL) + list(_PUNCT) + list(_LETTERS_IPA) + [_EOS]
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}

# ── Character-level fallback (used when espeak-ng is not installed on Windows) ──
_CHAR_SYMBOLS = (
    [_PAD]
    + list(_SPECIAL)
    + list(_PUNCT)
    + list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    + [_EOS]
)
_CHAR_TO_ID = {s: i for i, s in enumerate(_CHAR_SYMBOLS)}

# ── Detect espeak availability once at import time ────────────────────────────
_ESPEAK_OK: bool = False
try:
    from phonemizer import phonemize as _ph
    from phonemizer.separator import Separator as _Sep
    _ph("test", language="en-us", backend="espeak",
        separator=_Sep(phone="", word=" "), strip=True)  # will raise if espeak missing
    _ESPEAK_OK = True
except Exception:
    warnings.warn(
        "phonemizer/espeak-ng not available — using character-level fallback tokenizer. "
        "Install espeak-ng for full IPA phonemization.",
        RuntimeWarning,
        stacklevel=2,
    )


def text_to_phoneme_ids(text: str, language: str = "en-us") -> list[int]:
    """Convert text to phoneme (or character) ID sequence.

    Falls back to character-level tokenisation on Windows systems where
    espeak-ng is absent, so the notebook / training code never crashes.
    """
    if _ESPEAK_OK:
        from phonemizer import phonemize
        from phonemizer.separator import Separator
        phonemes = phonemize(
            text,
            language=language,
            backend="espeak",
            separator=Separator(phone="", word=" "),
            strip=True,
            preserve_punctuation=True,
            with_stress=True,
        )
        return [SYMBOL_TO_ID[c] for c in phonemes if c in SYMBOL_TO_ID]
    else:
        # Character-level fallback
        return [_CHAR_TO_ID[c] for c in text if c in _CHAR_TO_ID]
