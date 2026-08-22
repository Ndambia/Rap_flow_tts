"""Text cleaning + G2P -> phoneme id sequence."""
from phonemizer import phonemize
from phonemizer.separator import Separator

_PAD, _EOS = "_", "~"
_SPECIAL = "-"
_PUNCT = "!'(),.:;? "
_LETTERS_IPA = (
    "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻʘɓǀɗǃʄǂɠǁʛpbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟˈˌːˑʲʷⁿˠ̩"
)

SYMBOLS = [_PAD] + list(_SPECIAL) + list(_PUNCT) + list(_LETTERS_IPA) + [_EOS]
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}


def text_to_phoneme_ids(text: str, language: str = "en-us") -> list[int]:
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
