import re
import pandas as pd

# Emoji siyahısı — sadə Unicode range yoxlama
EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251]+",
    flags=re.UNICODE,
)


def clean_text(text: str) -> str:
    """Mətn temizleme: emoji, xüsusi simvol, boşluq normalizasiyası."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = EMOJI_PATTERN.sub(" ", text)       # emojiləri sil
    text = re.sub(r"[^\w\s]", " ", text)       # durğu işarələrini sil
    text = re.sub(r"\s+", " ", text).strip()   # çoxlu boşluqları normallaşdır
    return text.lower()


def build_features(row) -> str:
    """
    feedback və tag sütunlarını birləşdirir.
    tag NaN olduğu halda yalnız feedback istifadə olunur.
    """
    feedback = clean_text(str(row.get("feedback", "")))
    tag = str(row.get("tag", ""))
    if tag and tag.lower() not in ("nan", "none", ""):
        tag_clean = clean_text(tag)
        return f"{feedback} [TAG] {tag_clean}"
    return feedback
