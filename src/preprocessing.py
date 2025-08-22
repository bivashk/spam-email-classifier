import re

def basic_clean(text: str) -> str:
    """Very light cleaning: keep letters/numbers/space, collapse whitespace.
    Note: TfidfVectorizer already lowercases and removes English stopwords.
    """
    text = re.sub(r"[^A-Za-z0-9\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text
