import re

# whitelist of allowed characters
WHITELIST = set("abcdefghijklmnopqrstuvwxyzÄÖÜäöüß"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥ "
)

def clean_sentence(text: str) -> str:
    """
    Cleans a sentence for tokenizer/model training.

    The following cleaning steps are performed:
    1. Remove invalid (non-decodable) UTF-8 characters.
    2. Remove URLs (http, https, www).
    3. Remove HTML tags.
    4. Remove any character not present in the predefined whitelist.
    5. Convert text to lowercase.
    6. Normalize whitespace (collapse multiple spaces and trim).

    Args:
        text (str): Input text to be cleaned.

    Returns:
        str: Cleaned sentence.
    """
    # Remove non-UTF8 characters by converting string to UTF-8 bytes (while dropping invalid characters) and re-converting back to a normal string again 
    text = text.encode("utf-8", "ignore").decode("utf-8")

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove characters not in whitelist
    text = re.sub(r"[^\S\r\n]+", " ", text)

    # Lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_valid_length(sentence: str, min_len=5, max_len=64):
    """Check whether a sentence satisfies length constraints based on word count.
    Args:
        sentence (str): Input sentence.
        min_len (int): Minimum allowed number of words. Default is 5.
        max_len (int): Maximum allowed number of words. Default is 64.

    Returns:
        bool: True if the sentence length is within the specified bounds,
              False otherwise.
    """
    length = len(sentence.split())
    return min_len <= length <= max_len


def length_ratio_check(src: str, tgt: str, max_ratio=2.5):
    """
    Check whether the length ratio between a source and target sentence is acceptable.
    Args:
        src (str): Source sentence.
        tgt (str): Target sentence.
        max_ratio (float): Maximum allowed length ratio between source and target. Default is 2.5.

    Returns:
        bool: True if the length ratio is within the allowed threshold,
              False otherwise.
    """
    src_len = len(src.split())
    tgt_len = len(tgt.split())
    if src_len == 0 or tgt_len == 0:
        return False
    ratio = max(src_len / tgt_len, tgt_len / src_len) # guarantees ratio >= 1
    return ratio <= max_ratio


def clean_dataset(dataset, min_len=5, max_len=64, max_ratio=2.5):
    """
    Clean and filter a parallel dataset of sentence pairs.

    Each sentence pair (source, target) is processed as follows:
    1. Clean both sentences (UTF-8 sanitization, URL removal, HTML removal,
       whitelist filtering, lowercasing, whitespace normalization).
    2. Remove pairs where either sentence is too short or too long, based on
       word count.
    3. Remove pairs where the length ratio between source and target exceeds
       the specified threshold.

    Args:
        dataset (datasets.Dataset):
            Hugging Face dataset containing translation examples in the form:
            {"translation": {"de": str, "en": str}}.
        min_len (int, optional):
            Minimum allowed sentence length in words. Default is 5.
        max_len (int, optional):
            Maximum allowed sentence length in words. Default is 64.
        max_ratio (float, optional):
            Maximum allowed length ratio between source and target sentences.
            Default is 2.5.

    Returns:
        list[dict]: Each element is {"src": ..., "tgt": ...}
    """
    cleaned_data = []
    for example in dataset:
        translation = example.get("translation")
        if translation is None:
            continue

        source = translation.get("de")
        target = translation.get("en")

        if not source or not target:
            continue

        source_cleaned = clean_sentence(source)
        target_cleaned = clean_sentence(target)

        # Length filtering
        if not is_valid_length(source_cleaned, min_len, max_len):
            continue
        if not is_valid_length(target_cleaned, min_len, max_len):
            continue

        # Ratio filtering
        if not length_ratio_check(source_cleaned, target_cleaned, max_ratio):
            continue

        cleaned_data.append(
            {
                "src": source_cleaned,
                "tgt": target_cleaned,
            }
        )

    return cleaned_data