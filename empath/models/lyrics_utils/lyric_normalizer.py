import re
from opencc import OpenCC


t2s_converter = OpenCC("t2s")
s2t_converter = OpenCC("s2t")


EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # Emoticons
    "]+",
    flags=re.UNICODE,
)

# Create a translation table to replace and remove characters
TRANSLATION_TABLE = str.maketrans(
    {
        "-": " ",  # Replace '-' with a space
        ",": None,
        ".": None,
        "，": None,
        "。": None,
        "!": None,
        "！": None,
        "?": None,
        "？": None,
        "…": None,
        ";": None,
        "；": None,
        ":": None,
        "：": None,
        "\u3000": " ",  # Replace full-width space with a space
    }
)

# Replace content inside brackets, including square brackets and parentheses
BACKSLASH_PATTERN = re.compile(r"\(.*?\)|\[.*?\]")

SPACE_PATTERN = re.compile("(?<!^)\s+(?!$)")


def normalize_text(text, language, strip=True):
    """
    Standardize text, remove punctuation marks, convert to lowercase (if applicable)
    """
    # Step 1: Replace '-' with ' ' and remove punctuation
    text = text.translate(TRANSLATION_TABLE)

    # Step 2: Remove emojis
    text = EMOJI_PATTERN.sub("", text)

    # Step 3: Replace consecutive whitespaces with a single space, except at the beginning and end
    text = SPACE_PATTERN.sub(" ", text)

    # Step 4: Remove leading and trailing whitespaces (if needed)
    if strip:
        text = text.strip()

    # Step 5: Convert to lowercase
    text = text.lower()

    # Step 6: Multilingual conversion
    if language == "zh":
        text = t2s_converter.convert(text)
    if language == "yue":
        text = s2t_converter.convert(text)
    # Add other languages as needed
    return text
