import logging
import re
import sys
import unicodedata
from typing import List, Tuple

logger = logging.getLogger(__name__)


# Copy from: datahub/normalizing/normalize_text.py
# Normalizing reference: https://www.fileformat.info/info/unicode/category/index.htm

def get_unicode_bracket_pairs() -> Tuple[List[str], List[str]]:
    brackets_list = ""
    other_brackets_list = ""
    # Ref: https://stackoverflow.com/questions/13535172/list-of-all-unicodes-open-close-brackets
    for c in map(chr, range(sys.maxunicode + 1)):
        if unicodedata.category(c) in ["Ps", "Pe", "Pi", "Pf"]:
            if unicodedata.mirrored(c):
                brackets_list += c
            else:
                other_brackets_list += c

    if len(brackets_list) % 2 != 0 or len(other_brackets_list) % 2 != 0:
        logger.debug("Non-symmetric bracket list:")
        logger.debug(" - Bracket : " + brackets_list)
        logger.debug(" - Other   : " + other_brackets_list)

    brackets_list += other_brackets_list
    lefts = []
    rights = []
    for ii in range(0, len(brackets_list), 2):
        c1 = brackets_list[ii]
        c2 = brackets_list[ii + 1]
        lefts.append(c1)
        rights.append(c2)
    return lefts, rights


def get_unicode_chars_by_categories(categories: List[str]) -> str:
    res = ""
    other = ""
    for cc in map(chr, range(sys.maxunicode + 1)):
        if unicodedata.category(cc) in categories:
            if unicodedata.mirrored(cc):
                res += cc
            else:
                other += cc
    return res + other


def get_unicode_chars_by_similar_names(name_parts: List[str], categories: List[str] = None) -> str:
    def is_ok(name: str) -> bool:
        if name is None:
            return False
        if name in name_parts:
            return True
        for name_part in name_parts:
            if name_part in name:
                return True
        return False

    res = ""
    other = ""
    if categories:
        cur_str = get_unicode_chars_by_categories(categories)
    else:
        cur_str = map(chr, range(sys.maxunicode + 1))
    for cc in cur_str:
        try:
            name = unicodedata.name(cc)
        except KeyError:
            name = None
        if is_ok(name):
            if unicodedata.mirrored(cc):
                res += cc
            else:
                other += cc
    return res + other


def generate_pattern_from_list(strs: List[str]) -> re.compile:
    return r"[" + re.escape(r"".join(strs)) + r"]"


def generate_normalizers() -> List[re.compile]:
    left_brackets, right_brackets = get_unicode_bracket_pairs()
    dashes = get_unicode_chars_by_categories(["Pd"])
    spaces = get_unicode_chars_by_categories(["Zs", "Zl", "Zp"])
    dots = get_unicode_chars_by_similar_names(["DOT ", " DOT", " STOP", "STOP "], ["Po"])
    # modifiers = get_unicode_chars_by_categories(['Sk'])
    return [
        ("0", re.compile(r"[0-9]")),
        ('"', re.compile(r"\'")),
        (",", re.compile(r"\;")),
        ("-", re.compile(r"_")),
        (" ", re.compile(r"[\t\n\r]")),
        ("-", re.compile(generate_pattern_from_list(dashes))),
        (" ", re.compile(generate_pattern_from_list(spaces))),
        (".", re.compile(generate_pattern_from_list(dots))),
        ("(", re.compile(generate_pattern_from_list(left_brackets))),
        (")", re.compile(generate_pattern_from_list(right_brackets)))]


NORMALIZERS = generate_normalizers()


def normalize_text(text: str, corpus: List[str] = None) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    for target_unichr, normalizer in NORMALIZERS:
        text = normalizer.sub(target_unichr, text)
    if corpus is not None:
        text = "".join([tt if tt in corpus else "�" for tt in text])
    return text
