"""
Normalize Durham 10-codes and signals in Whisper output.

Whisper writes the same code many ways ("10 50", "10.50", "ten fifty",
"Ten-Fifty PI"); the summarizer and anyone reading the feed want "10-50PI".
Only codes on the RadioReference Durham County (NC) list are rewritten, so
ordinary numbers like "10 40th Street" don't turn into codes by accident.
"""
import re

# Durham 10-codes, from the RadioReference Durham County (NC) wiki page.
TEN_CODES = set(range(1, 37)) | set(range(38, 82)) | {89, 90, 91, 92, 93, 94, 97, 98}
SIGNALS = {20, 26, 29, 30, 70}
# Letter suffixes some codes take: 10-33a, 10-43a, 10-50PD/PI/F.
SUFFIXES = {33: ('a',), 43: ('a',), 50: ('pd', 'pi', 'f')}

_ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
         'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
         'sixteen', 'seventeen', 'eighteen', 'nineteen']
_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

WORD_NUMBERS = {w: n for n, w in enumerate(_ONES) if n}
for _t in range(2, 10):
    WORD_NUMBERS[_TENS[_t]] = _t * 10
    for _o in range(1, 10):
        for _sep in (' ', '-', ''):
            WORD_NUMBERS[f'{_TENS[_t]}{_sep}{_ONES[_o]}'] = _t * 10 + _o

# Longest first so "fifty one" wins over "fifty".
_WORDS = '|'.join(sorted((re.escape(w) for w in WORD_NUMBERS), key=len, reverse=True))
_NUMBER = rf'(\d{{1,2}}|{_WORDS})'
_SUFFIX = r'(?:[\s-]?(pd|pi|f|a))?\b'

TEN_CODE_RE = re.compile(rf'\b(?:10|ten)[\s.,-]+{_NUMBER}{_SUFFIX}', re.I)
SIGNAL_RE = re.compile(rf'\bsignal[\s.,-]+{_NUMBER}\b', re.I)


def _to_int(token: str) -> int:
    return int(token) if token.isdigit() else WORD_NUMBERS[token.lower()]


def normalize_radio_codes(text: str) -> str:
    """Rewrite spoken or oddly spaced Durham codes as "10-NN" / "Signal NN"."""
    def ten_code(m):
        n = _to_int(m.group(1))
        if n not in TEN_CODES:
            return m.group(0)
        suffix = (m.group(2) or '').lower()
        if suffix and suffix in SUFFIXES.get(n, ()):
            return f'10-{n}{suffix if suffix == "a" else suffix.upper()}'
        # Unknown suffix: keep it as ordinary text after the code.
        return f'10-{n}' + m.group(0)[m.end(1) - m.start(0):]

    def signal(m):
        n = _to_int(m.group(1))
        return f'Signal {n}' if n in SIGNALS else m.group(0)

    return SIGNAL_RE.sub(signal, TEN_CODE_RE.sub(ten_code, text))
