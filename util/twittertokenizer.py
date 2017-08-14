import pymorphy2
import re
from nltk.tokenize.casual import _replace_html_entities, remove_handles, reduce_lengthening, WORD_RE, \
    EMOTICON_RE

HANG_RE = re.compile(r'([^a-zA-Z0-9])\1{3,}')

class TweetTokenizer:
    r"""
    Tokenizer for tweets.

        >>> from nltk.tokenize import TweetTokenizer
        >>> tknzr = TweetTokenizer()
        >>> s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
        >>> tknzr.tokenize(s0)
        ['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3', 'and', 'some', 'arrows', '<', '>', '->', '<--']

    Examples using `strip_handles` and `reduce_len parameters`:

        >>> tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
        >>> s1 = '@remy: This is waaaaayyyy too much for you!!!!!!'
        >>> tknzr.tokenize(s1)
        [':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!']
    """

    def __init__(self, preserve_case=True, reduce_len=False, strip_handles=False):
        self.preserve_case = preserve_case
        self.reduce_len = reduce_len
        self.strip_handles = strip_handles

    def tokenize(self, text):
        """
        :param text: str
        :rtype: list(str)
        :return: a tokenized list of strings; concatenating this list returns\
        the original string if `preserve_case=False`
        """
        # Fix HTML character entities:
        text = _replace_html_entities(text)
        # Remove username handles
        if self.strip_handles:
            text = remove_handles(text)
        # Normalize word lengthening
        if self.reduce_len:
            text = reduce_lengthening(text)
        # Shorten problematic sequences of characters
        safe_text = HANG_RE.sub(r'\1\1\1', text)
        # Tokenize:
        words = WORD_RE.findall(safe_text)
        morph = pymorphy2.MorphAnalyzer()
        # Possibly alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:
            words = list(map((lambda x: x if EMOTICON_RE.search(x) else
                              morph.parse(x)[0].normal_form.lower()), words))
        return words
