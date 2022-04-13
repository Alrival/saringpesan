from tqdm import tqdm
from os import listdir
from os.path import join, exists, isdir
from collections import Counter

import re
from nltk import ngrams

# use_encoding = 'ISO-8859-1'
use_encoding = 'UTF-8'
symbol_dict_filepath = join('data', 'wordlist', 'normalised_symbol.tsv')
dict_separator = '\n'
split_separator = '\t'

# --------------- Load symbols dictionary --------------- #
symbol_dict = {}
with open(symbol_dict_filepath, 'r+', encoding=use_encoding) as fs:
    data = fs.read()
    lines = data.split(dict_separator)

    for line in lines:
        tokens = line.split(split_separator)
        symbol_dict.update({ str(tokens[0].encode(use_encoding)): tokens[1] })
# --------------- Load symbols dictionary --------------- #

# --------------- Regex dictionary --------------- #
find_expr = {
    'is_emoji': re.compile('^\:\w+\:$'),
    'has_value': re.compile('\d+([a-zA-Z]+|\%)'),
    'is_repetitive': re.compile('\w{2,}(2|\")'),
    'has_value_range': re.compile('^\d+([a-zA-Z]+|\%)(\~|\-|\/)\d+([a-zA-Z]+|\%)$'),
    'has_inbetween': re.compile('^\w*(\/|\_|\-|\:)\w*$'),
    'has_multiple_value-exp1': re.compile('^[A-Za-z]{2,}(\,[A-Za-z]{2,})+$'),
    'has_multiple_value-exp2': re.compile('^[A-Za-z]{2,}\&[A-Za-z]{2,}$'),
    'has_multiple_value-exp3': re.compile('^([A-Za-z]{2,}(\&|\/))?[A-Za-z]{2,}(\,([A-Za-z]{2,}(\&|\/))?[A-Za-z]{2,})+$'),
    'has_inbetween_abbr': re.compile('^\w([^\w\s]\w)+$'),
    'is_idr_currency': re.compile('[Rr][Pp](\s|\.)\d+((\.|\s|\,)\d+)*'),
    'has_website': re.compile('(https?:\/\/)?((.+\.)+[a-z]+|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})\/?(.*\/?)*'),
    'is_series_num': re.compile('\(?([0-9]{1,6}(\-|\.))+[0-9]{1,6}\)?'),
    'is_phone_num': re.compile('\(?(0|\+[0-9]{1,2})\)?[0-9]{1,13}\)?'),
    'has_punctuation': re.compile(r'[^\w\s]?(.*[^\w\s])+$')
}
fetch_expr = {
    'get_website_link': re.compile('(https?:\/\/)?((.+\.)+[a-z]+|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})\/?(.*\/?)*'),
    'get_series_num': re.compile(r'([0-9]{1,6})(\-|\.|\,)?'),
    'get_phone_num': re.compile(r'\(?(0|\+[0-9]{1,2})\)?[0-9]{1,13}\)?')
}
match_inbetween = {
    'word_': re.compile(r'^\w+(\/|\_|\-)$'),
    '_word': re.compile(r'^(\/|\_)\w+$'),
    'word:': re.compile(r'^\w+\:$'),
    ':word': re.compile(r'^\:\w+$'),
    'a/b': re.compile(r'[A-Za-z]\/[A-Za-z]'),
    'a-word': re.compile(r'[A-Za-z]\-\w{2,}'),
    'a/word': re.compile(r'\w\/\w{2,}'),
    'word/b': re.compile(r'\w{2,}\/\w'),
    'word/word': re.compile(r'\w{2,}\/\w{2,}'),
    'word_word': re.compile(r'\w+(\_|\-)\w+'),
    'word:word': re.compile(r'\w+\:\w+'),
    'word,word+': re.compile(r'^[A-Za-z]{2,}(\,[A-Za-z]{2,})+$'),
    'word&word': re.compile(r'^[A-Za-z]{2,}\&[A-Za-z]{2,}$'),
}
# --------------- Regex dictionary --------------- #

# ------------------------------ String filters ------------------------------ #
def split_per_sentence(text):
    return [sentence for sentence in re.split(r'(\.|\?|\!)\s+', text)]
def filter_symbol(sentence):
    return re.sub(r'[^a-zA-Z\d\s]', lambda substr: symbol_dict.get( str(substr.group().encode(use_encoding)), substr.group()), sentence)
def filter_sentence(sentence):
    line = sentence.strip()
    line = line.replace('\n', ' ').replace('\t', ' ')
    line = filter_symbol(line)
    return re.sub(r'\s{2,}', ' ', line).strip()
"""
    ---------- Sanitize string input ----------
    using: RegEx
"""
def is_emoji(phrase):
    return find_expr['is_emoji'].search(phrase) is not None
def has_value(phrase):
    return find_expr['has_value'].search(phrase) is not None
def is_repetitive(phrase):
    return find_expr['is_repetitive'].search(phrase) is not None

def split_value(phrase):
    ext = re.search(r'(\d+)([a-zA-Z]+|\%)', phrase)
    if ext.group(2) == '%':
        unit = 'persen'
    else:
        unit = ext.group(2)
    value = ext.group(1) + ' ' + ext.group(2)
    return value
def norm_value_range(phrase):
    currency = re.split(r'(\~|\-|\/)', phrase)
    if has_value(currency[0]): currency[0] = split_value(currency[0])
    if has_value(currency[-1]): currency[-1] = split_value(currency[-1])
    return currency[0] + ' sampai ' + currency[-1]
def norm_inbetween(phrase):
    # case 01: word/ | word_ | word- --> word
    if match_inbetween['word_'].match(phrase): strings = re.sub(r'(\w+)(\/|\_|\-)', r'\1', phrase) 
    # case 02: /word | _word --> word
    elif match_inbetween['_word'].match(phrase): strings = re.sub(r'(\/|\_)(\w+)', r'\2', phrase)
    # case 03: word: --> word :
    elif match_inbetween['word:'].match(phrase): strings = phrase.replace(':',' :')
    # case 04: :word --> word
    elif match_inbetween[':word'].match(phrase): strings = phrase.replace(':', '')
    # case 05a: a/b
    elif match_inbetween['a/b'].match(phrase):
        strings_part = re.search(r'([A-Za-z])(\/)([A-Za-z])', phrase)
        strings = strings_part.group(1) + '/' + strings_part.group(3)
    # case 05x: a-word
    elif match_inbetween['a-word'].match(phrase): strings = phrase
    # case 05b: a/word | word/b
    elif match_inbetween['a/word'].match(phrase) or match_inbetween['word/b'].match(phrase):
        strings_part = re.search(r'(\w+)(\/)(\w+)', phrase)
        strings = strings_part.group(1) + ' ' + strings_part.group(3)
    # case 05c: word/word
    elif match_inbetween['word/word'].match(phrase):
        strings_part = re.search(r'(\w{2,})(\/)(\w{2,})', phrase)
        strings = strings_part.group(1) + ' / ' + strings_part.group(3)
        strings = re.sub(r'\s{2,}', ' ', strings)
    # case 06: word_word | word-word
    elif match_inbetween['word_word'].match(phrase):
        strings_part = re.search(r'(\w+)(\_|\-)(\w+)', phrase)
        strings = strings_part.group(1) + ' ' + strings_part.group(3)
        strings = re.sub(r'\s{2,}', ' ', strings)
    # case 07: word:word
    elif match_inbetween['word:word'].match(phrase):
        strings_part = re.search(r'(\w+)\:(\w+)', phrase)
        if has_value(strings_part.group(2)):
            strings = strings_part.group(1) + ' : ' + split_value(strings_part.group(2))
        elif has_multiple_value(strings_part.group(2)):
            strings = strings_part.group(1) + ' : ' + norm_multiple_value(strings_part.group(2))
        else:
            strings = strings_part.group(1) + ' : ' + strings_part.group(2)
        strings = re.sub(r'\s{2,}', ' ', strings)
    else: strings = phrase
    
    return strings
def norm_multiple_value(phrase):
    # case 01: word,word,word,...
    if match_inbetween['word,word+'].match(phrase):
        ext = re.split(r'\,', phrase)
        return ' , '.join(ext) 
    # case 02: word&word
    elif match_inbetween['word&word'].match(phrase):
        strings_part = re.search(r'([A-Za-z]{2,})\&([A-Za-z]{2,})', phrase)
        return strings_part.group(1) + ' dan ' + strings_part.group(2)
    # case 03: word,word&word,word/word,...
    else:
        ext = re.split(r'\,', phrase)
        word = []
        for item in ext:
            if re.match(r'^[A-Za-z]{2,}\&[A-Za-z]{2,}$', item):
                word.append(item.replace('&', ' dan '))
            elif re.match(r'^[A-Za-z]{2,}\/[A-Za-z]{2,}$', item):
                word.append(item.replace('/', ' atau '))
            else: word.append(item)
        return ' , '.join(word)

def norm_idr_currency(phrase):
    currency = re.search(r'[Rr][Pp](\s|\.)(\d+)((?:\.|\s|\,)\d+)*[a-zA-Z]*', phrase)
    strings = currency.group(0).replace('.', ' ', 1)
    if re.search(r'\d+[a-zA-Z]+', strings) is not None:
        strings = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', strings)
        strings = re.sub(r'\s{2,}', ' ', strings)
    return strings

def get_website_link(phrase):
    return fetch_expr['get_website_link'].search(phrase).group()
def get_series_num(phrase):
    combi = ""
    for (num,sep) in fetch_expr['get_series_num'].findall(phrase):
        if sep and sep != "": combi += num+sep
        else: combi += num
    return combi
def get_phone_num(phrase):
    return fetch_expr['get_phone_num'].search(phrase).group()
    
def has_value_range(phrase):
    return find_expr['has_value_range'].search(phrase) is not None
def has_inbetween(phrase):
    return find_expr['has_inbetween'].search(phrase) is not None
def has_multiple_value(phrase):
    if find_expr['has_multiple_value-exp1'].search(phrase):
        return find_expr['has_multiple_value-exp1'].search(phrase) is not None
    if find_expr['has_multiple_value-exp2'].search(phrase):
        return find_expr['has_multiple_value-exp2'].search(phrase) is not None
    if find_expr['has_multiple_value-exp3'].search(phrase):
        return find_expr['has_multiple_value-exp3'].search(phrase) is not None
    return False
def has_inbetween_abbr(phrase):
    return find_expr['has_inbetween_abbr'].search(phrase) is not None
def is_idr_currency(phrase):
    return find_expr['is_idr_currency'].search(phrase) is not None
def has_website(phrase):
    return find_expr['has_website'].search(phrase) is not None
def is_series_num(phrase):
    return find_expr['is_series_num'].search(phrase) is not None
def is_phone_num(phrase):
    return find_expr['is_phone_num'].search(phrase) is not None
def has_punctuation(phrase):
    return find_expr['has_punctuation'].search(phrase) is not None

def convert2lower(text, convert:bool):
    if convert: return text.lower()
    else: return text
def sanitize_string(text, to_lower=False):
    text = filter_sentence(text)
    result = []
    
    for sentence in split_per_sentence(text):
        phrase = re.split('\s+', sentence)
        
        for idx, _ in enumerate(phrase):
            if not is_emoji(phrase[idx]) and not phrase[idx].isdigit():
                if phrase[idx].isalpha(): phrase[idx] = convert2lower(phrase[idx], to_lower)
                elif phrase[idx].isalnum():
                    if has_value(phrase[idx]):
                        phrase[idx] = split_value(phrase[idx])
                        phrase[idx] = convert2lower(phrase[idx], to_lower)
                    elif not phrase[idx].isupper() and is_repetitive(phrase[idx]):
                        phrase[idx] = re.sub(r'(\w+)(2|\")', r'\1 \1', phrase[idx])
                        phrase[idx] = convert2lower(phrase[idx], to_lower)
                    else:
                        phrase[idx] = convert2lower(phrase[idx], to_lower)
                else: 
                    if has_value_range(phrase[idx]): phrase[idx] = convert2lower(norm_value_range(phrase[idx]), to_lower)
                    elif has_inbetween(phrase[idx]): phrase[idx] = convert2lower(norm_inbetween(phrase[idx]), to_lower)
                    elif has_multiple_value(phrase[idx]): phrase[idx] = convert2lower(norm_multiple_value(phrase[idx]), to_lower)
                    elif has_inbetween_abbr(phrase[idx]):
                        phrase[idx] = re.sub(r'[^\w\s]', '', phrase[idx])
                        phrase[idx] = convert2lower(phrase[idx], to_lower)
                    elif is_idr_currency(phrase[idx]): phrase[idx] = norm_idr_currency(phrase[idx])
                    elif has_website(phrase[idx]): phrase[idx] = get_website_link(phrase[idx])
                    elif is_series_num(phrase[idx]): phrase[idx] = get_series_num(phrase[idx])
                    elif is_phone_num(phrase[idx]): phrase[idx] = get_phone_num(phrase[idx])
                    elif has_punctuation(phrase[idx]):
                        phrase[idx] = re.sub(r'([^\w\s])', r' \1 ', phrase[idx])
                        phrase[idx] = re.sub(r'\s{2,}', ' ', phrase[idx]).strip()
                        while re.search(r'[^\w\s]\s[^\w\s]', phrase[idx]):
                            phrase[idx] = re.sub(r'([^\w\s])\s([^\w\s])', r'\1\2', phrase[idx])
                        phrase[idx] = phrase[idx].strip()
                    else:
                        phrase[idx] = phrase[idx]
                        
        result.append(' '.join(phrase))
    
    return ' '.join(result)
# ------------------------------ String filters ------------------------------ #

def create_corpus_and_bigram_dict(file_dir, separator=''):
    error = False
    if not exists(file_dir) or not isdir(file_dir): error = True
    if separator == '': error = True
    if error == True: return None
    
    corpus = None
    bigrams = None
    loader_bar = tqdm(listdir(file_dir), leave=True, desc="Loading corpus",
                          bar_format='{desc}|{bar:30}|{percentage:3.0f}%')
    
    for text_file in loader_bar:
        data = ''
        with open(join(file_dir, text_file), 'r+', encoding=use_encoding) as fs:
            data = fs.read()
            lines = data.split(separator)
        
        for line in lines:
            line = filter_sentence(line)
            line = re.sub('[^A-Za-z ]', ' ', line)
            line = re.sub(r'\s{2,}', ' ', line)
            tokens = line.split(' ')
            tokens = [token for token in tokens if len(token) > 0]
            # ---------- corpus ----------
            if corpus == None: corpus = Counter(tokens)
            else: corpus.update(tokens)
            # ---------- corpus ----------
            # ---------- bigrams ----------
            words = ngrams(tokens, 2)
            words = ['{} {}'.format(word_1,word_2) for word_1,word_2 in words]
            if bigrams == None: bigrams = Counter(words)
            else: bigrams.update(words)
            # ---------- bigrams ----------
        
    # Clear unused value
    data, line, tokens, words = None, None, None, None
    return corpus, bigrams