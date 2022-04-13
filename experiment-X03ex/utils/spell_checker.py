from tqdm import tqdm
from os import makedirs
from os.path import join, exists
import re

from utils.helpers import sanitize_string, create_corpus_and_bigram_dict

from symspellpy import SymSpell, Verbosity
from symspellpy.symspellpy import SuggestItem
from symspellpy.editdistance import EditDistance

use_encoding = 'ISO-8859-1'
train_corpus_dir = join('data', 'indo4B_corpus_uncased_div8_cleaned_mod')
corpus_separator = '\n\n'
"""
    ---------- Check if string has specific format ----------
    using: RegEx
"""
def is_currency(phrase):
    exp = re.compile('[0-9]+((\.|\s|\,)[0-9]+)*')
    return re.search(exp, phrase) is not None
def is_series_num(phrase):
    exp = re.compile('^\(?([0-9]{1,6}(\-|\.))+[0-9]{1,6}\)?$')
    return re.match(exp, phrase) is not None
def is_phone_num(phrase):
    exp = re.compile('^\(?(0|\+[0-9]{1,2})\)?[0-9]{1,13}\)?$')
    return re.match(exp, phrase) is not None
def is_website(phrase):
    exp = re.compile('^(https?:\/\/)?((.+\.)+[a-z]+|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})\/?(.*\/?)*$')
    return re.match(exp, phrase) is not None
def is_special_char(phrase):
    exp = re.compile(r'^[^\w\s]+$')
    return re.match(exp, phrase) is not None

def has_specific_format(strings):
    if is_currency(strings): return True
    elif is_series_num(strings): return True
    elif is_phone_num(strings): return True
    elif is_website(strings): return True
    elif is_special_char(strings): return True
    else: return False

    
"""
    Reference: https://github.com/makcedward/nlp/blob/master/aion/util/spell_check.py
"""
class SpellChecker:
    def __init__(self, dictionary=None, dict_filepath=''):
        self.dictionary = dictionary
        self.dict_filepath = dict_filepath
        self.model = None
        self.abbr_dict = {}
        self.N = 0
        
    def create_new_vocabs(self, separator, vocabs_dir, vocabs_name):
        if not exists(vocabs_dir):
            makedirs(vocabs_dir)

        dict_list = create_corpus_and_bigram_dict(train_corpus_dir, separator=separator)
        if dict_list is None: print('[E] Invalid filepath or separator')
        else:
            corpus = dict_list[0]
            bigrams = dict_list[1]

            # ----------------------- Write corpus dictionary -----------------------
            with open(join(vocabs_dir, vocabs_name), "w", encoding=use_encoding) as text_file:
                loader_bar = tqdm(corpus.items(), leave=True, desc="Creating corpus dictionary",
                                  bar_format='{desc}|{bar:30}|{percentage:3.0f}%')
                """
                    Data format:
                        token frequency
                    Example:
                        edward 154
                        edwards 50
                        ...
                """ 
                for token, frequency in loader_bar:
                    # we don't need to include words which only occurs 1 or 2 times
                    # because it's most likely a non-existent or jumbled word
                    if frequency > 2:
                        text_file.write(token + ' ' + str(frequency))
                        text_file.write('\n')

            # ----------------------- Write bigrams dictionary -----------------------
            with open(join(vocabs_dir, 'bigrams_'+vocabs_name), "w", encoding=use_encoding) as text_file:
                loader_bar = tqdm(bigrams.items(), leave=True, desc="Creating bigrams dictionary",
                                  bar_format='{desc}|{bar:30}|{percentage:3.0f}%')
                """
                    Data format:
                        token_1 token_2 frequency
                    Example:
                        edward edward2 154
                        edwards edwards2 50
                        ...
                """ 
                for tokens, frequency in loader_bar:
                    # we don't need to include words which only occurs 1 or 2 times
                    # because it most likely wouldn't be used often
                    if frequency > 2:
                        text_file.write(tokens + ' ' + str(frequency))
                        text_file.write('\n')

            # Clear unused value
            corpus = None
            bigrams = None
                
    def load_vocab(self, corpus_dir,
                   corpus_filename, bigrams_filename, abbr_dict_filepath,
                   max_edit_distance=2, prefix_length=4):
        
        self.model = SymSpell(
            max_dictionary_edit_distance=max_edit_distance, 
            prefix_length=prefix_length)

        """
        [0] term_index | [1] count_index
        ---------------------------------
                x      |         y       
        """
        """ (dictionary_filename, term_col_in_file, count_col_in_file) """
        
        if not exists(join(corpus_dir, corpus_filename))\
            or not exists(join(corpus_dir, bigrams_filename)):
            self.create_new_vocabs(corpus_separator, corpus_dir, corpus_filename)
        
        if self.model.load_dictionary(join(corpus_dir, corpus_filename), 0, 1)\
            and self.model.load_bigram_dictionary(join(corpus_dir, bigrams_filename), 0, 2):

            with open(join(corpus_dir, corpus_filename), 'r', encoding=use_encoding) as fread:
                self.N = len(fread.readlines())

            if exists(abbr_dict_filepath):
                with open(abbr_dict_filepath, 'r', encoding=use_encoding) as fread:
                    for item in fread.read().split('\n'):
                        line = item.split('\t')
                        if line and len(line) > 1: self.abbr_dict.update({line[0]:line[-1]})
                
        else:
            print("[E] Dictionary can not be loaded")

    def lookup_sentence(self, phrase,
                        max_edit_distance=2,
                        mode='closest'):
        # ------------ Custom code ------------ #
        if mode == 'closest': suggestion_verbosity = Verbosity.CLOSEST
        elif mode == 'top': suggestion_verbosity = Verbosity.TOP
        elif mode == 'all': suggestion_verbosity = Verbosity.ALL
        # ------------ Custom code ------------ #
        """
        Original code:
        ---------------------------------------
        # Parse input string into single terms
        term_list_1 = helpers.parse_words(
            phrase, split_by_space=split_phrase_by_space)
        # Second list of single terms with preserved cases so we can
        # ignore acronyms (all cap words)
        if ignore_non_words:
            term_list_2 = helpers.parse_words(
                phrase, preserve_case=True,
                split_by_space=split_phrase_by_space)
        ---------------------------------------
        """
        # Parse input string into single terms
        term_list_1 = sanitize_string(phrase, to_lower=True).split()

        suggestions = list()
        suggestion_parts = list()
        distance_comparer = EditDistance(self.model._distance_algorithm)  # SymSpell() use Damerau-Levenshtein Algorithm

        # translate every item to its best suggestion, otherwise it
        # remains unchanged
        is_last_combi = False
        for i, __ in enumerate(term_list_1):
            """
            Original code:
            ---------------------------------------
            if ignore_non_words:
                if helpers.try_parse_int64(term_list_1[i]) is not None:
                    suggestion_parts.append(SuggestItem(term_list_1[i], 0, 0))
                    continue
                if helpers.is_acronym(
                        term_list_2[i],
                        match_any_term_with_digits=ignore_term_with_digits):
                    suggestion_parts.append(SuggestItem(term_list_2[i], 0, 0))
                    continue
            ---------------------------------------
            """
            # ------------ Custom code ------------ #
            if has_specific_format(term_list_1[i]):
                suggestion_parts.append(SuggestItem(term_list_1[i], 0, 0))
                continue
            # might want to search with case insensitive word or phrase 
            elif term_list_1[i].lower() in self.abbr_dict:
                suggestion_parts.append(SuggestItem(self.abbr_dict.get(term_list_1[i].lower()), 0, 0))
                continue
            # ------------ Custom code ------------ #
            """
            Original code:
            ---------------------------------------
            suggestions = self.lookup(term_list_1[i], Verbosity.TOP,
                                      max_edit_distance)
            ---------------------------------------
            """
            suggestions = self.model.lookup(term_list_1[i],
                                      suggestion_verbosity,
                                      max_edit_distance)

            # combi check, always before split
            if i > 0 and not is_last_combi:
                """
                Original code:
                ---------------------------------------
                suggestions_combi = self.lookup(
                    term_list_1[i - 1] + term_list_1[i], Verbosity.TOP,
                    max_edit_distance)
                ---------------------------------------
                """
                suggestions_combi = self.model.lookup(term_list_1[i - 1] + term_list_1[i],
                                                suggestion_verbosity,
                                                max_edit_distance)
                if suggestions_combi:
                    best_1 = suggestion_parts[-1]
                    if suggestions:
                        best_2 = suggestions[0]
                    else:
                        # estimated word occurrence probability
                        # P=10 / (N * 10^word length l)
                        best_2 = SuggestItem(term_list_1[i],
                                             max_edit_distance + 1,
                                             10 // 10 ** len(term_list_1[i]))
                    # distance_1=edit distance between 2 split terms and
                    # their best corrections : als comparative value
                    # for the combination
                    distance_1 = best_1.distance + best_2.distance
                    if (distance_1 >= 0
                            and (suggestions_combi[0].distance + 1 < distance_1
                                 or (suggestions_combi[0].distance + 1 == distance_1
                                     and (suggestions_combi[0].count > best_1.count / self.N * best_2.count)))):
                        suggestions_combi[0].distance += 1
                        suggestion_parts[-1] = suggestions_combi[0]
                        is_last_combi = True
                        continue
            is_last_combi = False

            # alway split terms without suggestion / never split terms
            # with suggestion ed=0 / never split single char terms
            if suggestions and (suggestions[0].distance == 0
                                or len(term_list_1[i]) == 1):
                # choose best suggestion
                suggestion_parts.append(suggestions[0])
            else:
                # if no perfect suggestion, split word into pairs
                suggestion_split_best = None
                # add original term
                if suggestions:
                    suggestion_split_best = suggestions[0]
                if len(term_list_1[i]) > 1:
                    for j in range(1, len(term_list_1[i])):
                        part_1 = term_list_1[i][: j]
                        part_2 = term_list_1[i][j :]

                        """
                        Original code:
                        ---------------------------------------
                        suggestions_1 = self.lookup(part_1, Verbosity.TOP,
                                                    max_edit_distance)
                        ---------------------------------------
                        """
                        suggestions_1 = self.model.lookup(part_1,
                                                    suggestion_verbosity,
                                                    max_edit_distance)
                        if suggestions_1:
                            """
                            Original code:
                            ---------------------------------------
                            suggestions_2 = self.lookup(part_2, Verbosity.TOP,
                                                        max_edit_distance)
                            ---------------------------------------
                            """
                            suggestions_2 = self.model.lookup(part_2,
                                                        suggestion_verbosity,
                                                        max_edit_distance)
                            if suggestions_2:
                                tmp_term = (suggestions_1[0].term + " " +
                                            suggestions_2[0].term)
                                # select best suggestion for split pair
                                tmp_distance = distance_comparer.compare(
                                    term_list_1[i], tmp_term,
                                    max_edit_distance)
                                if tmp_distance < 0:
                                    tmp_distance = max_edit_distance + 1
                                if suggestion_split_best is not None:
                                    if tmp_distance > suggestion_split_best.distance:
                                        continue
                                    if tmp_distance < suggestion_split_best.distance:
                                        suggestion_split_best = None                                
                                if tmp_term in self.model._bigrams:
                                    tmp_count = self.model._bigrams[tmp_term]
                                    # increase count, if split
                                    # corrections are part of or
                                    # identical to input single term
                                    # correction exists
                                    if suggestions:
                                        best_si = suggestions[0]
                                        # alternatively remove the
                                        # single term from
                                        # suggestion_split, but then
                                        # other splittings could win
                                        if suggestions_1[0].term + suggestions_2[0].term == term_list_1[i]:
                                            # make count bigger than
                                            # count of single term
                                            # correction
                                            tmp_count = max(tmp_count,
                                                            best_si.count + 2)
                                        elif (suggestions_1[0].term == best_si.term
                                              or suggestions_2[0].term == best_si.term):
                                            # make count bigger than
                                            # count of single term
                                            # correction
                                            tmp_count = max(tmp_count,
                                                            best_si.count + 1)
                                    # no single term correction exists
                                    elif suggestions_1[0].term + suggestions_2[0].term == term_list_1[i]:
                                        tmp_count = max(
                                            tmp_count,
                                            max(suggestions_1[0].count,
                                                suggestions_2[0].count) + 2)
                                else:
                                    # The Naive Bayes probability of
                                    # the word combination is the
                                    # product of the two word
                                    # probabilities: P(AB)=P(A)*P(B)
                                    # use it to estimate the frequency
                                    # count of the combination, which
                                    # then is used to rank/select the
                                    # best splitting variant
                                    tmp_count = min(
                                        self.model.bigram_count_min,
                                        int(suggestions_1[0].count /
                                            self.N * suggestions_2[0].count))
                                suggestion_split = SuggestItem(
                                    tmp_term, tmp_distance, tmp_count)
                                if (suggestion_split_best is None or
                                        suggestion_split.count > suggestion_split_best.count):
                                    suggestion_split_best = suggestion_split

                    if suggestion_split_best is not None:
                        # select best suggestion for split pair
                        suggestion_parts.append(suggestion_split_best)
                        self.model._replaced_words[term_list_1[i]] = suggestion_split_best
                    else:
                        si = SuggestItem(term_list_1[i],
                                         max_edit_distance + 1,
                                         int(10 / 10 ** len(term_list_1[i])))
                        suggestion_parts.append(si)
                        self.model._replaced_words[term_list_1[i]] = si
                else:
                    # estimated word occurrence probability
                    # P=10 / (N * 10^word length l)
                    si = SuggestItem(term_list_1[i], max_edit_distance + 1,
                                     int(10 / 10 ** len(term_list_1[i])))
                    suggestion_parts.append(si)
                    self.model._replaced_words[term_list_1[i]] = si
        
        joined_term = ""
        joined_count = self.N
        for si in suggestion_parts:
            joined_term += si.term + " "
            joined_count *= si.count / self.N
        joined_term = joined_term.rstrip()

        """
        Original code:
        ---------------------------------------
        if transfer_casing:
            joined_term = helpers.transfer_casing_for_similar_text(phrase,
                                                                   joined_term)
        ---------------------------------------
        """

        suggestion = SuggestItem(joined_term,
                                 distance_comparer.compare(
                                     phrase, joined_term,
                                     2 ** 31 - 1),
                                 int(joined_count))

        suggestions_line = list()
        suggestions_line.append(suggestion)

        return suggestions_line
