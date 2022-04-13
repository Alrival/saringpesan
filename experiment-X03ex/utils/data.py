import numpy as np
import pandas as pd
from os.path import join

from utils.spell_checker import SpellChecker

from tqdm import tqdm
from sys import version_info
from torch.utils.data import Dataset, DataLoader

use_encoding = 'ISO-8859-1'
symspell = SpellChecker()
symspell.load_vocab(corpus_dir=join('data', 'corpus'),
                    corpus_filename='corpus_dict.txt',
                    bigrams_filename='bigrams_corpus_dict.txt',
                    abbr_dict_filepath=join('data', 'wordlist', 'abbr_dict.tsv'))

"""
    Reference: https://github.com/makcedward/nlp/blob/master/sample/util/nlp-util-symspell.ipynb
"""
def clean_up_text(text, edit_distance=2):
    return symspell.lookup_sentence(text, max_edit_distance=edit_distance)[0].term

class SentimentDataset(Dataset):
    # Static constant variable
    label2index = {'penipuan': 0, 'iklan': 1, 'normal': 2}
    index2label = {0: 'penipuan', 1: 'iklan', 2: 'normal'}
    num_labels = 3
    
    def _read_file(self, filepath):
        with open(filepath, "r+", encoding=use_encoding) as fs:
            raw_content = fs.read()
            content_list = raw_content.split('\n\n')
            data = []
            
            loader_bar = tqdm(content_list, leave=True, desc="Loading {}".format(filepath),
                              bar_format='{desc}|{bar:30}|{percentage:3.0f}%')
            for content in loader_bar:
                try:
                    if content:
                        text,flag = content.split('\t')
                        text = clean_up_text(text)
                        data.append([text,flag]) 
                except:
                    print('[WARN] something went wrong when parsing data')
                    print('[DEBUG] {}'.format(content))
                    continue

        df = pd.DataFrame(data=data, columns=['content','flag'])
        return df

    def _load_dataset(self, path, convert_label2index=True): 
        df = self._read_file(path)
        if convert_label2index:
            df['flag'] = df['flag'].apply(lambda label: self.label2index[label])
        return df
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self._load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
    
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        text,flag = data['content'],data['flag']
        subwords = self.tokenizer.encode(text, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(flag), text
    
    def __len__(self):
        return len(self.data)

class SentimentDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        if version_info >= (3, 0): super().__init__(*args, **kwargs)
        else: super(SentimentDataLoader, self).__init__(*args, **kwargs)
        # -------------------------------- #
        self.collate_fn = self._collate_func
        self.max_seq_len = max_seq_len
        
    def _collate_func(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        # notation: np_arr[row, col]
        # create 2 tables: 1 with [subwords_value and 0], 1 with [1 and 0]
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        # create 1-col table for flag
        flag_batch = np.zeros((batch_size, 1), dtype=np.int64)
        
        seq_list = []
        for i, (subwords_arr, flag_arr, text) in enumerate(batch):
            # truncate if it goes past max length
            subwords_arr = subwords_arr[:max_seq_len]
            
            # notation: np_arr[row, col] | np_arr[row_start:row_end, col_start:col_end]
            subword_batch[i,:len(subwords_arr)] = subwords_arr
            mask_batch[i,:len(subwords_arr)] = 1
            flag_batch[i,0] = flag_arr
            
            seq_list.append(text)
            
        return subword_batch, mask_batch, flag_batch, seq_list