import torch
import constant as C

import logging
from collections import Counter, defaultdict
from random import shuffle, uniform, sample
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger()


def count2vocab(count, offset=0, pads=None, min_count=0, ignore_case=False):
    """Convert a token count dictionary to a vocabulary dict.
    :param count: Token count dictionary.
    :param offset: Begin start offset.
    :param pads: A list of padding (token, index) pairs.
    :param min_count: Minimum token count.
    :param ignore_case: Ignore token case.
    :return: Vocab dict.
    """
    if ignore_case:
        count_ = defaultdict(int)
        for k, v in count.items():
            count_[k.lower()] += v
        count = count_

    vocab = {}
    for token, freq in count.items():
        if freq > min_count:
            vocab[token] = len(vocab) + offset
    if pads:
        for k, v in pads:
            vocab[k] = v

    return vocab



class Parser(object):

    def parse(self, path: str):
        raise NotImplementedError


class ConllParser(Parser):

    def __init__(self,
                 token_col: int = 0,
                 label_col: int = 1,
                 separator: str = '\t',
                 skip_comment: bool = False):
        """
        :param token_col: Token column (default=0).
        :param label_col: Label column (default=1).
        :param separator: Separate character (default=\t).
        :param skip_comment: Skip lines starting with #.
        """
        self.token_col = token_col
        self.label_col = label_col
        self.separator = separator
        self.skip_comment = skip_comment

    def parse(self,
              path: str):
        token_col = self.token_col
        label_col = self.label_col
        separator = self.separator
        skip_comment = self.skip_comment

        with open(path, 'r', encoding='utf-8') as r:
            current_doc = []
            for line in r:
                line = line.rstrip()
                if skip_comment and line.startswith('#'):
                    continue
                if line:
                    segs = line.split(separator)
                    token, label = segs[token_col].strip(), segs[label_col]
                    token = C.PENN_TREEBANK_BRACKETS.get(token, token)
                    if label in {'B-O', 'I-O', 'E-O', 'S-O'}:
                        label = 'O'
                    current_doc.append((token, label))
                elif current_doc:
                    tokens = []
                    labels = []
                    for token, label in current_doc:
                        tokens.append(token)
                        labels.append(label)
                    current_doc = []
                    yield tokens, labels
            if current_doc:
                tokens = []
                labels = []
                for token, label in current_doc:
                    tokens.append(token)
                    labels.append(label)
                yield tokens, labels


class SeqLabelDataset(Dataset):

    def __init__(self,
                 path: str,
                 parser: Parser,
                 max_seq_len: int = -1):
        self.path = path
        self.parser = parser
        self.max_seq_len = max_seq_len
        self.raw_data = []
        self.data = []
        self.load()

    def __getitem__(self,
                    idx: int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def numberize(self,
                  token_vocab: dict,
                  label_vocab: dict,
                  char_vocab: dict = None,
                  ignore_case: bool = False):
        for tokens, labels in self.raw_data:
            if ignore_case:
                tokens_ = [t.lower() for t in tokens]
                tokens_ = [token_vocab[t] if t in token_vocab
                           else C.UNK_INDEX for t in tokens_]
            else:
                tokens_ = [token_vocab[t] if t in token_vocab
                           else C.UNK_INDEX for t in tokens]
            labels_ = [label_vocab[l] for l in labels]
            chars = None
            if char_vocab:
                chars = [[char_vocab[c] if c in char_vocab
                          else C.UNK_INDEX for c in t] for t in tokens]
                if self.max_seq_len > 0:
                    chars = chars[:self.max_seq_len]
            if self.max_seq_len > 0:
                tokens_ = tokens_[:self.max_seq_len]
                labels_ = labels_[:self.max_seq_len]
            self.data.append((tokens_, labels_, chars))

    def load(self):
        self.raw_data = [(tokens, labels)
                     for tokens, labels in self.parser.parse(self.path)]

    def stats(self,
              token_ignore_case: bool = False,
              char_ignore_case: bool = False,
              label_ignore_case: bool = False,
              ):
        token_counter = Counter()
        char_counter = Counter()
        label_counter = Counter()
        for item in self.raw_data:
            tokens, labels = item[0], item[1]
            token_lower = [t.lower() for t in tokens]
            if char_ignore_case:
                for token in token_lower:
                    for c in token:
                        char_counter[c] += 1
            else:
                for token in tokens:
                    for c in token:
                        char_counter[c] += 1
            if token_ignore_case:
                token_counter.update(token_lower)
            else:
                token_counter.update(tokens)
            if label_ignore_case:
                label_counter.update([l.lower() for l in labels])
            else:
                label_counter.update(labels)

        return token_counter, char_counter, label_counter


class BatchProcessor(object):

    def process(self, batch):
        assert NotImplementedError


class SeqLabelProcessor(BatchProcessor):

    def __init__(self,
                 sort: bool = True,
                 gpu: bool = False,
                 padding_idx: int = C.PAD_INDEX,
                 min_char_len: int = 4):
        self.sort = sort
        self.gpu = gpu
        self.padding_idx = padding_idx
        self.min_char_len = min_char_len

    def process(self, batch: list):
        padding_idx = self.padding_idx
        # if self.sort:
        batch.sort(key=lambda x: len(x[0]), reverse=True)

        seq_lens = [len(x[0]) for x in batch]
        max_seq_len = max(seq_lens)

        char_lens = []
        for seq in batch:
            seq_char_lens = [len(x) for x in seq[2]] + \
                            [padding_idx] * (max_seq_len - len(seq[0]))
            char_lens.extend(seq_char_lens)
        max_char_len = max(max(char_lens), self.min_char_len)

        # Padding
        batch_tokens = []
        batch_labels = []
        batch_chars = []
        for tokens, labels, chars in batch:
            batch_tokens.append(tokens + [padding_idx] * (max_seq_len - len(tokens)))
            batch_labels.append(labels + [padding_idx] * (max_seq_len - len(tokens)))
            batch_chars.extend([x + [0] * (max_char_len - len(x)) for x in chars]
                               + [[0] * max_char_len] * (max_seq_len - len(tokens)))

        batch_tokens = torch.LongTensor(batch_tokens)
        batch_labels = torch.LongTensor(batch_labels)
        batch_chars = torch.LongTensor(batch_chars)
        seq_lens = torch.LongTensor(seq_lens)
        char_lens = torch.LongTensor(char_lens)

        if self.gpu:
            batch_tokens = batch_tokens.cuda()
            batch_labels = batch_labels.cuda()
            batch_chars = batch_chars.cuda()
            seq_lens = seq_lens.cuda()
            char_lens = char_lens.cuda()

        return (batch_tokens, batch_labels, batch_chars,
                seq_lens, char_lens)