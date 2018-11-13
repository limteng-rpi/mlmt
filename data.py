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

# from torch.autograd import Variable
# from util import get_logger

#
# logger = get_logger(__name__)
#
#
# PARSERS = {}
# DATASETS = {}
#
#
# def register_parser(name):
#     def register(cls):
#         if name not in PARSERS:
#             PARSERS[name] = cls
#         return cls
#     return register
#
#
# def register_dataset(name):
#     def register(cls):
#         if name not in DATASETS:
#             DATASETS[name] = cls
#         return cls
#     return register
#
#
# def create_parser(name, conf):
#     if name in PARSERS:
#         return PARSERS[name](conf)
#     else:
#         raise ValueError('Parser {} is not registered'.format(name))
#
#
# def create_dataset(name, conf):
#     if name in DATASETS:
#         return DATASETS[name](conf)
#     else:
#         raise ValueError('Dataset {} is not registered'.format(name))
#
#
# def compute_metadata(datasets):
#     """Compute tokens, labels, and characters in the given data sets.
#
#     :param datasets: A list of data sets.
#     :return: dicts of token, label, and character counts.
#     """
#     token_count = defaultdict(int)
#     label_count = defaultdict(int)
#     char_count = defaultdict(int)
#
#     for dataset in datasets:
#         if dataset:
#             t, l, c = dataset.metadata()
#             for k, v in t.items():
#                 token_count[k] += v
#             for k, v in l.items():
#                 label_count[k] += v
#             for k, v in c.items():
#                 char_count[k] += v
#
#     return token_count, label_count, char_count
#
#
# def count2vocab(counts,
#                 start_idx=0,
#                 ignore_case=False,
#                 min_count=0,
#                 sort=False,
#                 sort_func=lambda x: (len(x[0]), x[0])):
#     """
#
#     :param counts:
#     :param start_idx:
#     :param ignore_case:
#     :param min_count:
#     :param sort: Sort the keys.
#     :param sort_func: Key sorting lambda function.
#     :return:
#     """
#
#     current_idx = start_idx
#     merge_count = defaultdict(int)
#     for count in counts:
#         for k, v in count.items():
#             if ignore_case:
#                 k = k.lower()
#             merge_count[k] += v
#
#     vocab = {}
#     if sort:
#         merge_count_list = [(k, v) for k, v in merge_count.items()]
#         merge_count_list.sort(key=sort_func)
#         for k, v in merge_count_list:
#             if v >= min_count:
#                 vocab[k] = current_idx
#                 current_idx += 1
#     else:
#         for k, v in merge_count.items():
#             if v >= min_count:
#                 vocab[k] = current_idx
#                 current_idx += 1
#     return vocab
#
#
# def numberize_datasets(confs,
#                        token_ignore_case=True,
#                        label_ignore_case=False,
#                        char_ignore_case=False):
#     for dataset, token_vocab, label_vocab, char_vocab in confs:
#         dataset.numberize(token_vocab,
#                           label_vocab,
#                           char_vocab,
#                           token_ignore_case=token_ignore_case,
#                           label_ignore_case=label_ignore_case,
#                           char_ignore_case=char_ignore_case)
#
#
# @register_parser('conll')
# class ConllParser(object):
#     """Parse CoNLL format file."""
#
#     # def __init__(self,
#     #              separator='\t',
#     #              token_col=0,
#     #              label_col=1,
#     #              skip_comment=True):
#     def __init__(self, conf):
#         """
#         :param conf: Config object with the following fields:
#             - separator: Column separator (default='\t').
#             - token_col: Index of the token column.
#             - label_col: Index of the label column.
#             - skip_comment: Skip lines starting with '#'.
#         """
#         # self.separator = separator
#         # self.token_col = token_col
#         # self.label_col = label_col
#         # self.skip_comment = skip_comment
#         self.separator = getattr(conf, 'separator', '\t')
#         self.token_col = getattr(conf, 'token_col', 0)
#         self.label_col = getattr(conf, 'label_col', 1)
#         self.skip_comment = getattr(conf, 'skip_comment', True)
#
#     def parse(self, path):
#         """
#         :param path: Path to the file to be parsed.
#         :return: Lists of tokens and labels.
#         """
#         with open(path, 'r', encoding='utf-8') as r:
#             current_doc = []
#             for line in r:
#                 line = line.strip()
#                 if self.skip_comment and line.startswith('#'):
#                     continue
#                 if line:
#                     segs = line.split(self.separator)
#                     token, label = segs[self.token_col].strip(), segs[self.label_col]
#                     if label in {'B-O', 'I-O', 'E-O', 'S-O'}:
#                         label = 'O'
#                     current_doc.append((token, label))
#                 elif current_doc:
#                     tokens = []
#                     labels = []
#                     for token, label in current_doc:
#                         tokens.append(token)
#                         labels.append(label)
#                     current_doc = []
#                     yield tokens, labels
#             if current_doc:
#                 tokens = []
#                 labels = []
#                 for token, label in current_doc:
#                     tokens.append(token)
#                     labels.append(label)
#                 yield tokens, labels
#
#
# @register_dataset('sequence')
# class SequenceDataset(object):
#
#     # def __init__(self,
#     #              path,
#     #              parser,
#     #              batch_size=1,
#     #              sample=None,
#     #              max_len=10000):
#     def __init__(self, conf):
#         """
#         :param conf: Config object with the following fields:
#             - path: Path to the data set.
#             - parser: File parser.
#             - batch_size: Batch size (default=1).
#             - sample: Sample rate (default=None). It can be set to:
#                * None or 'all': the data set won't be sampled.
#                * An int number: sample <sample> examples from the data set.
#                * A float number in (0, 1]: the data set will be sampled at the given rate.
#             - max_len: Max example token number.
#         """
#
#         # self.path = path
#         # self.parser = parser
#         # self.batch_size = batch_size
#         # self.sample = sample
#         # self.max_len = max_len
#
#         assert hasattr(conf, 'path'), 'dataset path is required'
#         assert hasattr(conf, 'parser'), 'dataset parser is required'
#
#         self.path = getattr(conf, 'path')
#         self.parser = getattr(conf, 'parser')
#         self.batch_size = getattr(conf, 'batch_size', 1)
#         self.sample = getattr(conf, 'sample', None)
#         self.max_len = getattr(conf, 'max_len', 10000)
#
#         self.dataset = []
#         self.batches = []
#         self.dataset_numberized = []
#         self.doc_num = 0
#
#         self.load()
#
#     def load(self):
#         if self.sample is None or self.sample == 'all':
#             self.dataset = [x for x in self.parser.parse(self.path)
#                             if len(x[0]) < self.max_len]
#         elif type(self.sample) is int:
#             self.dataset = [x for x in self.parser.parse(self.path)
#                             if len(x[0]) < self.max_len]
#             if len(self.dataset) > self.sample:
#                 self.dataset = sample(self.dataset, self.sample)
#         elif type(self.sample) is float:
#             assert 0 < self.sample <= 1
#             self.dataset = [x for x in self.parser.parse(self.path)
#                             if uniform(0, 1) < self.sample
#                             and len(x[0]) < self.max_len]
#         self.doc_num = len(self.dataset)
#
#     def metadata(self):
#         token_count = Counter()
#         label_count = Counter()
#         char_count = Counter()
#         for tokens, labels in self.dataset:
#             token_count.update(tokens)
#             label_count.update(labels)
#             for token in tokens:
#                 char_count.update([c for c in token])
#         return token_count, label_count, char_count
#
#     def numberize(self,
#                   token_vocab,
#                   label_vocab,
#                   char_vocab,
#                   token_ignore_case=True,
#                   label_ignore_case=False,
#                   char_ignore_case=False):
#         self.dataset_numberized = []
#         for tokens, labels in self.dataset:
#             if char_ignore_case:
#                 chars = [t.lower() for t in tokens]
#             else:
#                 chars = tokens
#             char_idxs = [[char_vocab[c] if c in char_vocab
#                           else C.UNKNOWN_TOKEN_INDEX for c in t] for t in chars]
#             if token_ignore_case:
#                 tokens = [t.lower() for t in tokens]
#             token_idxs = [token_vocab[x] if x in token_vocab
#                           else token_vocab[x.lower()] if x.lower() in token_vocab
#                           else C.UNKNOWN_TOKEN_INDEX for x in tokens]
#             if label_ignore_case:
#                 labels = [l.lower() for l in labels]
#             label_idxs = [label_vocab[l] for l in labels]
#             self.dataset_numberized.append((token_idxs, label_idxs, char_idxs))
#
#     def sample_batches(self, shuffle_inst=True):
#         self.batches = []
#         inst_idxs = [i for i in range(len(self.dataset_numberized))]
#         if shuffle_inst:
#             shuffle(inst_idxs)
#         self.batches = [inst_idxs[i:i + self.batch_size] for i in
#                         range(0, len(self.dataset_numberized), self.batch_size)]
#
#     def get_batch(self, volatile=False, gpu=False, shuffle_inst=True):
#         if len(self.batches) == 0:
#             self.sample_batches(shuffle_inst)
#
#         batch = self.batches.pop()
#         batch = [self.dataset_numberized[idx] for idx in batch]
#         batch.sort(key=lambda x: len(x[0]), reverse=True)
#
#         seq_lens = [len(s[0]) for s in batch]
#         max_seq_len = max(seq_lens)
#
#         char_lens = []
#         for seq in batch:
#             seq_char_lens = [len(x) for x in seq[2]] \
#                             + [1] * (max_seq_len - len(seq[0]))
#             char_lens.extend(seq_char_lens)
#         max_char_len = max(max(char_lens), 4)
#
#         tokens, labels, chars = [], [], []
#         for t, l, c in batch:
#             tokens.append(t + [0] * (max_seq_len - len(t)))
#             labels.append(l + [0] * (max_seq_len - len(l)))
#             chars_padded = [x + [0] * (max_char_len - len(x))
#                             for x in c] \
#                            + [[0] * max_char_len] * (max_seq_len - len(t))
#             chars.extend(chars_padded)
#
#         tokens = Variable(torch.LongTensor(tokens), volatile=volatile)
#         labels = Variable(torch.LongTensor(labels), volatile=volatile)
#         chars = Variable(torch.LongTensor(chars), volatile=volatile)
#         seq_lens = Variable(torch.LongTensor(seq_lens), volatile=volatile)
#         char_lens = Variable(torch.LongTensor(char_lens), volatile=volatile)
#
#         if gpu:
#             tokens = tokens.cuda()
#             labels = labels.cuda()
#             chars = chars.cuda()
#             seq_lens = seq_lens.cuda()
#             char_lens = char_lens.cuda()
#
#         return tokens, labels, chars, seq_lens, char_lens
#
#     def get_dataset(self, volatile=False, gpu=False, batch_size=100,
#                     shuffle_inst=False):
#         batches_ = self.batches
#         batch_size_ = self.batch_size
#
#         self.batches = []
#         self.batch_size = batch_size
#         self.sample_batches(shuffle_inst=shuffle_inst)
#
#         while self.batches:
#             yield self.get_batch(volatile, gpu)
#
#         self.batches = batches_
#         self.batch_size = batch_size_
#
#     def batch_num(self, batch_size):
#         return -(-len(self.dataset_numberized) // batch_size)