import torch
from torch.autograd import Variable
from collections import Counter
from random import shuffle, uniform, sample
import constant as C


class ConllParser(object):
    """Parse CoNLL format file."""

    def __init__(self,
                 separator='\t',
                 token_col=0,
                 label_col=1,
                 skip_comment=True):
        """
        :param separator: Column separator (default='\t').
        :param token_col: Index of the token column.
        :param label_col: Index of the label column.
        :param skip_comment: Skip lines starting with '#'.
        """
        self.separator = separator
        self.token_col = token_col
        self.label_col = label_col
        self.skip_comment = skip_comment

    def parse(self, path):
        """
        :param path: Path to the file to be parsed. 
        :return: Lists of tokens and labels.
        """
        with open(path, 'r', encoding='utf-8') as r:
            current_doc = []
            for line in r:
                line = line.strip()
                if self.skip_comment and line.startswith('#'):
                    continue
                if line:
                    segs = line.split(self.separator)
                    token, label = segs[self.token_col], segs[self.label_col]
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


class SequenceDataset(object):

    def __init__(self,
                 path,
                 parser,
                 batch_size=1,
                 sample=None,
                 max_len=10000):
        """
        :param path: Path to the data set. 
        :param parser: File parser.
        :param batch_size: Batch size (default=1).
        :param sample: Sample rate (default=None). It can be set to:
               - None or 'all': the data set won't be sampled.
               - An int number: sample <sample> examples from the data set.
               - A float number in (0, 1]: the data set will be sampled at the given rate.
        :param max_len: Max example token number.
        """

        self.path = path
        self.parser = parser
        self.batch_size = batch_size
        self.sample = sample
        self.max_len = max_len

        self.dataset = []
        self.batches = []
        self.dataset_numberized = []

        self.load()

    def load(self):
        if self.sample is None or self.sample == 'all':
            self.dataset = [x for x in self.parser.parse(self.path)
                            if len(x[0]) < self.max_len]
        elif type(self.sample) is int:
            self.dataset = [x for x in self.parser.parse(self.path)
                            if len(x[0]) < self.max_len]
            if len(self.dataset) > self.sample:
                self.dataset = sample(self.dataset, self.sample)
        elif type(self.sample) is float:
            assert 0 < self.sample <= 1
            self.dataset = [x for x in self.parser.parse(self.path)
                            if uniform(0, 1) < self.sample
                            and len(x[0]) < self.max_len]

    def metadata(self):
        token_count = Counter()
        label_count = Counter()
        char_count = Counter()
        for tokens, labels in self.dataset:
            token_count.update(tokens)
            label_count.update(labels)
            for token in tokens:
                char_count.update([c for c in token])
        return token_count, label_count, char_count

    def numberize(self,
                  token_vocab,
                  label_vocab,
                  char_vocab,
                  token_ignore_case=True,
                  label_ignore_case=False,
                  char_ignore_case=False):
        self.dataset_numberized = []
        for tokens, labels in self.dataset:
            if char_ignore_case:
                chars = [t.lower() for t in tokens]
            else:
                chars = tokens
            char_idxs = [[char_vocab[c] if c in char_vocab
                          else C.UNKNOWN_TOKEN_INDEX for c in t] for t in chars]
            if token_ignore_case:
                tokens = [t.lower() for t in tokens]
            token_idxs = [token_vocab[x] if x in token_vocab
                          else C.UNKNOWN_TOKEN_INDEX for x in tokens]
            if label_ignore_case:
                labels = [l.lower() for l in labels]
            label_idxs = [label_vocab[l] for l in labels]
            self.dataset_numberized.append((token_idxs, label_idxs, char_idxs))

    def sample_batches(self):
        self.batches = []
        inst_idxs = [i for i in range(len(self.dataset_numberized))]
        shuffle(inst_idxs)
        self.batches = [inst_idxs[i:i + self.batch_size] for i in
                        range(0, len(self.dataset_numberized), self.batch_size)]

    def get_batch(self, volatile=False, gpu=False):
        if len(self.batches) == 0:
            self.sample_batches()

        batch = self.batches.pop()
        batch = [self.dataset_numberized[idx] for idx in batch]
        batch.sort(key=lambda x: len(x[0]), reverse=True)

        seq_lens = [len(s[0]) for s in batch]
        max_seq_len = max(seq_lens)

        char_lens = []
        for seq in batch:
            seq_char_lens = [len(x) for x in seq[2]] \
                            + [1] * (max_seq_len - len(seq[0]))
            char_lens.extend(seq_char_lens)
        max_char_len = max(max(char_lens), 4)

        tokens, labels, chars = [], [], []
        for t, l, c in batch:
            tokens.append(t + [0] * (max_seq_len - len(t)))
            labels.append(l + [0] * (max_seq_len - len(l)))
            chars_padded = [x + [0] * (max_char_len - len(x))
                            for x in c] \
                           + [[0] * max_char_len] * (max_seq_len - len(t))
            chars.extend(chars_padded)

        tokens = Variable(torch.LongTensor(tokens), volatile=volatile)
        labels = Variable(torch.LongTensor(labels), volatile=volatile)
        chars = Variable(torch.LongTensor(chars), volatile=volatile)
        seq_lens = Variable(torch.LongTensor(seq_lens), volatile=volatile)
        char_lens = Variable(torch.LongTensor(char_lens), volatile=volatile)

        if gpu:
            tokens = tokens.cuda()
            labels = labels.cuda()
            chars = chars.cuda()
            seq_lens = seq_lens.cuda()
            char_lens = char_lens.cuda()

        return tokens, labels, chars, seq_lens, char_lens

    def get_dataset(self, volatile=False, gpu=False, batch_size=100):
        batches_ = self.batches
        batch_size_ = self.batch_size

        self.batches = []
        self.batch_size = batch_size
        self.sample_batches()

        while self.batches:
            yield self.get_batch(volatile, gpu)

        self.batches = batches_
        self.batch_size = batch_size_