import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as I
import torch.nn.utils.rnn as R
import torch.nn.functional as F
import torch.optim as O

from torch.autograd import Variable
from random import shuffle


logger = logging.getLogger(__name__)
MODULES = {}


def log_sum_exp(vec, dim=0):
    """Calculate LogSumExp (used in the CRF layer).
    
    :param vec: Input vector.
    :param dim: 
    :return: 
    """
    m, _ = torch.max(vec, dim)
    m_exp = m.unsqueeze(-1).expand_as(vec)
    return m + torch.log(torch.sum(torch.exp(vec - m_exp), dim))


def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().data[0]

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = Variable(ranges)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask


def sequence_masks(lens):
    batch_size = lens.size(0)

    max_len = lens.max().data[0]

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = Variable(ranges)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)

    return (ranges < lens_exp).float(), (ranges >= lens_exp).float()


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):

        super(Linear, self).__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.initialize()

    def initialize(self):
        I.orthogonal(self.weight.data)

    def zero(self):
        self.weight.data.fill_(0.0)
        self.bias.data.fill_(0.0)


class Highway(nn.Module):

    def __init__(self, size, num_layers, activation='relu'):
        super(Highway, self).__init__()

        self.size = size
        self.num_layers = num_layers
        self.f = getattr(F, activation)
        self.nonlinear = nn.ModuleList(
            [Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList(
            [Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList(
            [Linear(size, size) for _ in range(num_layers)])

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x


class Embedding(nn.Embedding):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2,
                 scale_grad_by_freq=False,
                 sparse=False,
                 trainable=False,
                 padding=0,
                 file=None,
                 stats=False,
                 vocab=None):

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.trainable = trainable
        self.padding = padding
        self.file = file
        self.stats = stats
        self.vocab = vocab

        super(Embedding, self).__init__(num_embeddings + padding,
                                        embedding_dim,
                                        padding_idx,
                                        max_norm,
                                        norm_type,
                                        scale_grad_by_freq,
                                        sparse)
        # self.gpu = False
        self.output_size = embedding_dim
        if not trainable:
            self.weight.requires_grad = False
        if file and vocab:
            self.load(file, vocab, stats=stats)
        else:
            self.initialize()

    def initialize(self):
        I.xavier_normal(self.weight.data)

    # def cuda(self, device=None):
    #     self.gpu = True
    #     if self.allow_gpu:
    #         return super(Embedding, self).cuda(device=None)
    #
    # def cpu(self):
    #     self.gpu = False
    #     return super(Embedding, self).cpu()

    def save(self, path, vocab, stats=False):
        """Save embedding to file.

        :param path: Path to the embedding file.
        :param vocab: Token vocab.
        :param stats: Write stats line (default=False).
        """
        embeds = self.weight.data.cpu().numpy()
        with open(path, 'w', encoding='utf-8') as w:
            if stats:
                embed_num, embed_dim = self.weight.data.size()
                w.write('{} {}\n'.format(embed_num, embed_dim))
            for token, idx in vocab.items():
                embed = ' '.join(map(lambda x: str(x), embeds[idx]))
                w.write('{} {}\n'.format(token, embed))

    def load(self, path, vocab, stats=False):
        logger.info('Loading embedding from {}'.format(path))
        with open(path, 'r', encoding='utf-8') as r:
            if stats:
                r.readline()
            try:
                for line in r:
                    line = line.strip().split(' ')
                    token = line[0]
                    if token in vocab:
                        vector = self.weight.data.new(
                            [float(v) for v in line[1:]])
                        self.weight.data[vocab[token]] = vector
            except UnicodeDecodeError as e:
                print(e)


class LSTM(nn.LSTM):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False,
                 forget_bias=0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.forget_bias = forget_bias
        self.output_size = hidden_size * (2 if bidirectional else 1)

        super(LSTM, self).__init__(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bias=bias,
                                   batch_first=batch_first,
                                   dropout=dropout,
                                   bidirectional=bidirectional)
        self.initialize()

    def initialize(self):
        for n, p in self.named_parameters():
            if 'weight' in n:
                I.orthogonal(p)
            elif 'bias' in n:
                bias_size = p.size(0)
                p.data[bias_size // 4:bias_size // 2].fill_(self.forget_bias)


class CharCNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 dimension,
                 filters
                 ):
        super(CharCNN, self).__init__()

        self.output_size = sum([x[1] for x in filters])
        self.embedding = Embedding(vocab_size,
                                   dimension,
                                   padding_idx=0,
                                   sparse=True,
                                   padding=2)
        self.convs = nn.ModuleList([nn.Conv2d(1, x[1], (x[0], dimension))
                                    for x in filters])

    def forward(self, inputs, lens=None):
        inputs_embed = self.embedding.forward(inputs)
        # input channel
        inputs_embed = inputs_embed.unsqueeze(1)
        # sequeeze output channel
        conv_outputs = [conv.forward(inputs_embed).squeeze(3)
                        for conv in self.convs]
        max_pool_outputs = [F.max_pool1d(i, i.size(2)).squeeze(2)
                            for i in conv_outputs]
        outputs = torch.cat(max_pool_outputs, 1)
        return outputs


class CRF(nn.Module):

    def __init__(self, label_vocab):

        super(CRF, self).__init__()

        self.label_vocab = label_vocab
        self.label_size = len(label_vocab) + 2
        self.start = self.label_size - 2
        self.end = self.label_size - 1
        transition = torch.randn(self.label_size, self.label_size)
        self.transition = nn.Parameter(transition)
        self.initialize()

    def initialize(self):
        self.transition.data[:, self.end] = -100.0
        self.transition.data[self.start, :] = -100.0

    def pad_logits(self, logits, lens):
        lens = lens.data
        batch_size, seq_len, label_num = logits.size()
        pads = Variable(logits.data.new(batch_size, seq_len, 2).fill_(-100.0),
                        requires_grad=False)
        logits = torch.cat([logits, pads], dim=2)
        # e_s = logits.data.new([-100.0] * label_num + [0, 100])
        # e_s_mat = logits.data.new(logits.size()).fill_(0)
        # for i in range(batch_size):
        #     if lens[i] < seq_len:
        #         # logits[i][lens[i]] += e_s
        #         e_s_mat[i][lens[i]] = e_s
        # logits += Variable(e_s_mat)
        return logits

    def calc_binary_score(self, labels, lens):
        batch_size, seq_len = labels.size()

        labels_ext = Variable(labels.data.new(batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=(seq_len + 2)).long()
        pad_stop = Variable(labels.data.new(1).fill_(self.end))
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transition
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr

        return score

    def calc_unary_score(self, logits, labels, lens):
        labels_exp = labels.unsqueeze(-1)
        scores = torch.gather(logits, 2, labels_exp).squeeze(-1)
        mask = sequence_mask(lens).float()
        scores = scores * mask
        return scores

    def calc_gold_score(self, logits, labels, lens):
        unary_score = self.calc_unary_score(logits, labels, lens).sum(
            1).squeeze(-1)
        binary_score = self.calc_binary_score(labels, lens).sum(1).squeeze(-1)
        return unary_score + binary_score

    def calc_norm_score(self, logits, lens):
        batch_size, seq_len, feat_dim = logits.size()
        alpha = logits.data.new(batch_size, self.label_size).fill_(-10000.0)
        alpha[:, self.start] = 0
        alpha = Variable(alpha)
        lens_ = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   *self.transition.size())
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  *self.transition.size())
            trans_exp = self.transition.unsqueeze(0).expand_as(alpha_exp)
            mat = logit_exp + alpha_exp + trans_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (lens_ > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            lens_ = lens_ - 1

        alpha = alpha + self.transition[self.end].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def viterbi_decode(self, logits, lens):
        """Borrowed from pytorch tutorial
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        vit = logits.data.new(batch_size, self.label_size).fill_(-10000)
        vit[:, self.start] = 0
        vit = Variable(vit)
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transition.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transition[self.end].unsqueeze(
                0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths


class LstmCrf(nn.Module):
    def __init__(self,
                 token_vocab,
                 label_vocab,
                 char_vocab,

                 word_embedding,
                 char_embedding,
                 crf,
                 lstm,
                 input_layer=None,
                 univ_layer=None,
                 spec_layer=None,

                 embedding_dropout_prob=0,
                 lstm_dropout_prob=0,
                 linear_dropout_prob=0,
                 use_char_embedding=True,
                 char_highway=None,
                 ):
        super(LstmCrf, self).__init__()

        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab
        self.idx_label = {idx: label for label, idx in label_vocab.items()}
        self.use_char_embedding = use_char_embedding

        self.word_embedding = word_embedding
        self.char_embedding = char_embedding
        self.feat_dim = word_embedding.output_size
        if use_char_embedding:
            self.feat_dim += char_embedding.output_size

        self.lstm = lstm
        self.input_layer = input_layer
        self.univ_layer = univ_layer
        self.spec_layer = spec_layer
        self.crf = crf
        self.char_highway = char_highway
        self.lstm_dropout = nn.Dropout(p=lstm_dropout_prob)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout_prob)
        self.linear_dropout = nn.Dropout(p=linear_dropout_prob)
        self.label_size = len(label_vocab)
        if spec_layer:
            self.spec_gate = Linear(spec_layer.in_features,
                                    spec_layer.out_features)

    def cuda(self, device=None):
        for module in self.children():
            module.cuda(device)
        return self

    def cpu(self):
        for module in self.children():
            module.cpu()
        return self

    def forward_model(self, inputs, lens, chars=None, char_lens=None):
        """From the input to the linear layer, not including the CRF layer.
        
        :param inputs: Input tensor of size batch_size * max_seq_len (word indexes).
        :param lens: Sequence length tensor of size batch_size (sequence lengths).
        :param chars: Input character tensor of size batch_size * max_seq_len * max_word_len (character indexes).
        :param char_lens: Word length tensor of size (batch_size * max_seq_len) * max_word_len.
        :return: Linear layer output tensor of size batch_size * max_seq_len * label_num.  
        """
        batch_size, seq_len = inputs.size()

        # Word embedding
        inputs_embed = self.word_embedding.forward(inputs)
        # Character embedding
        if self.use_char_embedding:
            chars_embed = self.char_embedding.forward(chars, char_lens)
            if self.char_highway:
                chars_embed = self.char_highway.forward(chars_embed)
            chars_embed = chars_embed.view(batch_size, seq_len, -1)
            inputs_embed = torch.cat([inputs_embed, chars_embed], dim=2)
        inputs_embed = self.embedding_dropout.forward(inputs_embed)

        # LSTM layer
        inputs_packed = R.pack_padded_sequence(inputs_embed,
                                               lens.data.tolist(),
                                               batch_first=True)
        lstm_out, _ = self.lstm.forward(inputs_packed)
        lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = lstm_out.contiguous().view(-1, self.lstm.output_size)
        lstm_out = self.lstm_dropout.forward(lstm_out)

        # Linear layer
        univ_feats = self.univ_layer.forward(lstm_out)
        if self.spec_layer:
            spec_feats = self.spec_layer.forward(lstm_out)
            gate = F.sigmoid(self.spec_gate.forward(lstm_out))
            outputs = gate * spec_feats + (1 - gate) * univ_feats
        else:
            outputs = univ_feats
        outputs = outputs.view(batch_size, seq_len, self.label_size)

        return outputs

    def predict(self, inputs, labels, lens, chars=None, char_lens=None):
        """From the input to the CRF output (prediction mode).
        
        :param inputs: Input tensor of size batch_size * max_seq_len (word indexes).
        :param labels: Gold labels.
        :param lens: Sequence length tensor of size batch_size (sequence lengths).
        :param chars: Input character tensor of size batch_size * max_seq_len * max_word_len (character indexes).
        :param char_lens: Word length tensor of size (batch_size * max_seq_len) * max_word_len.
        :return: Prediction and loss.
        """
        self.eval()
        loglik, logits = self.loglik(inputs, labels, lens, chars, char_lens)
        loss = -loglik.mean()
        scores, preds = self.crf.viterbi_decode(logits, lens)
        self.train()
        return preds, loss

    def loglik(self, inputs, labels, lens, chars=None, char_lens=None):
        logits = self.forward_model(inputs, lens, chars, char_lens)
        logits = self.crf.pad_logits(logits, lens)
        norm_score = self.crf.calc_norm_score(logits, lens)
        gold_score = self.crf.calc_gold_score(logits, labels, lens)
        loglik = gold_score - norm_score

        return loglik, logits