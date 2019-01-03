from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.utils.rnn as R
import torch.nn.functional as F

import re
import logging
import constant as C

logger = logging.getLogger()


def log_sum_exp(tensor, dim=0):
    """LogSumExp operation."""
    m, _ = torch.max(tensor, dim)
    m_exp = m.unsqueeze(-1).expand_as(tensor)
    return m + torch.log(torch.sum(torch.exp(tensor - m_exp), dim))


def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().item()

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask


def load_embedding(path: str,
                   dimension: int,
                   vocab: dict = None,
                   skip_first_line: bool = True,
                   ):
    logger.info('Scanning embedding file: {}'.format(path))

    embed_vocab = set()
    lower_mapping = {}  # lower case - original
    digit_mapping = {}  # lower case + replace digit with 0 - original
    digit_pattern = re.compile('\d')
    with open(path, 'r', encoding='utf-8') as r:
        if skip_first_line:
            r.readline()
        for line in r:
            try:
                token = line.split(' ')[0].strip()
                if token:
                    embed_vocab.add(token)
                    token_lower = token.lower()
                    token_digit = re.sub(digit_pattern, '0', token_lower)
                    if token_lower not in lower_mapping:
                        lower_mapping[token_lower] = token
                    if token_digit not in digit_mapping:
                        digit_mapping[token_digit] = token
            except UnicodeDecodeError:
                continue

    token_mapping = defaultdict(list)  # embed token - vocab token
    for token in vocab:
        token_lower = token.lower()
        token_digit = re.sub(digit_pattern, '0', token_lower)
        if token in embed_vocab:
            token_mapping[token].append(token)
        elif token_lower in lower_mapping:
            token_mapping[lower_mapping[token_lower]].append(token)
        elif token_digit in digit_mapping:
            token_mapping[digit_mapping[token_digit]].append(token)

    logger.info('Loading embeddings')
    weight = [[.0] * dimension for _ in range(len(vocab))]
    with open(path, 'r', encoding='utf-8') as r:
        if skip_first_line:
            r.readline()
        for line in r:
            try:
                segs = line.rstrip().split(' ')
                token = segs[0]
                if token in token_mapping:
                    vec = [float(v) for v in segs[1:]]
                    for t in token_mapping.get(token):
                        weight[vocab[t]] = vec.copy()
            except UnicodeDecodeError:
                continue
            except ValueError:
                continue
    embed = nn.Embedding(
        len(vocab),
        dimension,
        padding_idx=C.PAD_INDEX,
        sparse=True,
        _weight=torch.FloatTensor(weight)
    )
    return embed


class Linear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)
        I.orthogonal_(self.weight)


class Linears(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hiddens: list,
                 bias: bool = True,
                 activation: str = 'tanh'):
        super(Linears, self).__init__()
        assert len(hiddens) > 0

        self.in_features = in_features
        self.out_features = self.output_size = out_features

        in_dims = [in_features] + hiddens[:-1]
        self.linears = nn.ModuleList([Linear(in_dim, out_dim, bias=bias)
                                      for in_dim, out_dim
                                      in zip(in_dims, hiddens)])
        self.output_linear = Linear(hiddens[-1], out_features, bias=bias)
        self.activation = getattr(F, activation)

    def forward(self, inputs):
        linear_outputs = inputs
        for linear in self.linears:
            linear_outputs = linear(linear_outputs)
            linear_outputs = self.activation(linear_outputs)
        return self.output_linear(linear_outputs)


class Highway(nn.Module):
    def __init__(self,
                 size: int,
                 layer_num: int = 1,
                 activation: str = 'relu'):
        super(Highway, self).__init__()
        self.size = self.output_size = size
        self.layer_num = layer_num
        self.activation = getattr(F, activation)
        self.non_linear = nn.ModuleList([Linear(size, size)
                                         for _ in range(layer_num)])
        self.gate = nn.ModuleList([Linear(size, size)
                                   for _ in range(layer_num)])

    def forward(self, inputs):
        for layer in range(self.layer_num):
            gate = F.sigmoid(self.gate[layer](inputs))
            non_linear = self.activation(self.non_linear[layer](inputs))
            inputs = gate * non_linear + (1 - gate) * inputs
        return inputs


class LSTM(nn.LSTM):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 batch_first: bool = False,
                 dropout: float = 0,
                 bidirectional: bool = False,
                 forget_bias: float = 0
                 ):
        super(LSTM, self).__init__(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bias=bias,
                                   batch_first=batch_first,
                                   dropout=dropout,
                                   bidirectional=bidirectional)
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.forget_bias = forget_bias

    def initialize(self):
        for n, p in self.named_parameters():
            if 'weight' in n:
                I.orthogonal_(p)
            elif 'bias' in n:
                bias_size = p.size(0)
                p[bias_size // 4:bias_size // 2].fill_(self.forget_bias)


class CharCNN(nn.Module):

    def __init__(self, embedding_num, embedding_dim, filters):
        super(CharCNN, self).__init__()
        self.output_size = sum([x[1] for x in filters])
        self.embedding = nn.Embedding(embedding_num,
                                      embedding_dim,
                                      padding_idx=0,
                                      sparse=True)
        self.convs = nn.ModuleList([nn.Conv2d(1, x[1], (x[0], embedding_dim))
                                    for x in filters])

    def forward(self, inputs):
        inputs_embed = self.embedding(inputs)
        inputs_embed = inputs_embed.unsqueeze(1)
        conv_outputs = [F.relu(conv(inputs_embed)).squeeze(3)
                        for conv in self.convs]
        max_pool_outputs = [F.max_pool1d(i, i.size(2)).squeeze(2)
                            for i in conv_outputs]
        outputs = torch.cat(max_pool_outputs, 1)
        return outputs


class CRF(nn.Module):
    def __init__(self, label_size):
        super(CRF, self).__init__()

        self.label_size = label_size
        self.start = self.label_size - 2
        self.end = self.label_size - 1
        transition = torch.randn(self.label_size, self.label_size)
        self.transition = nn.Parameter(transition)
        self.initialize()

    def initialize(self):
        self.transition.data[:, self.end] = -100.0
        self.transition.data[self.start, :] = -100.0

    def pad_logits(self, logits):
        # lens = lens.data
        batch_size, seq_len, label_num = logits.size()
        # pads = Variable(logits.data.new(batch_size, seq_len, 2).fill_(-1000.0),
        #                 requires_grad=False)
        pads = logits.new_full((batch_size, seq_len, 2), -1000.0,
                               requires_grad=False)
        logits = torch.cat([logits, pads], dim=2)
        return logits

    def calc_binary_score(self, labels, lens):
        batch_size, seq_len = labels.size()

        # labels_ext = Variable(labels.data.new(batch_size, seq_len + 2))
        labels_ext = labels.new_empty((batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=(seq_len + 2)).long()
        # pad_stop = Variable(labels.data.new(1).fill_(self.end))
        pad_stop = labels.new_full((1,), self.end, requires_grad=False)
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
        # alpha = logits.data.new(batch_size, self.label_size).fill_(-10000.0)
        alpha = logits.new_full((batch_size, self.label_size), -100.0)
        alpha[:, self.start] = 0
        # alpha = Variable(alpha)
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
        # vit = logits.data.new(batch_size, self.label_size).fill_(-10000)
        vit = logits.new_full((batch_size, self.label_size), -100.0)
        vit[:, self.start] = 0
        # vit = Variable(vit)
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
        # idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gpu = False

    def cuda(self, device=None):
        self.gpu = True
        for module in self.children():
            module.cuda(device)
        return self

    def cpu(self):
        self.gpu = False
        for module in self.children():
            module.cpu()
        return self


class LstmCrf(Model):
    def __init__(self,
                 token_vocab,
                 label_vocab,
                 char_vocab,
                 word_embedding,
                 char_embedding,
                 crf,
                 lstm,
                 input_layer=None,
                 univ_fc_layer=None,
                 spec_fc_layer=None,
                 output_layer=None,
                 embed_dropout_prob=0,
                 lstm_dropout_prob=0,
                 use_char_embedding=True,
                 char_highway=None
                 ):
        super(LstmCrf, self).__init__()

        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab
        self.idx_label = {idx: label for label, idx in label_vocab.items()}
        self.embed_dropout_prob = embed_dropout_prob
        self.lstm_dropout_prob = lstm_dropout_prob
        self.use_char_embedding = use_char_embedding

        self.word_embedding = word_embedding
        self.char_embedding = char_embedding

        self.feat_dim = word_embedding.embedding_dim
        if use_char_embedding:
            self.feat_dim += char_embedding.output_size

        self.lstm = lstm
        self.input_layer = input_layer
        self.univ_fc_layer = univ_fc_layer
        self.spec_fc_layer = spec_fc_layer
        self.output_layer = output_layer
        self.crf = crf
        self.char_highway = char_highway
        self.lstm_dropout = nn.Dropout(p=lstm_dropout_prob)
        self.embed_dropout = nn.Dropout(p=embed_dropout_prob)
        self.label_size = len(label_vocab)
        if spec_fc_layer:
            self.spec_gate = Linear(spec_fc_layer.in_features,
                                    spec_fc_layer.out_features)

    def forward_model(self, inputs, lens, chars=None, char_lens=None):
        batch_size, seq_len = inputs.size()

        # Word embedding
        inputs_embed = self.word_embedding(inputs)

        # Character embedding
        if self.use_char_embedding:
            chars_embed = self.char_embedding(chars)
            if self.char_highway:
                chars_embed = self.char_highway(chars_embed)
            chars_embed = chars_embed.view(batch_size, seq_len, -1)
            inputs_embed = torch.cat([inputs_embed, chars_embed], dim=2)

        inputs_embed = self.embed_dropout(inputs_embed)

        # LSTM layer
        inputs_packed = R.pack_padded_sequence(inputs_embed, lens.tolist(),
                                               batch_first=True)
        lstm_out, _ = self.lstm(inputs_packed)
        lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)

        lstm_out = lstm_out.contiguous().view(-1, self.lstm.output_size)
        lstm_out = self.lstm_dropout(lstm_out)

        # Fully-connected layer
        univ_feats = self.univ_fc_layer(lstm_out)
        if self.spec_fc_layer is not None:
            spec_feats = self.spec_fc_layer(lstm_out)
            gate = F.sigmoid(self.spec_gate(lstm_out))
            outputs = gate * spec_feats + (1 - gate) * univ_feats
        else:
            outputs = univ_feats
        outputs = outputs.view(batch_size, seq_len, self.label_size)

        return outputs

    def predict(self, inputs, labels, lens, chars=None, char_lens=None):
        self.eval()

        loglik, logits = self.loglik(inputs, labels, lens, chars, char_lens)
        loss = -loglik.mean()
        scores, preds = self.crf.viterbi_decode(logits, lens)

        self.train()
        return preds, loss

    def loglik(self, inputs, labels, lens, chars=None, char_lens=None):
        logits = self.forward_model(inputs, lens, chars, char_lens)
        logits = self.crf.pad_logits(logits)
        norm_score = self.crf.calc_norm_score(logits, lens)
        gold_score = self.crf.calc_gold_score(logits, labels, lens)
        loglik = gold_score - norm_score

        return loglik, logits
