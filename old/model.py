import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.utils.rnn as R
import torch.nn.functional as F
import traceback

from util import get_logger, Config
from torch.autograd import Variable

logger = get_logger(__name__)

MODULES = {}


def register_module(name):
    def register(cls):
        if name not in MODULES:
            MODULES[name] = cls
        return cls
    return register


def create_module(name, conf):
    if name in MODULES:
        return MODULES[name](conf)
    else:
        raise ValueError('Module {} is not registered'.format(name))


def log_sum_exp(tensor, dim=0):
    """LogSumExp operation."""
    m, _ = torch.max(tensor, dim)
    m_exp = m.unsqueeze(-1).expand_as(tensor)
    return m + torch.log(torch.sum(torch.exp(tensor - m_exp), dim))


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


@register_module('linear')
class Linear(nn.Linear):

    def __init__(self, conf):
        assert hasattr(conf, 'in_features')
        assert hasattr(conf, 'out_features')

        super(Linear, self).__init__(
            conf.in_features,
            conf.out_features,
            bias=getattr(conf, 'bias', True)
        )
        self.in_features = conf.in_features
        self.out_features = conf.out_features
        self.initialize()

    def initialize(self):
        I.orthogonal(self.weight.data)

    def zero(self):
        self.weight.data.fill_(0.0)
        self.bias.data.fill_(0.0)


@register_module('highway')
class Highway(nn.Module):
    def __init__(self, conf):
        assert hasattr(conf, 'num_layers')
        assert hasattr(conf, 'size')

        super(Highway, self).__init__()
        self.size = size = conf.size
        self.num_layers = num_layers = conf.num_layers
        self.f = getattr(F, getattr(conf, 'activation', 'relu'))
        self.nonlinear = nn.ModuleList([
            Linear(Config({'in_features': size, 'out_features': size}))
            for _ in range(num_layers)])
        # self.linear = nn.ModuleList([
        #     Linear(Config({'in_features': size, 'out_features': size}))
        #     for _ in range(num_layers)])
        self.gate = nn.ModuleList([
            Linear(Config({'in_features': size, 'out_features': size}))
            for _ in range(num_layers)])

    def forward(self, x):
        """
        :param x: tensor with shape of [batch_size, size]
        :return: tensor with shape of [batch_size, size]
        See: https://github.com/kefirski/pytorch_Highway/blob/master/highway/highway.py
        """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))

            # Remove the affine transformation
            # linear = self.linear[layer](x)
            # x = gate * nonlinear + (1 - gate) * linear

            x = gate * nonlinear + (1 - gate) * x

        return x


@register_module('embedding')
class Embedding(nn.Embedding):
    def __init__(self, conf):
        assert hasattr(conf, 'num_embeddings'), 'num_embedding is required'
        assert hasattr(conf, 'embedding_dim'), 'embedding_dim is required'

        self.num_embeddings = num_embeddings = getattr(conf, 'num_embeddings')
        self.embedding_dim = embedding_dim = getattr(conf, 'embedding_dim')
        self.padding_idx = padding_idx = getattr(conf, 'padding_idx', None)
        self.max_norm = max_norm = getattr(conf, 'max_norm', None)
        self.norm_type = norm_type = getattr(conf, 'norm_type', 2)
        self.scale_grad_by_freq = scale_grad_by_freq = getattr(
            conf, 'scale_grad_by_freq', False)
        self.sparse = sparse = getattr(conf, 'sparse', False)
        self.allow_gpu = allow_gpu = getattr(conf, 'allow_gpu', False)
        self.trainable = trainable = getattr(conf, 'trainable', False)
        self.padding = padding = getattr(conf, 'padding', 0)
        self.file = file = getattr(conf, 'file', None)
        self.stats = stats = getattr(conf, 'stats', False)
        self.vocab = vocab = getattr(conf, 'vocab', None)
        self.ignore_case = getattr(conf, 'ignore_case', True)

        super(Embedding, self).__init__(num_embeddings + padding,
                                        embedding_dim,
                                        padding_idx,
                                        max_norm,
                                        norm_type,
                                        scale_grad_by_freq,
                                        sparse)
        self.gpu = False
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
            for line_idx, line in enumerate(r):
                try:
                    segs = line.rstrip().split(' ')
                    # if len(segs) != self.embedding_dim + 1:
                    #     continue
                    token = segs[0]
                    if self.ignore_case:
                        token = token.lower()
                    if token in vocab:
                        vector = self.weight.data.new(
                            [float(v) for v in segs[1:]])
                        self.weight.data[vocab[token]] = vector
                except UnicodeDecodeError as e:
                    traceback.print_exc()
                except ValueError as e:
                    traceback.print_exc()
                    print('line {}'.format(line_idx), line)
                except RuntimeError as e:
                    traceback.print_exc()
                    print('line {}'.format(line_idx), line)


@register_module('lstm')
class LSTM(nn.LSTM):
    def __init__(self, conf):
        assert hasattr(conf, 'input_size'), 'input_size is required'
        assert hasattr(conf, 'hidden_size'), 'hidden_size is required'

        self.input_size = input_size = getattr(conf, 'input_size')
        self.hidden_size = hidden_size = getattr(conf, 'hidden_size')
        self.num_layers = num_layers = getattr(conf, 'num_layers', 1)
        self.bias = bias = getattr(conf, 'bias', True)
        self.batch_first = batch_first = getattr(conf, 'batch_first', False)
        self.dropout = dropout = getattr(conf, 'dropout', 0)
        self.bidirectional = bidirectional = getattr(conf, 'bidirectional',
                                                     False)
        self.forget_bias = forget_bias = getattr(conf, 'forget_bias', 0)
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

    def repack_init_state(self):
        self.init_hidden = (
            Variable(self.init_hidden[0].data, requires_grad=True),
            Variable(self.init_hidden[1].data, requires_grad=True)
        )


@register_module('char_cnn')
class CharCNN(nn.Module):

    def __init__(self, conf):
        assert hasattr(conf, 'vocab_size')
        assert hasattr(conf, 'dimension')
        assert hasattr(conf, 'filters')

        super(CharCNN, self).__init__()

        vocab_size = getattr(conf, 'vocab_size')
        dimension = getattr(conf, 'dimension')
        filters = getattr(conf, 'filters')
        padding = getattr(conf, 'padding', 2)

        self.output_size = sum([x[1] for x in filters])
        self.embedding = Embedding(Config({
            'num_embeddings': vocab_size,
            'embedding_dim': dimension,
            'padding_idx': 0,
            'sparse': True,
            'allow_gpu': True,
            'padding': padding
        }))
        self.convs = nn.ModuleList([nn.Conv2d(1, x[1], (x[0], dimension))
                                    for x in filters])

    def forward(self, inputs, lens=None):
        # batch_size, seq_len = inputs.size()
        inputs_embed = self.embedding.forward(inputs)
        # input channel
        inputs_embed = inputs_embed.unsqueeze(1)
        # sequeeze output channel
        conv_outputs = [F.relu(conv.forward(inputs_embed)).squeeze(3)
                        for conv in self.convs]
        max_pool_outputs = [F.max_pool1d(i, i.size(2)).squeeze(2)
                            for i in conv_outputs]
        outputs = torch.cat(max_pool_outputs, 1)
        return outputs


@register_module('crf')
class CRF(nn.Module):
    def __init__(self, conf):
        assert hasattr(conf, 'label_vocab')

        super(CRF, self).__init__()

        self.label_vocab = label_vocab = getattr(conf, 'label_vocab')
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


class Model(nn.Module):
    """Override the cuda function.
    Used for top-level modules containing other modules.
    It only works for modules without direct parameters."""

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


@register_module('lstm_crf')
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
                 linear_dropout_prob=0,
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

        self.feat_dim = word_embedding.output_size
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
        self.linear_dropout = nn.Dropout(p=linear_dropout_prob)
        self.label_size = len(label_vocab)
        if spec_fc_layer:
            self.spec_gate = Linear(Config({
                'in_features': spec_fc_layer.in_features,
                'out_features': spec_fc_layer.out_features
            }))

    def forward_model(self, inputs, lens, chars=None, char_lens=None):
        """
        From input to linear layer.

        :param inputs:
        :param lens:
        :param chars:
        :param char_lens:
        :return:
        """
        batch_size, seq_len = inputs.size()

        # Word embedding
        inputs_embed = self.word_embedding.forward(inputs)
        if self.gpu:
            inputs_embed = inputs_embed.cuda()

        # Character embedding
        if self.use_char_embedding:
            chars_embed = self.char_embedding.forward(chars, char_lens)
            if self.char_highway:
                chars_embed = self.char_highway.forward(chars_embed)
            chars_embed = chars_embed.view(batch_size, seq_len, -1)
            inputs_embed = torch.cat([inputs_embed, chars_embed], dim=2)

        inputs_embed = self.embed_dropout.forward(inputs_embed)

        # LSTM layer
        inputs_packed = R.pack_padded_sequence(inputs_embed, lens.data.tolist(),
                                               batch_first=True)
        lstm_out, _ = self.lstm(inputs_packed)
        lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)

        lstm_out = lstm_out.contiguous().view(-1, self.lstm.output_size)
        lstm_out = self.lstm_dropout.forward(lstm_out)

        # Fully-connected layer
        univ_feats = self.univ_fc_layer.forward(lstm_out)
        if self.spec_fc_layer is not None:
            spec_feats = self.spec_fc_layer.forward(lstm_out)
            gate = F.sigmoid(self.spec_gate.forward(lstm_out))
            outputs = gate * spec_feats + (1 - gate) * univ_feats
        else:
            outputs = univ_feats
        outputs = outputs.view(batch_size, seq_len, self.label_size)
        # outputs = self.linear_dropout.forward(outputs)

        return outputs

    def predict(self, inputs, labels, lens, chars=None, char_lens=None):
        self.eval()
        # for module in self.children():
        #     module.eval()

        loglik, logits = self.loglik(inputs, labels, lens, chars, char_lens)
        loss = -loglik.mean()
        scores, preds = self.crf.viterbi_decode(logits, lens)

        self.train()
        # for module in self.children():
        #     module.train()
        return preds, loss

    def loglik(self, inputs, labels, lens, chars=None, char_lens=None):
        logits = self.forward_model(inputs, lens, chars, char_lens)
        logits = self.crf.pad_logits(logits, lens)
        norm_score = self.crf.calc_norm_score(logits, lens)
        gold_score = self.crf.calc_gold_score(logits, labels, lens)
        loglik = gold_score - norm_score

        return loglik, logits
