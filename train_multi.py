"""
This script demonstrates how to train a multi-lingual multi-task model with four
tasks:
- Target task
- Auxiliary task 1 (different language)
- Auxiliary task 2 (different task)
- Auxiliary task 3 (different language and task)
For example, if the target task is Spanish Name Tagging, related task Part-of-
speech Tagging, and related language English, task 1 is English Name Tagging,
task 2 is Spanish Part-of-speech Tagging, and task 3 is English Part-of-speech
Tagging.
"""

import math
import os
from tqdm import tqdm
import time
import logging
import traceback
from collections import Counter

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

import constant as C
from random import shuffle
from argparse import ArgumentParser

from torch.utils.data import DataLoader

from util import evaluate
from data import ConllParser, SeqLabelDataset, SeqLabelProcessor, count2vocab
from model import Linear, LSTM, CRF, CharCNN, Highway, LstmCrf, load_embedding

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

argparser = ArgumentParser()

# Target
argparser.add_argument('--train_tgt', help='Path to the training set file')
argparser.add_argument('--dev_tgt', help='Path to the dev set file')
argparser.add_argument('--test_tgt', help='Path to the test set file')
# Cross-lingual: same task, diff languages
argparser.add_argument('--train_cl', help='Path to the training set file')
argparser.add_argument('--dev_cl', help='Path to the dev set file')
argparser.add_argument('--test_cl', help='Path to the test set file')
# Cross-task: same languages, diff tasks
argparser.add_argument('--train_ct', help='Path to the training set file')
argparser.add_argument('--dev_ct', help='Path to the dev set file')
argparser.add_argument('--test_ct', help='Path to the test set file')
# Cross-lingual Cross-task, diff languages and tasks
argparser.add_argument('--train_clct', help='Path to the training set file')
argparser.add_argument('--dev_clct', help='Path to the dev set file')
argparser.add_argument('--test_clct', help='Path to the test set file')

argparser.add_argument('--log', help='Path to the log dir')
argparser.add_argument('--model', help='Path to the model file')
argparser.add_argument('--batch_size', default=10, type=int, help='Batch size')
argparser.add_argument('--max_epoch', default=100, type=int)
argparser.add_argument('--word_embed_1',
                       help='Path to the pre-trained embedding file for lang 1')
argparser.add_argument('--word_embed_2',
                       help='Path to the pre-trained embedding file for lang 2')
argparser.add_argument('--word_embed_dim', type=int, default=100,
                       help='Word embedding dimension')
argparser.set_defaults(word_ignore_case=False)
argparser.add_argument('--char_embed_dim', type=int, default=50,
                       help='Character embedding dimension')
argparser.add_argument('--charcnn_filters', default='2,25;3,25;4,25',
                       help='Character-level CNN filters')
argparser.add_argument('--charhw_layer', default=1, type=int)
argparser.add_argument('--charhw_func', default='relu')
argparser.add_argument('--use_highway', action='store_true')
argparser.add_argument('--lstm_hidden_size', default=100, type=int,
                       help='LSTM hidden state size')
argparser.add_argument('--lstm_forget_bias', default=0, type=float,
                       help='LSTM forget bias')
argparser.add_argument('--feat_dropout', default=.5, type=float,
                       help='Word feature dropout probability')
argparser.add_argument('--lstm_dropout', default=.5, type=float,
                       help='LSTM output dropout probability')
argparser.add_argument('--lr', default=0.005, type=float,
                       help='Learning rate')
argparser.add_argument('--momentum', default=.9, type=float)
argparser.add_argument('--decay_rate', default=.9, type=float)
argparser.add_argument('--decay_step', default=10000, type=int)
argparser.add_argument('--grad_clipping', default=5, type=float)
argparser.add_argument('--gpu', action='store_true')
argparser.add_argument('--device', default=0, type=int)
argparser.add_argument('--thread', default=5, type=int)

args = argparser.parse_args()
batch_size = args.batch_size

use_gpu = args.gpu and torch.cuda.is_available()
if use_gpu:
    torch.cuda.set_device(args.device)
torch.set_num_threads(args.thread)

# Model file
model_dir = args.model
assert model_dir and os.path.isdir(model_dir), 'Model output dir is required'
model_file = os.path.join(model_dir, 'model.{}.mdl'.format(timestamp))

# Logging file
log_writer = None
if args.log:
    log_file = os.path.join(args.log, 'log.{}.txt'.format(timestamp))
    log_writer = open(log_file, 'a', encoding='utf-8')
    logger.addHandler(logging.FileHandler(log_file, encoding='utf-8'))
logger.info('----------')
logger.info('Parameters:')
for arg in vars(args):
    logger.info('{}: {}'.format(arg, getattr(args, arg)))
logger.info('----------')

# Data file
logger.info('Loading data sets')
ner_parser = ConllParser(skip_comment=True)
pos_parser = ConllParser(token_col=1, label_col=3, skip_comment=True)

train_set_tgt = SeqLabelDataset(args.train_tgt, parser=ner_parser)
dev_set_tgt = SeqLabelDataset(args.dev_tgt, parser=ner_parser)
test_set_tgt = SeqLabelDataset(args.test_tgt, parser=ner_parser)

train_set_cl = SeqLabelDataset(args.train_cl, parser=ner_parser)
dev_set_cl = SeqLabelDataset(args.dev_cl, parser=ner_parser)
test_set_cl = SeqLabelDataset(args.test_cl, parser=ner_parser)

train_set_ct = SeqLabelDataset(args.train_ct, parser=pos_parser)
dev_set_ct = SeqLabelDataset(args.dev_ct, parser=pos_parser)
test_set_ct = SeqLabelDataset(args.test_ct, parser=pos_parser)

train_set_clct = SeqLabelDataset(args.train_clct, parser=pos_parser)
dev_set_clct = SeqLabelDataset(args.dev_clct, parser=pos_parser)
test_set_clct = SeqLabelDataset(args.test_clct, parser=pos_parser)

# datasets = {'train': train_set, 'dev': dev_set, 'test': test_set}
datasets = {
    'tgt': {'train': train_set_tgt, 'dev': dev_set_tgt, 'test': test_set_tgt},
    'cl': {'train': train_set_cl, 'dev': dev_set_cl, 'test': test_set_cl},
    'ct': {'train': train_set_ct, 'dev': dev_set_ct, 'test': test_set_ct},
    'clct': {'train': train_set_clct, 'dev': dev_set_clct, 'test': test_set_clct}
}

# Vocabs
logger.info('Building vocabs')
(
    token_count_1, token_count_2, char_count, label_count_1, label_count_2
) = Counter(), Counter(), Counter(), Counter(), Counter()
for _, ds in datasets['tgt'].items():
    tc, cc, lc = ds.stats()
    token_count_1.update(tc)
    char_count.update(cc)
    label_count_1.update(lc)
for _, ds in datasets['cl'].items():
    tc, cc, lc = ds.stats()
    token_count_2.update(tc)
    char_count.update(cc)
    label_count_1.update(lc)
for _, ds in datasets['ct'].items():
    tc, cc, lc = ds.stats()
    token_count_1.update(tc)
    char_count.update(cc)
    label_count_2.update(lc)
for _, ds in datasets['clct'].items():
    tc, cc, lc = ds.stats()
    token_count_2.update(tc)
    char_count.update(cc)
    label_count_2.update(lc)
token_vocab_1 = count2vocab(token_count_1, offset=len(C.TOKEN_PADS), pads=C.TOKEN_PADS)
token_vocab_2 = count2vocab(token_count_2, offset=len(C.TOKEN_PADS), pads=C.TOKEN_PADS)
char_vocab = count2vocab(char_count, offset=len(C.CHAR_PADS), pads=C.CHAR_PADS)
label_vocab_1 = count2vocab(label_count_1, pads=[(C.PAD, C.PAD_INDEX)])
label_vocab_2 = count2vocab(label_count_2, pads=[(C.PAD, C.PAD_INDEX)])

idx_token_1 = {idx: token for token, idx in token_vocab_1.items()}
idx_token_2 = {idx: token for token, idx in token_vocab_2.items()}
idx_label_1 = {idx: label for label, idx in label_vocab_1.items()}
idx_label_2 = {idx: label for label, idx in label_vocab_2.items()}
idx_tokens = {
    'tgt': idx_token_1,
    'cl': idx_token_2,
    'ct': idx_token_1,
    'clct': idx_token_2
}
idx_labels = {
    'tgt': idx_label_1,
    'cl': idx_label_1,
    'ct': idx_label_2,
    'clct': idx_label_2
}

train_set_tgt.numberize(token_vocab_1, label_vocab_1, char_vocab)
dev_set_tgt.numberize(token_vocab_1, label_vocab_1, char_vocab)
test_set_tgt.numberize(token_vocab_1, label_vocab_1, char_vocab)

train_set_cl.numberize(token_vocab_2, label_vocab_1, char_vocab)
dev_set_cl.numberize(token_vocab_2, label_vocab_1, char_vocab)
test_set_cl.numberize(token_vocab_2, label_vocab_1, char_vocab)

train_set_ct.numberize(token_vocab_1, label_vocab_2, char_vocab)
dev_set_ct.numberize(token_vocab_1, label_vocab_2, char_vocab)
test_set_ct.numberize(token_vocab_1, label_vocab_2, char_vocab)

train_set_clct.numberize(token_vocab_2, label_vocab_2, char_vocab)
dev_set_clct.numberize(token_vocab_2, label_vocab_2, char_vocab)
test_set_clct.numberize(token_vocab_2, label_vocab_2, char_vocab)

# Embedding file
word_embed_1 = load_embedding(args.word_embed_1,
                              dimension=args.word_embed_dim,
                              vocab=token_vocab_1)
word_embed_2 = load_embedding(args.word_embed_2,
                              dimension=args.word_embed_dim,
                              vocab=token_vocab_2)
charcnn_filters = [[int(f.split(',')[0]), int(f.split(',')[1])]
                   for f in args.charcnn_filters.split(';')]
char_embed = CharCNN(len(char_vocab),
                     args.char_embed_dim,
                     filters=charcnn_filters)
char_hw = Highway(char_embed.output_size,
                  layer_num=args.charhw_layer,
                  activation=args.charhw_func)
feat_dim = args.word_embed_dim + char_embed.output_size
lstm = LSTM(feat_dim,
            args.lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
            forget_bias=args.lstm_forget_bias
            )
crf_1 = CRF(label_size=len(label_vocab_1) + 2)
crf_2 = CRF(label_size=len(label_vocab_2) + 2)
# Linear layers for task 1
shared_linear_1 = Linear(in_features=lstm.output_size,
                         out_features=len(label_vocab_1))
spec_linear_1_1 = Linear(in_features=lstm.output_size,
                         out_features=len(label_vocab_1))
spec_linear_1_2 = Linear(in_features=lstm.output_size,
                         out_features=len(label_vocab_1))
# Linear layers for task 2
shared_linear_2 = Linear(in_features=lstm.output_size,
                         out_features=len(label_vocab_2))
spec_linear_2_1 = Linear(in_features=lstm.output_size,
                         out_features=len(label_vocab_2))
spec_linear_2_2 = Linear(in_features=lstm.output_size,
                         out_features=len(label_vocab_2))

lstm_crf_tgt = LstmCrf(
    token_vocab_1, label_vocab_1, char_vocab,
    word_embedding=word_embed_1,
    char_embedding=char_embed,
    crf=crf_1,
    lstm=lstm,
    univ_fc_layer=shared_linear_1,
    spec_fc_layer=spec_linear_1_1,
    embed_dropout_prob=args.feat_dropout,
    lstm_dropout_prob=args.lstm_dropout,
    char_highway=char_hw if args.use_highway else None
)
lstm_crf_cl = LstmCrf(
    token_vocab_2, label_vocab_1, char_vocab,
    word_embedding=word_embed_2,
    char_embedding=char_embed,
    crf=crf_1,
    lstm=lstm,
    univ_fc_layer=shared_linear_1,
    spec_fc_layer=spec_linear_1_2,
    embed_dropout_prob=args.feat_dropout,
    lstm_dropout_prob=args.lstm_dropout,
    char_highway=char_hw if args.use_highway else None
)
lstm_crf_ct = LstmCrf(
    token_vocab_1, label_vocab_2, char_vocab,
    word_embedding=word_embed_1,
    char_embedding=char_embed,
    crf=crf_2,
    lstm=lstm,
    univ_fc_layer=shared_linear_2,
    spec_fc_layer=spec_linear_2_1,
    embed_dropout_prob=args.feat_dropout,
    lstm_dropout_prob=args.lstm_dropout,
    char_highway=char_hw if args.use_highway else None
)
lstm_crf_clct = LstmCrf(
    token_vocab_2, label_vocab_2, char_vocab,
    word_embedding=word_embed_2,
    char_embedding=char_embed,
    crf=crf_2,
    lstm=lstm,
    univ_fc_layer=shared_linear_2,
    spec_fc_layer=spec_linear_2_2,
    embed_dropout_prob=args.feat_dropout,
    lstm_dropout_prob=args.lstm_dropout,
    char_highway=char_hw if args.use_highway else None
)
if use_gpu:
    lstm_crf_tgt.cuda()
    lstm_crf_cl.cuda()
    lstm_crf_ct.cuda()
    lstm_crf_clct.cuda()
models = {
    'tgt': lstm_crf_tgt,
    'cl': lstm_crf_cl,
    'ct': lstm_crf_ct,
    'clct': lstm_crf_clct
}

# Task
optimizer_tgt = optim.SGD(
    filter(lambda p: p.requires_grad, lstm_crf_tgt.parameters()),
    lr=args.lr, momentum=args.momentum)
optimizer_cl = optim.SGD(
    filter(lambda p: p.requires_grad, lstm_crf_cl.parameters()),
    lr=args.lr, momentum=args.momentum)
optimizer_ct = optim.SGD(
    filter(lambda p: p.requires_grad, lstm_crf_ct.parameters()),
    lr=args.lr, momentum=args.momentum)
optimizer_clct = optim.SGD(
    filter(lambda p: p.requires_grad, lstm_crf_clct.parameters()),
    lr=args.lr, momentum=args.momentum)
optimizers = {
    'tgt': optimizer_tgt,
    'cl': optimizer_cl,
    'ct': optimizer_ct,
    'clct': optimizer_clct
}
processor = SeqLabelProcessor(gpu=use_gpu)

train_args = vars(args)
train_args['word_embed_size'] = word_embed_1.num_embeddings
state = {
    'model': {
        'word_embed': word_embed_1.state_dict(),
        'char_embed': char_embed.state_dict(),
        'char_hw': char_hw.state_dict(),
        'lstm': lstm.state_dict(),
        'crf': crf_1.state_dict(),
        'univ_linear': shared_linear_1.state_dict(),
        'spec_linear': spec_linear_1_1.state_dict(),
        'lstm_crf': lstm_crf_tgt.state_dict()
    },
    'args': train_args,
    'vocab': {
        'token': token_vocab_1,
        'label': label_vocab_1,
        'char': char_vocab,
    }
}

# Calculate mixing rates
batch_num = len(train_set_tgt) // batch_size
r_tgt = math.sqrt(len(train_set_tgt))
r_cl = 1.0 * .1 * math.sqrt(len(datasets['cl']['train']))
r_ct = .1 * 1.0 * math.sqrt(len(datasets['ct']['train']))
r_clct = .1 * .1 * math.sqrt(len(datasets['clct']['train']))
num_cl = int(r_cl / r_tgt * batch_num)
num_ct = int(r_ct / r_tgt * batch_num)
num_clct = int(r_clct / r_tgt * batch_num)
print('{}, {}, {}, {}'.format(batch_num, num_cl, num_ct, num_clct))

data_loaders = {}
data_loader_iters = {}
for task, task_datasets in datasets.items():
    data_loaders[task] = {
        k: DataLoader(v,
                      batch_size=batch_size,
                      shuffle=k == 'train',
                      collate_fn=processor.process)
        for k, v in task_datasets.items()
    }
    data_loader_iters[task] = {k: iter(v) for k, v
                               in data_loaders[task].items()}

try:
    global_step = 0
    best_dev_score = best_test_score = 0.0

    for epoch in range(args.max_epoch):
        logger.info('Epoch {}: Training'.format(epoch + 1))
        best = False

        for ds in ['train', 'dev', 'test']:
            if ds == 'train':
                tasks = ['tgt'] * batch_num + ['cl'] * num_cl\
                          + ['ct'] * num_ct + ['clct'] * num_clct
                shuffle(tasks)
                progress = tqdm(total=len(tasks), mininterval=1,
                                desc=ds)
                for task in tasks:
                    progress.update(1)
                    global_step += 1
                    model = models[task]
                    optimizer = optimizers[task]
                    optimizer.zero_grad()
                    try:
                        batch = next(data_loader_iters[task]['train'])
                    except StopIteration:
                        data_loader_iters[task]['train'] = iter(
                            data_loaders[task]['train']
                        )
                        batch = next(data_loader_iters[task]['train'])
                    tokens, labels, chars, seq_lens, char_lens = batch
                    loglik, _ = model.loglik(
                        tokens, labels, seq_lens, chars, char_lens)
                    loss = -loglik.mean()
                    loss.backward()

                    clip_grad_norm_(model.parameters(), args.grad_clipping)
                    optimizer.step()
                progress.close()
            else:
                for task in ['tgt', 'cl', 'ct', 'clct']:
                    logger.info('task: {} dataset: {}'.format(task, ds))
                    results = []
                    for batch in data_loaders[task][ds]:
                        tokens, labels, chars, seq_lens, char_lens = batch
                        pred, _loss = models[task].predict(
                            tokens, labels, seq_lens, chars, char_lens
                        )
                        results.append((pred, labels, seq_lens, tokens))
                    fscore, prec, rec = evaluate(
                        results, idx_tokens[task], idx_labels[task],
                        writer=log_writer)
                    if ds == 'dev' and task == 'tgt' and fscore > best_dev_score:
                        logger.info('New best score: {:.4f}'.format(fscore))
                        best_dev_score = fscore
                        best = True
                        logger.info('Saving the model to {}'.format(model_file))
                        torch.save(state, model_file)
                    if best and ds == 'test' and task == 'tgt':
                        best_test_score = fscore

        # learning rate decay
        lr = args.lr * args.decay_rate ** (global_step / args.decay_step)
        for opt in optimizers.values():
            for p in opt.param_groups:
                p['lr'] = lr
        logger.info('New learning rate: {}'.format(lr))

    logger.info('Best dev score: {}'.format(best_dev_score))
    logger.info('Best test score: {}'.format(best_test_score))
    logger.info('Model file: {}'.format(model_file))
    if args.log:
        logger.info('Log file: {}'.format(log_file))
        log_writer.close()
except Exception:
    traceback.print_exc()
    if log_writer:
        log_writer.close()