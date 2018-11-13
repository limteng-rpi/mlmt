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
import time
import traceback

import tqdm

import torch
from random import shuffle
from torch import optim
from torch.nn.utils import clip_grad_norm

import constant as C
from model import Linear, LSTM, CRF, CharCNN, Highway, LstmCrf, Embedding
from argparse import ArgumentParser
from util import get_logger, evaluate, Config
from data import (
    SequenceDataset, ConllParser,
    compute_metadata, count2vocab, numberize_datasets
)

timestamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())

argparser = ArgumentParser()

argparser.add_argument('--train_tgt',
                       help='Path to the training set file of the target task')
argparser.add_argument('--dev_tgt',
                       help='Path to the dev set file of the target task')

argparser.add_argument('--train_cl',
                       help='Path to the training set file of auxiliary task 1 (same task, different language)')
argparser.add_argument('--dev_cl',
                       help='Path to the dev set file of auxiliary task 1 (same task, different language)')

argparser.add_argument('--log', help='Path to the log dir')
argparser.add_argument('--model', help='Path to the model file')
argparser.add_argument('--batch_size', default=10, type=int, help='Batch size')
argparser.add_argument('--max_epoch', default=100, type=int)
argparser.add_argument('--embedding1',
                       help='Path to the pre-trained embedding file for language 1')
argparser.add_argument('--embedding2',
                       help='Path to the pre-trained embedding file for language 2')
argparser.add_argument('--embed_skip_first', dest='embed_skip_first',
                       action='store_true',
                       help='Skip the first line of the embedding file')
argparser.set_defaults(embed_skip_first=True)
argparser.add_argument('--word_embed_dim', type=int, default=100,
                       help='Word embedding dimension')
argparser.add_argument('--word_ignore_case', dest='word_ignore_case',
                       action='store_true')
argparser.set_defaults(word_ignore_case=False)
argparser.add_argument('--char_embed_dim', type=int, default=50,
                       help='Character embedding dimension')
argparser.add_argument('--charcnn_filters', default='2,25;3,25;4,25',
                       help='Character-level CNN filters')
argparser.add_argument('--lstm_hidden_size', default=100, type=int,
                       help='LSTM hidden state size')
argparser.add_argument('--embed_dropout', default=.2, type=float,
                       help='Embedding dropout probability')
argparser.add_argument('--lstm_dropout', default=.5, type=float,
                       help='LSTM output dropout probability')
argparser.add_argument('--linear_dropout', default=0, type=float,
                       help='Output linear layer dropout probability')
argparser.add_argument('--lr', default=0.005, type=float,
                       help='Learning rate')
argparser.add_argument('--momentum', default=.9, type=float)
argparser.add_argument('--decay_rate', default=.9, type=float)
argparser.add_argument('--decay_step', default=10000, type=int)
argparser.add_argument('--grad_clipping', default=5, type=float)
argparser.add_argument('--gpu', default=1, type=int)
argparser.add_argument('--gpu_idx', default=0, type=int)

args = argparser.parse_args()

# Parameters
model_file = args.model
assert model_file, 'Model output path is required'
model_file = os.path.join(args.model, 'model.{}.mdl'.format(timestamp))

embed_file_1 = args.embedding1
embed_file_2 = args.embedding2

charcnn_filters = [[int(f.split(',')[0]), int(f.split(',')[1])]
                   for f in args.charcnn_filters.split(';')]
use_gpu = (args.gpu == 1)
word_ignore_case = args.word_ignore_case
batch_size = args.batch_size
log_writer = None
if args.log:
    log_file = os.path.join(args.log, 'log.{}.txt'.format(timestamp))
    log_writer = open(log_file, 'a', encoding='utf-8')
    logger = get_logger(__name__, log_file=log_file)
else:
    logger = get_logger(__name__)

logger.info('----------')
logger.info('Parameters:')
for arg in vars(args):
    logger.info('{}: {}'.format(arg, getattr(args, arg)))
logger.info('----------')

# Parser for CoNLL format file
name_tagging_parser = ConllParser(Config({
    'separator': ' ',
    'token_col': 0,
    'label_col': 2,
    'skip_comment': True,
}))

# Load data sets
logger.info('Loading data sets')
datasets = {}
train_set_tgt = SequenceDataset(Config({
    'path': args.train_tgt,
    'parser': name_tagging_parser,
    'batch_size': batch_size
}))
dev_set_tgt = SequenceDataset(Config({
    'path': args.dev_tgt,
    'parser': name_tagging_parser,
    'batch_size': batch_size
}))
train_set_cl = SequenceDataset(Config({
    'path': args.train_cl,
    'parser': name_tagging_parser,
    'batch_size': batch_size
}))
dev_set_cl = SequenceDataset(Config({
    'path': args.dev_cl,
    'parser': name_tagging_parser,
    'batch_size': batch_size
}))

datasets['tgt'] = {
    'train': train_set_tgt,
    'dev': dev_set_tgt
}
datasets['cl'] = {
    'train': train_set_cl,
    'dev': dev_set_cl
}

# Vocabs
logger.info('Building vocabularies')
token_count_tgt, label_count_tgt, char_count_tgt = compute_metadata(
    [train_set_tgt, dev_set_tgt]
)
token_count_cl, label_count_cl, char_count_cl = compute_metadata(
    [train_set_cl, dev_set_cl]
)

token_vocab_1 = count2vocab([token_count_tgt],
                            start_idx=C.EMBED_START_IDX, ignore_case=word_ignore_case)
token_vocab_2 = count2vocab([token_count_cl],
                            start_idx=C.EMBED_START_IDX, ignore_case=word_ignore_case)
label_vocab_1 = count2vocab([label_count_tgt, label_count_cl], start_idx=0)
char_vocab = count2vocab([char_count_tgt, char_count_cl],
                         start_idx=C.CHAR_EMBED_START_IDX)

# Scan embedding file
if embed_file_1:
    logger.info('Scaning pre-trained embeddings for language 1')
    token_vocab_1 = {}
    with open(embed_file_1, 'r', encoding='utf-8') as embed_r:
        if args.embed_skip_first:
            embed_r.readline()
        for line in embed_r:
            try:
                token = line[:line.find(' ')]
                if word_ignore_case:
                    token = token.lower()
                if token not in token_vocab_1:
                    token_vocab_1[token] = len(token_vocab_1) + C.EMBED_START_IDX
                if token.lower() not in token_vocab_1:
                    token_vocab_1[token.lower()] = len(token_vocab_1) \
                                                 + C.EMBED_START_IDX
            except UnicodeDecodeError as e:
                logger.warning(e)

if embed_file_2:
    logger.info('Scaning pre-trained embeddings for language 2')
    token_vocab_2 = {}
    with open(embed_file_2, 'r', encoding='utf-8') as embed_r:
        if args.embed_skip_first:
            embed_r.readline()
        for line in embed_r:
            try:
                token = line[:line.find(' ')]
                if word_ignore_case:
                    token = token.lower()
                if token not in token_vocab_2:
                    token_vocab_2[token] = len(token_vocab_2) + C.EMBED_START_IDX
                if token.lower() not in token_vocab_2:
                    token_vocab_2[token.lower()] = len(token_vocab_2) \
                                                 + C.EMBED_START_IDX
            except UnicodeDecodeError as e:
                logger.warning(e)

idx_token_1 = {idx: token for token, idx in token_vocab_1.items()}
idx_token_2 = {idx: token for token, idx in token_vocab_2.items()}
idx_label_1 = {idx: label for label, idx in label_vocab_1.items()}
idx_token_1[C.UNKNOWN_TOKEN_INDEX] = C.UNKNOWN_TOKEN
idx_token_2[C.UNKNOWN_TOKEN_INDEX] = C.UNKNOWN_TOKEN
idx_tokens = {
    'tgt': idx_token_1,
    'cl': idx_token_2
}
idx_labels = {
    'tgt': idx_label_1,
    'cl': idx_label_1
}

# Numberize data sets
logger.info('Numberizing data sets')
numberize_datasets(
    [
        # Target task
        (train_set_tgt, token_vocab_1, label_vocab_1, char_vocab),
        (dev_set_tgt, token_vocab_1, label_vocab_1, char_vocab),
        # Auxiliary task: Cross-lingual
        (train_set_cl, token_vocab_2, label_vocab_1, char_vocab),
        (dev_set_cl, token_vocab_2, label_vocab_1, char_vocab)
    ],
    token_ignore_case=word_ignore_case,
    label_ignore_case=False,
    char_ignore_case=False
)

# Model components
logger.info('Building the models')
word_embed_1 = Embedding(Config({
    'num_embeddings': len(token_vocab_1),
    'embedding_dim': args.word_embed_dim,
    'padding': C.EMBED_START_IDX,
    'padding_idx': 0,
    'sparse': True,
    'trainable': True,
    'file': embed_file_1,
    'stats': args.embed_skip_first,
    'vocab': token_vocab_1,
    'ignore_case': word_ignore_case
}))
word_embed_2 = Embedding(Config({
    'num_embeddings': len(token_vocab_2),
    'embedding_dim': args.word_embed_dim,
    'padding': C.EMBED_START_IDX,
    'padding_idx': 0,
    'sparse': True,
    'trainable': True,
    'file': embed_file_2,
    'stats': args.embed_skip_first,
    'vocab': token_vocab_2,
    'ignore_case': word_ignore_case
}))
char_cnn = CharCNN(Config({
    'vocab_size': len(char_vocab),
    'padding': C.CHAR_EMBED_START_IDX,
    'dimension': args.char_embed_dim,
    'filters': charcnn_filters
}))
char_highway = Highway(Config({
    'num_layers': 2,
    'size': char_cnn.output_size,
    'activation': 'selu'
}))
lstm = LSTM(Config({
    'input_size': word_embed_1.output_size + char_cnn.output_size,
    'hidden_size': args.lstm_hidden_size,
    'forget_bias': 1.0,
    'batch_size': True,
    'bidirectional': True
}))
# CRF layer for task 1
crf_1 = CRF(Config({
    'label_vocab': label_vocab_1
}))
# Linear layers for task 1
shared_output_linear_1 = Linear(Config({
    'in_features': lstm.output_size,
    'out_features': len(label_vocab_1)
}))
spec_output_linear_1_1 = Linear(Config({
    'in_features': lstm.output_size,
    'out_features': len(label_vocab_1)
}))
spec_output_linear_1_2 = Linear(Config({
    'in_features': lstm.output_size,
    'out_features': len(label_vocab_1)
}))

# LSTM CRF Models
lstm_crf_tgt = LstmCrf(
    token_vocab=token_vocab_1,
    label_vocab=label_vocab_1,
    char_vocab=char_vocab,
    word_embedding=word_embed_1,
    char_embedding=char_cnn,
    crf=crf_1,
    lstm=lstm,
    univ_fc_layer=shared_output_linear_1,
    spec_fc_layer=spec_output_linear_1_1,
    embed_dropout_prob=args.embed_dropout,
    lstm_dropout_prob=args.lstm_dropout,
    linear_dropout_prob=args.linear_dropout,
    char_highway=char_highway
)
lstm_crf_cl = LstmCrf(
    token_vocab=token_vocab_2,
    label_vocab=label_vocab_1,
    char_vocab=char_vocab,
    word_embedding=word_embed_2,
    char_embedding=char_cnn,
    crf=crf_1,
    lstm=lstm,
    univ_fc_layer=shared_output_linear_1,
    spec_fc_layer=spec_output_linear_1_2,
    embed_dropout_prob=args.embed_dropout,
    lstm_dropout_prob=args.lstm_dropout,
    linear_dropout_prob=args.linear_dropout,
    char_highway=char_highway
)

models = {
    'tgt': lstm_crf_tgt,
    'cl': lstm_crf_cl
}

if use_gpu:
    torch.cuda.set_device(args.gpu_idx)
    lstm_crf_tgt.cuda()
    lstm_crf_cl.cuda()

# Task
optimizer_tgt = optim.SGD(
    filter(lambda p: p.requires_grad, lstm_crf_tgt.parameters()),
    lr=args.lr, momentum=args.momentum)
optimizer_cl = optim.SGD(
    filter(lambda p: p.requires_grad, lstm_crf_cl.parameters()),
    lr=args.lr, momentum=args.momentum)

optimizers = {
    'tgt': optimizer_tgt,
    'cl': optimizer_cl
}

state = {
    'model': {
        'word_embed_1': word_embed_1.state_dict(),
        'word_embed_2': word_embed_2.state_dict(),
        'char_cnn': char_cnn.state_dict(),
        'char_highway': char_highway.state_dict(),
        'lstm': lstm.state_dict(),
        'crf_1': crf_1.state_dict(),
        'shared_output_linear_1': shared_output_linear_1.state_dict(),
        'spec_output_linear_1_1': spec_output_linear_1_1.state_dict(),
        'spec_output_linear_1_2': spec_output_linear_1_2.state_dict(),
        'lstm_crf_tgt': lstm_crf_tgt.state_dict(),
        'lstm_crf_cl': lstm_crf_cl.state_dict()
    },
    'args': vars(args),
    'vocab': {
        'token_1': token_vocab_1,
        'token_2': token_vocab_2,
        'label_1': label_vocab_1,
        'char': char_vocab
    }
}

# Calculate mixing rates
r_tgt = math.sqrt(train_set_tgt.doc_num)
r_cl = 1.0 * .1 * math.sqrt(datasets['cl']['train'].doc_num)
r_sum = r_tgt + r_cl

num_cl = int(datasets['cl']['train'].doc_num * (r_cl / r_tgt) // batch_size)

try:
    global_step = 0
    best_dev_score = 0.0

    for epoch in range(args.max_epoch):
        logger.info('Epoch {}: Training'.format(epoch + 1))

        for ds in ['train', 'dev']:
            epoch_loss = []

            if ds =='train':
                batches = ['tgt'] * train_set_tgt.batch_num(batch_size) \
                          + ['cl'] * num_cl
                shuffle(batches)
                progress = tqdm.tqdm(total=len(batches), mininterval=1,
                                     desc=ds)
                for batch_ds in batches:
                    progress.update(1)
                    model = models[batch_ds]
                    optimizer = optimizers[batch_ds]
                    optimizer.zero_grad()

                    global_step += 1
                    batch = datasets[batch_ds]['train'].get_batch(gpu=use_gpu)
                    # TODO: Change the order of elements return by get_batch()
                    # to tokens, labels, seq_lens, chars, char_lens
                    # so that we can pass `batch` to model.loglik(*batch) and
                    # model.predict(*batch) without unpacking it.
                    tokens, labels, chars, seq_lens, char_lens = batch
                    loglik, _ = model.loglik(
                        tokens, labels, seq_lens, chars, char_lens)
                    loss = -loglik.mean()
                    loss.backward()

                    params = [p for n, p in model.named_parameters()
                              if 'embedding.weight' not in n]
                    clip_grad_norm(params, args.grad_clipping)
                    optimizer.step()
                progress.close()

            else:
                for task in ['tgt', 'cl']:
                    results = []
                    progress = tqdm.tqdm(
                        total=datasets[task][ds].batch_num(C.EVAL_BATCH_SIZE),
                        mininterval=1,
                        desc='{} {}'.format(task, ds))
                    for batch in datasets[task][ds].get_dataset(
                            gpu=use_gpu,
                            shuffle_inst=False,
                            # Using a larger batch size to accelerate the
                            # prediction. Change it to a smaller value if you
                            # get an error (e.g., out of memory).
                            batch_size=C.EVAL_BATCH_SIZE
                    ):
                        progress.update(1)
                        tokens, labels, chars, seq_lens, char_lens = batch
                        pred, _loss = models[task].predict(
                            tokens, labels, seq_lens, chars, char_lens
                        )
                        results.append((pred, labels, seq_lens, tokens))
                    progress.close()

                    fscore, prec, rec = evaluate(
                        results, idx_tokens[task], idx_labels[task],writer=log_writer)
                    if ds == 'dev' and task == 'tgt' and fscore > best_dev_score:
                        logger.info('New best score: {:.4f}'.format(fscore))
                        best_dev_score = fscore
                        logger.info('Saving the model to {}'.format(model_file))
                        torch.save(state, model_file)

        # learning rate decay
        lr = args.lr * args.decay_rate ** (global_step / args.decay_step)
        for optimizer in optimizers.values():
            for p in optimizer.param_groups:
                p['lr'] = lr
        logger.info('New learning rate: {}'.format(lr))

        logger.info('Best score: {}'.format(best_dev_score))
        logger.info('Model file: {}'.format(model_file))
        if args.log:
            logger.info('Log file: {}'.format(log_file))

except Exception:
    traceback.print_exc()
    if log_writer:
        log_writer.close()
