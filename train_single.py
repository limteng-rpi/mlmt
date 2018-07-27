import os
import time
import traceback

import tqdm

import torch
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

argparser.add_argument('--train', help='Path to the training set file')
argparser.add_argument('--dev', help='Path to the dev set file')
argparser.add_argument('--test', help='Path to the test set file')
argparser.add_argument('--log', help='Path to the log dir')
argparser.add_argument('--model', help='Path to the model file')
argparser.add_argument('--batch_size', default=10, type=int, help='Batch size')
argparser.add_argument('--max_epoch', default=100, type=int)
argparser.add_argument('--embedding',
                       help='Path to the pre-trained embedding file')
argparser.add_argument('--embed_skip_first', dest='embed_skip_first',
                       action='store_true',
                       help='Skip the first line of the embedding file')
argparser.set_defaults(embed_skip_first=True)
argparser.add_argument('--word_embed_dim', type=int, default=100,
                       help='Word embedding dimension')
# argparser.add_argument('--word_embed_case', type=int, default=0,
#                        help='Word embedding case-sensitive')
argparser.add_argument('--word_ignore_case', dest='word_ignore_case',
                       action='store_true')
argparser.set_defaults(word_ignore_case=True)
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

embed_file = args.embedding
charcnn_filters = [[int(f.split(',')[0]), int(f.split(',')[1])]
                   for f in args.charcnn_filters.split(';')]
use_gpu = (args.gpu == 1)
# word_ignore_case = (args.word_embed_case == 0)
word_ignore_case = args.word_ignore_case
log_writer = None
if args.log:
    log_file = os.path.join(args.log, 'log.{}.txt'.format(timestamp))
    log_writer = open(log_file, 'a', encoding='utf-8')
    logger = get_logger(__name__, log_file=log_file)
    # logger.info('Log file: {}'.format(log_file))
else:
    logger = get_logger(__name__)

logger.info('----------')
logger.info('Parameters:')
for arg in vars(args):
    logger.info('{}: {}'.format(arg, getattr(args, arg)))
logger.info('----------')

# Parser for CoNLL format file
conll_parser = ConllParser(Config({
    'separator': '\t',
    'token_col': 0,
    'label_col': 1,
    'skip_comment': True,
}))

# Load datasets
logger.info('Loading datasets')
train_set = SequenceDataset(Config({
    'path': args.train, 'parser': conll_parser, 'batch_size': args.batch_size}))
dev_set = SequenceDataset(Config({
    'path': args.dev, 'parser': conll_parser}))
test_set = SequenceDataset(Config({
    'path': args.test, 'parser': conll_parser}))
datasets = {'train': train_set, 'dev': dev_set, 'test': test_set}

# Vocabs
logger.info('Building vocabularies')
token_count, label_count, char_count = compute_metadata(
    [train_set, dev_set, test_set])
token_vocab = count2vocab([token_count],
                          start_idx=C.EMBED_START_IDX,
                          ignore_case=word_ignore_case)
label_vocab = count2vocab([label_count],
                          start_idx=0,
                          sort=True,
                          ignore_case=False)
char_vocab = count2vocab([char_count],
                         ignore_case=False,
                         start_idx=C.CHAR_EMBED_START_IDX)
if embed_file:
    logger.info('Scaning pre-trained embeddings')
    token_vocab = {}
    with open(embed_file, 'r', encoding='utf-8') as embed_r:
        if args.embed_skip_first:
            embed_r.readline()
        for line in embed_r:
            try:
                token = line[:line.find(' ')]
                if word_ignore_case:
                    token = token.lower()
                if token not in token_vocab:
                    token_vocab[token] = len(token_vocab) + C.EMBED_START_IDX
                if token.lower() not in token_vocab:
                    token_vocab[token.lower()] = len(token_vocab) \
                                                 + C.EMBED_START_IDX
            except UnicodeDecodeError as e:
                logger.warning(e)
idx_token = {idx: token for token, idx in token_vocab.items()}
idx_label = {idx: label for label, idx in label_vocab.items()}
idx_token[C.UNKNOWN_TOKEN_INDEX] = C.UNKNOWN_TOKEN

# Numberize datasets
logger.info('Numberizing datasets')
numberize_datasets(
    [
        (train_set, token_vocab, label_vocab, char_vocab),
        (dev_set, token_vocab, label_vocab, char_vocab),
        (test_set, token_vocab, label_vocab, char_vocab),
    ],
    token_ignore_case=word_ignore_case,
    label_ignore_case=False,
    char_ignore_case=False
)

# Model components
logger.info('Building the model')
word_embed = Embedding(Config({
    'num_embeddings': len(token_vocab),
    'embedding_dim': args.word_embed_dim,
    'padding': C.EMBED_START_IDX,
    'padding_idx': 0,
    'sparse': True,
    'trainable': True,
    'file': embed_file,
    'stats': args.embed_skip_first,
    'vocab': token_vocab,
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
    'input_size': word_embed.output_size + char_cnn.output_size,
    'hidden_size': args.lstm_hidden_size,
    'forget_bias': 1.0,
    'batch_first': True,
    'bidirectional': True
}))
crf = CRF(Config({
    'label_vocab': label_vocab
}))
output_linear = Linear(Config({
    'in_features': lstm.output_size,
    'out_features': len(label_vocab)
}))

# LSTM CRF Model
lstm_crf = LstmCrf(
    token_vocab=token_vocab,
    label_vocab=label_vocab,
    char_vocab=char_vocab,
    word_embedding=word_embed,
    char_embedding=char_cnn,
    crf=crf,
    lstm=lstm,
    univ_fc_layer=output_linear,
    embed_dropout_prob=args.embed_dropout,
    lstm_dropout_prob=args.lstm_dropout,
    linear_dropout_prob=args.linear_dropout,
    char_highway=char_highway
)

if use_gpu:
    torch.cuda.set_device(args.gpu_idx)
    lstm_crf.cuda()

# Task
optimizer = optim.SGD(filter(lambda p: p.requires_grad, lstm_crf.parameters()),
                      lr=args.lr, momentum=args.momentum)

state = {
    'model': {
        'word_embed': word_embed.state_dict(),
        'char_cnn': char_cnn.state_dict(),
        'char_highway': char_highway.state_dict(),
        'lstm': lstm.state_dict(),
        'crf': crf.state_dict(),
        'output_linear': output_linear.state_dict(),
        'lstm_crf': lstm_crf.state_dict()
    },
    'args': vars(args),
    'vocab': {
        'token': token_vocab,
        'label': label_vocab,
        'char': char_vocab,
    }
}

try:
    global_step = 0
    best_dev_score = 0.0

    for epoch in range(args.max_epoch):
        logger.info('Epoch {}: Training'.format(epoch))

        for ds in ['train', 'dev', 'test']:
            dataset = datasets[ds]
            epoch_loss = []
            results = []

            progress = tqdm.tqdm(total=dataset.batch_num(args.batch_size),
                                 mininterval=1, desc=ds)
            for batch in dataset.get_dataset(
                    gpu=use_gpu,
                    shuffle_inst=ds == 'train',
                    batch_size=args.batch_size):
                optimizer.zero_grad()
                progress.update(1)
                tokens, labels, chars, seq_lens, char_lens = batch
                if ds == 'train':
                    global_step += 1
                    loglik, _ = lstm_crf.loglik(
                        tokens, labels, seq_lens, chars, char_lens)
                    loss = -loglik.mean()
                    loss.backward()

                    params = [p for n, p in lstm_crf.named_parameters()
                              if 'embedding.weight' not in n]
                    clip_grad_norm(params, args.grad_clipping)
                    optimizer.step()

                else:
                    pred, loss = lstm_crf.predict(
                        tokens, labels, seq_lens, chars, char_lens)
                    results.append((pred, labels, seq_lens, tokens))

                epoch_loss.append(loss.data[0])
            progress.close()

            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            logger.info('{} Loss: {:.4f}'.format(ds, epoch_loss))

            if ds == 'dev' or ds == 'test':
                fscore, prec, rec = evaluate(
                    results, idx_token, idx_label, writer=log_writer)
                if ds == 'dev' and fscore > best_dev_score:
                    logger.info('New best score: {:.4f}'.format(fscore))
                    best_dev_score = fscore
                    logger.info(
                        'Saving the current model to {}'.format(model_file))
                    torch.save(state, model_file)

        # learning rate decay
        lr = args.lr * args.decay_rate ** (global_step / args.decay_step)
        for p in optimizer.param_groups:
            p['lr'] = lr
        logger.info('New learning rate: {}'.format(lr))

    logger.info('Best score: {}'.format(best_dev_score))
except KeyboardInterrupt:
    traceback.print_exc()
    if log_writer:
        log_writer.close()
