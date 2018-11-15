import os
import time
import logging
import traceback
from collections import Counter

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

import constant as C
from argparse import ArgumentParser

from torch.utils.data import DataLoader

from util import evaluate
from data import ConllParser, SeqLabelDataset, SeqLabelProcessor, count2vocab
from model import Linears, LSTM, CRF, CharCNN, Highway, LstmCrf, load_embedding

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

argparser = ArgumentParser()

argparser.add_argument('--train', help='Path to the training set file')
argparser.add_argument('--dev', help='Path to the dev set file')
argparser.add_argument('--test', help='Path to the test set file')
argparser.add_argument('--log', help='Path to the log dir')
argparser.add_argument('--model', help='Path to the model file')
argparser.add_argument('--batch_size', default=10, type=int, help='Batch size')
argparser.add_argument('--max_epoch', default=100, type=int)
argparser.add_argument('--word_embed',
                       help='Path to the pre-trained embedding file')
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

use_gpu = args.gpu and torch.cuda.is_available()
if use_gpu:
    torch.cuda.set_device(args.device)

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
parser = ConllParser(separator=' ', token_col=0, label_col=2, skip_comment=True)
train_set = SeqLabelDataset(args.train, parser=parser)
dev_set = SeqLabelDataset(args.dev, parser=parser)
test_set = SeqLabelDataset(args.test, parser=parser)
datasets = {'train': train_set, 'dev': dev_set, 'test': test_set}

# Vocabs
logger.info('Building vocabs')
token_count, char_count, label_count = Counter(), Counter(), Counter()
for _, ds in datasets.items():
    tc, cc, lc = ds.stats()
    token_count.update(tc)
    char_count.update(cc)
    label_count.update(lc)
token_vocab = count2vocab(token_count, offset=len(C.TOKEN_PADS), pads=C.TOKEN_PADS)
char_vocab = count2vocab(char_count, offset=len(C.CHAR_PADS), pads=C.CHAR_PADS)
label_vocab = count2vocab(label_count, offset=1, pads=[(C.PAD, C.PAD_INDEX)])
idx_token = {v: k for k, v in token_vocab.items()}
idx_label = {v: k for k, v in label_vocab.items()}
train_set.numberize(token_vocab, label_vocab, char_vocab)
dev_set.numberize(token_vocab, label_vocab, char_vocab)
test_set.numberize(token_vocab, label_vocab, char_vocab)
print('#token: {}'.format(len(token_vocab)))
print('#char: {}'.format(len(char_vocab)))
print('#label: {}'.format(len(label_vocab)))

# Embedding file
word_embed = load_embedding(args.word_embed,
                            dimension=args.word_embed_dim,
                            vocab=token_vocab)
charcnn_filters = [[int(f.split(',')[0]), int(f.split(',')[1])]
                   for f in args.charcnn_filters.split(';')]
char_embed = CharCNN(len(char_vocab),
                     args.char_embed_dim,
                     filters=charcnn_filters)
char_hw = Highway(char_embed.output_size,
                  layer_num=args.charhw_layer,
                  activation=args.charhw_func)
feat_dim = word_embed.embedding_dim + char_embed.output_size
lstm = LSTM(feat_dim,
            args.lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
            forget_bias=args.lstm_forget_bias
            )
crf = CRF(label_size=len(label_vocab) + 2)
linear = Linears(in_features=lstm.output_size,
                 out_features=len(label_vocab),
                 hiddens=[lstm.output_size // 2])
lstm_crf = LstmCrf(
    token_vocab, label_vocab, char_vocab,
    word_embedding=word_embed,
    char_embedding=char_embed,
    crf=crf,
    lstm=lstm,
    univ_fc_layer=linear,
    embed_dropout_prob=args.feat_dropout,
    lstm_dropout_prob=args.lstm_dropout,
    char_highway=char_hw if args.use_highway else None
)
if use_gpu:
    lstm_crf.cuda()
torch.set_num_threads(args.thread)

logger.debug(lstm_crf)

# Task
optimizer = optim.SGD(filter(lambda p: p.requires_grad, lstm_crf.parameters()),
                      lr=args.lr, momentum=args.momentum)
processor = SeqLabelProcessor(gpu=use_gpu)

train_args = vars(args)
train_args['word_embed_size'] = word_embed.num_embeddings
state = {
    'model': {
        'word_embed': word_embed.state_dict(),
        'char_embed': char_embed.state_dict(),
        'char_hw': char_hw.state_dict(),
        'lstm': lstm.state_dict(),
        'crf': crf.state_dict(),
        'linear': linear.state_dict(),
        'lstm_crf': lstm_crf.state_dict()
    },
    'args': train_args,
    'vocab': {
        'token': token_vocab,
        'label': label_vocab,
        'char': char_vocab,
    }
}
try:
    global_step = 0
    best_dev_score = best_test_score = 0.0

    for epoch in range(args.max_epoch):
        logger.info('Epoch {}: Training'.format(epoch))

        best = False
        for ds in ['train', 'dev', 'test']:
            dataset = datasets[ds]
            epoch_loss = []
            results = []

            for batch in DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=ds == 'train',
                drop_last=ds == 'train',
                collate_fn=processor.process
            ):
                optimizer.zero_grad()
                tokens, labels, chars, seq_lens, char_lens = batch
                if ds == 'train':
                    global_step += 1
                    loglik, _ = lstm_crf.loglik(
                        tokens, labels, seq_lens, chars, char_lens)
                    loss = -loglik.mean()
                    loss.backward()
                    clip_grad_norm_(lstm_crf.parameters(), args.grad_clipping)
                    optimizer.step()
                else:
                    pred, loss = lstm_crf.predict(
                        tokens, labels, seq_lens, chars, char_lens)
                    results.append((pred, labels, seq_lens, tokens))

                epoch_loss.append(loss.item())

            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            logger.info('{} Loss: {:.4f}'.format(ds, epoch_loss))

            if ds == 'dev' or ds == 'test':
                fscore, prec, rec = evaluate(
                    results, idx_token, idx_label, writer=log_writer
                )
                if ds == 'dev' and fscore > best_dev_score:
                    logger.info('New best score: {:.4f}'.format(fscore))
                    best_dev_score = fscore
                    best = True
                    logger.info(
                        'Saving the current model to {}'.format(model_file))
                    torch.save(state, model_file)
                if best and ds == 'test':
                    best_test_score = fscore

        # learning rate decay
        lr = args.lr * args.decay_rate ** (global_step / args.decay_step)
        for p in optimizer.param_groups:
            p['lr'] = lr
        logger.info('New learning rate: {}'.format(lr))

    logger.info('Best score: {}'.format(best_dev_score))
    logger.info('Best test score: {}'.format(best_test_score))
    logger.info('Model file: {}'.format(model_file))
    if args.log:
        logger.info('Log file: {}'.format(log_file))
        log_writer.close()
except Exception:
    traceback.print_exc()
    if log_writer:
        log_writer.close()