import logging
import os
import time
import traceback

from torch.utils.data import DataLoader

import constant as C

import torch

from argparse import ArgumentParser
from model import Linear, LSTM, CRF, CharCNN, Highway, LstmCrf
from util import evaluate
from data import ConllParser, SeqLabelDataset, SeqLabelProcessor

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

argparser = ArgumentParser()

argparser.add_argument('--model', help='Path to the model file')
argparser.add_argument('--file', help='Path to the file to evaluate')
argparser.add_argument('--log', help='Path to the log dir')
argparser.add_argument('--gpu', action='store_true')
argparser.add_argument('--device', default=0, type=int)

args = argparser.parse_args()

use_gpu = args.gpu and torch.cuda.is_available()
if use_gpu:
    torch.cuda.set_device(args.device)

# Parameters
model_file = args.model
data_file = args.file

log_writer = None
if args.log:
    log_file = os.path.join(args.log, 'log.{}.txt'.format(timestamp))
    log_writer = open(log_file, 'a', encoding='utf-8')
    logger.addHandler(logging.FileHandler(log_file, encoding='utf-8'))

# Load saved model
logger.info('Loading saved model from {}'.format(model_file))
state = torch.load(model_file)
token_vocab = state['vocab']['token']
label_vocab = state['vocab']['label']
char_vocab = state['vocab']['char']
train_args = state['args']
charcnn_filters = [[int(f.split(',')[0]), int(f.split(',')[1])]
                   for f in train_args['charcnn_filters'].split(';')]

# Resume model
logger.info('Resuming the model')
word_embed = torch.nn.Embedding(train_args['word_embed_size'],
                                train_args['word_embed_dim'],
                                sparse=True,
                                padding_idx=C.PAD_INDEX)
char_embed = CharCNN(len(char_vocab),
                     train_args['char_embed_dim'],
                     filters=charcnn_filters)
char_hw = Highway(char_embed.output_size,
                  layer_num=train_args['charhw_layer'],
                  activation=train_args['charhw_func'])
feat_dim = word_embed.embedding_dim + char_embed.output_size
lstm = LSTM(feat_dim,
            train_args['lstm_hidden_size'],
            batch_first=True,
            bidirectional=True,
            forget_bias=train_args['lstm_forget_bias'])
crf = CRF(label_size=len(label_vocab) + 2)
linear = Linear(in_features=lstm.output_size,
                out_features=len(label_vocab))
lstm_crf = LstmCrf(
    token_vocab, label_vocab, char_vocab,
    word_embedding=word_embed,
    char_embedding=char_embed,
    crf=crf,
    lstm=lstm,
    univ_fc_layer=linear,
    embed_dropout_prob=train_args['feat_dropout'],
    lstm_dropout_prob=train_args['lstm_dropout'],
    char_highway=char_hw if train_args['use_highway'] else None
)

word_embed.load_state_dict(state['model']['word_embed'])
char_embed.load_state_dict(state['model']['char_embed'])
char_hw.load_state_dict(state['model']['char_hw'])
lstm.load_state_dict(state['model']['lstm'])
crf.load_state_dict(state['model']['crf'])
linear.load_state_dict(state['model']['linear'])
lstm_crf.load_state_dict(state['model']['lstm_crf'])

if use_gpu:
    lstm_crf.cuda()

# Load dataset
logger.info('Loading data')
parser = ConllParser()
test_set = SeqLabelDataset(data_file, parser=parser)
test_set.numberize(token_vocab, label_vocab, char_vocab)
idx_token = {v: k for k, v in token_vocab.items()}
idx_label = {v: k for k, v in label_vocab.items()}
processor = SeqLabelProcessor(gpu=use_gpu)

try:
    results = []
    dataset_loss = []
    for batch in DataLoader(
            test_set,
            batch_size=50,
            shuffle=False,
            collate_fn=processor.process
    ):
        tokens, labels, chars, seq_lens, char_lens = batch
        pred, loss = lstm_crf.predict(
            tokens, labels, seq_lens, chars, char_lens)
        results.append((pred, labels, seq_lens, tokens))
        dataset_loss.append(loss.data[0])

    dataset_loss = sum(dataset_loss) / len(dataset_loss)
    fscore, prec, rec = evaluate(results, idx_token, idx_label,
                                 writer=log_writer)
    if args.log:
        logger.info('Log file: {}'.format(log_file))
        log_writer.close()
except KeyboardInterrupt:
    traceback.print_exc()
    if log_writer:
        log_writer.close()
