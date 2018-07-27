import os
import time
import traceback
import constant as C

import torch

from model import Linear, LSTM, CRF, CharCNN, Highway, LstmCrf, Embedding
from argparse import ArgumentParser
from util import get_logger, evaluate, Config
from data import (
    SequenceDataset, ConllParser,
    compute_metadata, count2vocab, numberize_datasets
)

timestamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())

argparser = ArgumentParser()

argparser.add_argument('--model', help='Path to the model file')
argparser.add_argument('--file', help='Path to the file to evaluate')
argparser.add_argument('--log', help='Path to the log dir')
argparser.add_argument('--gpu', default=1, type=int, help='Use GPU')
argparser.add_argument('--gpu_idx', default=0, type=int)

args = argparser.parse_args()

# Parameters
model_file = args.model
data_file = args.file
assert model_file, 'Model file is required'
assert data_file, 'Data file is required'

use_gpu = (args.gpu == 1)


log_writer = None
if args.log:
    log_file = os.path.join(args.log, 'log.{}.txt'.format(timestamp))
    log_writer = open(log_file, 'a', encoding='utf-8')
    logger = get_logger(__name__, log_file=log_file)
else:
    logger = get_logger(__name__)

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
word_embed = Embedding(Config({
    'num_embeddings': len(token_vocab),
    'embedding_dim': train_args['word_embed_dim'],
    'padding': C.EMBED_START_IDX,
    'padding_idx': 0,
    'sparse': True,
    'trainable': True,
    'stats': train_args['embed_skip_first'],
    'vocab': token_vocab,
    'ignore_case': train_args['word_ignore_case']
}))
char_cnn = CharCNN(Config({
    'vocab_size': len(char_vocab),
    'padding': C.CHAR_EMBED_START_IDX,
    'dimension': train_args['char_embed_dim'],
    'filters': charcnn_filters
}))
char_highway = Highway(Config({
    'num_layers': 2,
    'size': char_cnn.output_size,
    'activation': 'selu'
}))
lstm = LSTM(Config({
    'input_size': word_embed.output_size + char_cnn.output_size,
    'hidden_size': train_args['lstm_hidden_size'],
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
word_embed.load_state_dict(state['model']['word_embed'])
char_cnn.load_state_dict(state['model']['char_cnn'])
char_highway.load_state_dict(state['model']['char_highway'])
lstm.load_state_dict(state['model']['lstm'])
crf.load_state_dict(state['model']['crf'])
output_linear.load_state_dict(state['model']['output_linear'])
lstm_crf = LstmCrf(
    token_vocab=token_vocab,
    label_vocab=label_vocab,
    char_vocab=char_vocab,
    word_embedding=word_embed,
    char_embedding=char_cnn,
    crf=crf,
    lstm=lstm,
    univ_fc_layer=output_linear,
    embed_dropout_prob=train_args['embed_dropout'],
    lstm_dropout_prob=train_args['lstm_dropout'],
    linear_dropout_prob=train_args['linear_dropout'],
    char_highway=char_highway
)
lstm_crf.load_state_dict(state['model']['lstm_crf'])

if use_gpu:
    torch.cuda.set_device(args.gpu_idx)
    lstm_crf.cuda()
else:
    lstm_crf.cpu()

# Load dataset
logger.info('Loading data')
conll_parser = ConllParser(Config({
    'separator': '\t',
    'token_col': 0,
    'label_col': 1,
    'skip_comment': True,
}))
test_set = SequenceDataset(Config({
    'path': data_file, 'parser': conll_parser
}))
numberize_datasets([(test_set, token_vocab, label_vocab, char_vocab)],
                   token_ignore_case=train_args['word_ignore_case'],
                   label_ignore_case=False,
                   char_ignore_case=False)
idx_token = {idx: token for token, idx in token_vocab.items()}
idx_label = {idx: label for label, idx in label_vocab.items()}
idx_token[C.UNKNOWN_TOKEN_INDEX] = C.UNKNOWN_TOKEN

try:
    results = []
    dataset_loss = []
    for batch in test_set.get_dataset(gpu=use_gpu,
                                      shuffle_inst=False,
                                      batch_size=100):
        tokens, labels, chars, seq_lens, char_lens = batch
        pred, loss = lstm_crf.predict(
            tokens, labels, seq_lens, chars, char_lens)
        results.append((pred, labels, seq_lens, tokens))
        dataset_loss.append(loss.data[0])

    dataset_loss = sum(dataset_loss) / len(dataset_loss)
    fscore, prec, rec = evaluate(results, idx_token, idx_label, writer=log_writer)
except KeyboardInterrupt:
    traceback.print_exc()
    if log_writer:
        log_writer.close()
