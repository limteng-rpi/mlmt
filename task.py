import io
import sys
import time
import json
import torch
import logging
import numpy as np
import constant as C
import torch.optim as optim

from torch.nn.utils import clip_grad_norm
from numpy.random import choice, randint
from conlleval import evaluate, report, metrics
from collections import defaultdict, namedtuple
from data import (ConllParser, SequenceDataset, Numberizer, CharNumberizer,
                  count2vocab, Word2VecDataset, CbowDataset, create_parser,
                  create_dataset, numberize_datasets)
from model import Linear, LstmCrf

logger = logging.getLogger(__name__)

SCORES = namedtuple('SCORES', ['fscore', 'precision', 'recall', 'loss'])

class Task(object):

    def __init__(self,
                 name,
                 model,
                 datasets,
                 vocabs,
                 gpu=False,
                 prob=1.0,
                 lr=0.001,
                 momentum=.9,
                 decay_rate=.9,
                 decay_step=10000,
                 gradient_clipping=5.0,
                 require_eval=True,
                 ref=False,
                 aux_task=False,
                 aux_lang=False
                 ):
        self.name = name
        self.model = model
        self.prob = prob
        self.gpu = gpu
        self.require_eval = require_eval

        self.datasets = datasets
        self.train = datasets.get('train', None)
        self.dev = datasets.get('dev', None)
        self.test = datasets.get('test', None)

        self.vocabs = vocabs
        self.token_vocab = vocabs.get('token')
        self.label_vocab = vocabs.get('label')
        self.char_vocab = vocabs.get('char')
        self.ref = ref

        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, momentum=momentum)
        self.lr = lr
        self.momentum = momentum
        self.task_step = 0
        self.decay_rate = decay_rate
        self.decay_step = float(decay_step)
        self.gradient_clipping = gradient_clipping

        self.aux_task = aux_task
        self.aux_lang = aux_lang

        if gpu:
            self.model.cuda()

    def step(self):
        raise NotImplementedError()

    def eval(self, dataset_name, log_output=None):
        raise NotImplementedError()

    def learning_rate_decay(self):
        lr = self.lr * self.decay_rate ** (self.task_step / self.decay_step)
        for p in self.optimizer.param_groups:
            p['lr'] = lr

    def update_learning_rate(self, lr):
        for p in self.optimizer.param_groups:
            p['lr'] = lr


class SequenceTask(Task):

    def __init__(self,
                 name,
                 model,
                 datasets,
                 vocabs,
                 gpu=False,
                 prob=1.0,
                 lr=0.001,
                 momentum=.9,
                 decay_rate=.9,
                 decay_step=10000,
                 gradient_clipping=5.0,
                 require_eval=True,
                 ref=False,
                 aux_task=False,
                 aux_lang=False
                 ):
        super(SequenceTask, self).__init__(name,
                                           model,
                                           datasets,
                                           vocabs,
                                           gpu,
                                           prob,
                                           lr,
                                           momentum,
                                           decay_rate,
                                           decay_step,
                                           gradient_clipping,
                                           require_eval,
                                           ref,
                                           aux_task,
                                           aux_lang
                                           )
        self.label_size = len(self.label_vocab)
        self.idx_label = {i: l for l, i in self.label_vocab.items()}
        self.idx_token = {i: t for t, i in self.token_vocab.items()}


    def step(self):
        self.task_step += 1
        self.optimizer.zero_grad()
        (
            tokens, labels, chars, seq_lens, char_lens
        ) = self.train.get_batch(gpu=self.gpu)
        loglik, _ = self.model.loglik(tokens, labels, seq_lens, chars,
                                      char_lens)
        loss = -loglik.mean()
        loss.backward()

        params = []
        for n, p in self.model.named_parameters():
            if 'embedding.weight' not in n:
                params.append(p)
        clip_grad_norm(params, self.gradient_clipping)
        self.optimizer.step()


class NameTagging(SequenceTask):

    def eval(self, dataset_name, log_output=None):
        dataset = self.datasets.get(dataset_name, None)
        if dataset is None:
            return

        results = []
        logger.info('Evaluating {} ({})'.format(self.name, dataset_name))
        set_loss = 0
        for tokens, labels, chars, seq_lens, char_lens in dataset.get_dataset(volatile=True, gpu=self.gpu):
            preds, loss = self.model.predict(tokens,
                                             labels,
                                             seq_lens,
                                             chars,
                                             char_lens)
            set_loss += float(loss.data[0])
            for pred, gold, seq_len, ts in zip(preds, labels, seq_lens, tokens):
                l = int(seq_len.data[0])
                pred = pred.data.tolist()[:l]
                gold = gold.data.tolist()[:l]
                ts = ts.data.tolist()[:l]
                for p, g, t in zip(pred, gold, ts):
                    t = self.idx_token.get(t, 'UNK')
                    results.append('{} {} {}'.format(t,
                                                     self.idx_label[g],
                                                     self.idx_label[p]))
                results.append('')
        counts = evaluate(results)
        overall, by_type = metrics(counts)
        report(counts)
        logger.info('Loss: {:.5f}'.format(set_loss))
        return SCORES(fscore=overall.fscore,
                      precision=overall.prec,
                      recall=overall.rec,
                      loss=set_loss)


class PosTagging(SequenceTask):

    def eval(self, dataset_name, log_output=None):
        dataset = self.datasets.get(dataset_name, None)
        if dataset is None:
            return

        total_num = 0
        correct_num = 0
        logger.info('Evaluating {} ({})'.format(self.name, dataset_name))
        set_loss = 0

        results = []
        for tokens, labels, chars, seq_lens, char_lens in dataset.get_dataset(
            volatile=True, gpu=self.gpu):
            preds, loss = self.model.predict(tokens, labels, seq_lens, chars, char_lens)
            set_loss += float(loss.data[0])
            for pred, gold, seq_len, ts in zip(preds, labels, seq_lens, tokens):
                l = int(seq_len.data[0])
                total_num += l
                pred = pred.data.tolist()[:l]
                gold = gold.data.tolist()[:l]
            pred = np.array(pred)
            gold = np.array(gold)
            correct = (pred == gold).sum()
            correct_num += correct
        accuracy = correct_num / total_num
        logger.info('Accuracy: {0:.5f}'.format(accuracy))
        logger.info('Loss: {}'.format(set_loss))
        return SCORES(fscore=accuracy,
                      precision=accuracy,
                      recall=accuracy,
                      loss=set_loss)


class MultiTask(object):

    def __int__(self, tasks, eval_freq=1000):
        self.tasks = tasks
        self.task_probs = []
        self.update_probs()
        self.global_step = 0
        self.eval_freq = eval_freq
        self.ref_task = 0
        self.best_ref_score = -1.0
        self.best_scores = []
        for task_idx, task in enumerate(self.tasks):
            if task.ref:
                self.ref_tasks = task_idx
                break

    def update_probs(self):

        def auto_prob(task):
            doc_num = len(task.train.dataset)
            theta_task = .1 if task.aux_task else 1
            theta_lang = .1 if task.aux_lang else 1
            prob = doc_num ** .5 * theta_task * theta_lang
            return prob

        task_probs = [auto_prob(t) for t in self.tasks]
        task_prob_sum = sum(task_probs)
        self.task_probs = [p / task_prob_sum for p in task_probs]

    def step(self):
        self.global_step += 1
        task = choice(self.tasks,p=self.task_probs)
        task.learning_rate_decay()
        task.step()

        if self.global_step % self.eval_freq == 0:
            scores = []
            ref_score = 0
            for task_idx, task in enumerate(self.tasks):
                if task.require_eval:
                    dev_scores = task.eval('dev')
                    test_scores = task.eval('test')
                    if task_idx == self.ref_task:
                        ref_score = dev_scores.fscore
                    scores.append((task_idx, dev_scores, test_scores))
            if  ref_score > self.best_ref_score:
                self.best_ref_score = ref_score
                self.best_scores = scores



