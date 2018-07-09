import numpy as np
import torch.optim as optim

from torch.nn.utils import clip_grad_norm
from numpy.random import choice
from conlleval import evaluate, report, metrics
from collections import defaultdict, namedtuple
from data import (count2vocab, create_parser, create_dataset,
                  numberize_datasets)
from model import LstmCrf, create_module
from util import Config, get_logger

logger = get_logger(__name__)

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

    def __init__(self, tasks, eval_freq=1000):
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


def compute_metadata(datasets):
    """Compute tokens, labels, and characters in the given data sets.

    :param datasets: A list of data sets.
    :return: dicts of token, label, and character counts.
    """
    token_count = defaultdict(int)
    label_count = defaultdict(int)
    char_count = defaultdict(int)

    for dataset in datasets:
        if dataset:
            t, l, c = dataset.metadata()
            for k, v in t.items():
                token_count[k] += v
            for k, v in l.items():
                label_count[k] += v
            for k, v in c.items():
                char_count[k] += v

    return token_count, label_count, char_count


def build_tasks_from_file(conf_path, options=None):
    if type(conf_path) is str:
        conf = Config.read(conf_path)
    elif type(conf_path) is Config:
        conf = conf_path
    else:
        raise TypeError('Unknown configuration type. Expect str or Config.')

    if options:
        for k, v in options:
            conf.update_value(k, v)

    # Create data sets
    logger.info('Loading data sets')
    datasets = {}
    lang_datasets = defaultdict(list)
    task_datasets = defaultdict(list)
    for dataset in conf.datasets:
        parser = create_parser(dataset.parser.format, dataset.parser)
        (
            train_conf, dev_conf, test_conf
        ) = dataset.clone(), dataset.clone(), dataset.clone()
        train_conf.update({'path': dataset.files.train,
                           'parser': parser})
        dev_conf.update({'path': dataset.files.dev,
                         'parser': parser,
                         'sample': None})
        train_dataset = create_dataset(dataset.type, train_conf)
        dev_dataset = create_dataset(dataset.type, dev_conf)
        if hasattr(dataset.files, 'test'):
            test_conf.update({'path': dataset.files.test,
                              'parser': parser,
                              'sample': None})
            test_dataset = create_dataset(dataset.type, test_conf)
        datasets[dataset.name] = {
            'train': train_dataset,
            'dev': dev_dataset,
            'test': test_dataset,
            'language': dataset.language,
            'task': dataset.task
        }
        lang_datasets[dataset.language].append(dataset.name)
        task_datasets[dataset.task].append(dataset.name)

    # Create vocabs
    # I only keep words in the data sets to save memory
    # If the model will be applied to an unknown test set, it is better to keep
    # all words in pre-trained embeddings.
    logger.info('Creating vocabularies')
    dataset_counts = {}
    lang_token_vocabs = {}
    task_label_vocabs = {}
    for name, ds in datasets.items():
        dataset_counts[name] = compute_metadata(
            [ds['train'], ds['dev'], ds['test']]
        )
    for lang, ds in lang_datasets.items():
        counts = [dataset_counts[d][0] for d in ds]
        lang_token_vocabs[lang] = count2vocab(counts,
                                              ignore_case=True,
                                              start_idx=2)
    for task, ds in task_datasets.items():
        counts = [dataset_counts[d][1] for d in ds]
        task_label_vocabs[task] = count2vocab(counts,
                                              ignore_case=False,
                                              start_idx=0,
                                              sort=True)
    char_vocab = count2vocab([c[2] for c in dataset_counts.values()],
                             ignore_case=False, start_idx=0)

    # Report stats
    for lang, vocab in lang_token_vocabs.items():
        logger.info('#{} token: {}'.format(lang, len(vocab)))
    for task, vocab in task_label_vocabs.items():
        logger.info('#{} label: {}'.format(task, len(vocab)))
        logger.info(vocab)

    # Numberize datasets
    logger.info('Numberizing data sets')
    numberize_conf = []
    for ds in datasets.values():
        numberize_conf.append((ds['train'],
                               lang_token_vocabs[ds['language']],
                               task_label_vocabs[ds['task']],
                               char_vocab))
        numberize_conf.append((ds['dev'],
                               lang_token_vocabs[ds['language']],
                               task_label_vocabs[ds['task']],
                                                 char_vocab))
        numberize_conf.append((ds['test'],
                               lang_token_vocabs[ds['language']],
                               task_label_vocabs[ds['task']],
                                                 char_vocab))
    numberize_datasets(numberize_conf,
                       token_ignore_case=True,
                       label_ignore_case=False,
                       char_ignore_case=False)

    # Initialize component confs
    logger.info('Initializing component configurations')
    word_embed_dim = char_embed_dim = lstm_output_dim = 0
    cpnt_confs = {}
    for cpnt in conf.components:
        if cpnt.model == 'embedding':
            cpnt.embedding_dim = cpnt.dimension
            word_embed_dim = cpnt.dimension
        elif cpnt.model == 'char_cnn':
            cpnt.vocab_size = len(char_vocab)
            char_embed_dim = sum([x[1] for x in cpnt.filters])
        elif cpnt.model == 'lstm':
            lstm_output_dim = cpnt.hidden_size * (2 if cpnt.bidirectional else 1)
        cpnt_confs[cpnt.name] = cpnt.clone()

    # Update component configurations
    target_task = ''
    target_lang = ''
    for task_conf in conf.tasks:
        language = task_conf.language
        task = task_conf.task
        if task_conf.get('ref', False):
            target_lang = language
            target_task = task
        model_conf = task_conf.model
        if model_conf.model != 'lstm_crf':
            continue
        # Update word embedding configuration
        cpnt_confs[model_conf.word_embed].num_embeddings = len(
            lang_token_vocabs[language])
        cpnt_confs[model_conf.word_embed].vocab = lang_token_vocabs[language]
        # Update output layer configuration
        cpnt_confs[model_conf.univ_layer].out_features = len(
            task_label_vocabs[task]
        )
        if hasattr(model_conf, 'spec_layer'):
            cpnt_confs[model_conf.spec_layer].out_features = len(
                task_label_vocabs[task]
            )
        # Update CRF configuration
        cpnt_confs[model_conf.crf].label_vocab = task_label_vocabs[task]

    for _, cpnt_conf in cpnt_confs.items():
        if cpnt_conf.model == 'linear' and cpnt_conf.position == 'output':
            cpnt_conf.in_features = lstm_output_dim
        if cpnt_conf.model == 'lstm':
            cpnt_conf.input_size = char_embed_dim + word_embed_dim
        if cpnt_conf.model == 'highway' and cpnt_conf.position == 'char':
            cpnt_conf.size == char_embed_dim

    # Create components
    logger.info('Creating components')
    components = {k: create_module(v.model, v) for k, v in cpnt_confs.items()}

    # Construct models
    tasks = []
    for task_conf in conf.tasks:
        model_conf = task_conf.model
        language = task_conf.language
        task = task_conf.task
        if model_conf.model == 'lstm_crf':
            model = LstmCrf(
                lang_token_vocabs[language],
                task_label_vocabs[task],
                char_vocab,
                word_embedding=components[model_conf.word_embed],
                char_embedding=components[model_conf.char_embed] if hasattr(
                    model_conf, 'char_embed') else None,
                crf=components[model_conf.crf],
                lstm=components[model_conf.lstm],
                input_layer=None,
                univ_layer=components[model_conf.univ_layer],
                spec_layer=components[model_conf.spec_layer] if hasattr(
                    model_conf, 'spec_linear') else None,
                embedding_dropout_prob=model_conf.embed_dropout,
                lstm_dropout_prob=model_conf.lstm_dropout,
                linear_dropout_prob=model_conf.linear_dropout,
                char_highway=components[model_conf.char_highway] if hasattr(
                    model_conf, 'char_highway') else None,
                use_char_embedding=model_conf.use_char_embedding if hasattr(
                    model_conf, 'use_char_embedding') else True,
            )
        # elif model_conf.model == 'cbow':
        #     pass
        else:
            raise ValueError('Unknown model: {}'.format(model_conf.model))
        logger.debug(model)

        task_classes = {'ner': NameTagging, 'pos': PosTagging}
        if task in task_classes:
            task_obj = task_classes[task](
                task_conf.name,
                model,
                datasets=datasets[task_conf.dataset],
                vocabs={
                    'token': lang_token_vocabs[language],
                    'label': task_label_vocabs[task],
                    'char': char_vocab
                },
                gpu=task_conf.gpu,
                # TODO: 'gpu' -> global config
                prob=getattr(task_conf, 'prob', 1.0),
                lr=getattr(task_conf, 'learning_rate', .001),
                momentum=getattr(task_conf, 'momentum', .9),
                decay_rate=getattr(task_conf, 'decay_rate', .9),
                decay_step=getattr(task_conf, 'decay_step', 10000),
                gradient_clipping=getattr(task_conf, 'gradient_clipping', 5.0),
                require_eval=getattr(task_conf, 'require_eval', True),
                ref=getattr(task_conf, 'ref', False),
                aux_task=task_conf.task != target_task,
                aux_lang=task_conf.language != target_lang,
            )
        else:
            raise ValueError('Unknown task {}'.format(task))
        tasks.append(task_obj)

    return tasks, {
        'lang_token_vocabs': lang_token_vocabs,
        'task_token_vocabs': task_label_vocabs,
        'components': components
    }






