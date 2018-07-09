import json
import logging
import constant as C
import conlleval


def get_logger(name, level=C.LOGGING_LEVEL, log_file=None):
    """Get a logger by name.

    :param name: Logger name (usu. __name__).
    :param level: Logging level (default=logging.INFO).
    """
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler())
    if log_file:
        logger.addHandler(logging.FileHandler(log_file, encoding='utf-8'))
    logger.setLevel(level)
    return logger


def evaluate(results, idx_token, idx_label, writer=None):
    """Evaluate prediction results.

    :param results: A List of which each item is a tuple
        (predictions, gold labels, sequence lengths, tokens) of a batch.
    :param idx_token: Index to token dictionary.
    :param idx_label: Index to label dictionary.
    :param writer: An object (file object) with a write() function. Extra output.
    :return: F-score, precision, and recall.
    """
    # b: batch, s: sequence
    outputs = []
    for preds_b, golds_b, len_b, tokens_b in results:
        for preds_s, golds_s, len_s, tokens_s in zip(preds_b, golds_b, len_b, tokens_b):
            l = int(len_s.data[0])
            preds_s = preds_s.data.tolist()[:l]
            golds_s = golds_s.data.tolist()[:l]
            tokens_s = tokens_s.data.tolist()[:l]
            for p, g, t in zip(preds_s, golds_s, tokens_s):
                token = idx_token.get(t, C.UNKNOWN_TOKEN)
                outputs.append('{} {} {}'.format(
                    token, idx_label[g], idx_label[p]))
            outputs.append('')
    counts = conlleval.evaluate(results)
    overall, by_type = conlleval.metrics(counts)
    conlleval.report(counts)
    if writer:
        conlleval.report(writer)
    return overall.fscore, overall.prec, overall.rec


class Config(dict):

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        __getattr__ = dict.__getitem__

        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        v = Config(v)
                    if isinstance(v, list):
                        v = [Config(x) if isinstance(x, dict) else x for x in v]
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Config, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Config, self).__delitem__(key)
        del self.__dict__[key]

    def set_dict(self, dict_obj):
        for k, v in dict_obj.items():
            if isinstance(v, dict):
                v = Config(v)
            self[k] = v

    def update(self, dict_obj):
        for k, v in dict_obj.items():
            if isinstance(v, dict):
                v = Config(v)
            if isinstance(v, list):
                v = [Config(x) if isinstance(x, dict) else x for x in v]
            self[k] = v

    def clone(self):
        return Config(dict(self))

    @staticmethod
    def read(path):
        """Read configuration from JSON format file.

        :param path: Path to the configuration file.
        :return: Config object.
        """
        # logger.info('loading configuration from {}'.format(path))
        json_obj = json.load(open(path, 'r', encoding='utf-8'))
        return Config(json_obj)

    def update_value(self, keys, value):
        keys = keys.split('.')
        assert len(keys) > 0

        tgt = self
        for k in keys[:-1]:
            try:
                tgt = tgt[int(k)]
            except Exception:
                tgt = tgt[k]
        tgt[keys[-1]] = value