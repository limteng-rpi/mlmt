*The code is being refactored :)*

## Requirements
Python 3.5+
Pytorch 0.3.1
tqdm (used to display training progress)

## Architecture
![Overall architecture](https://github.com/limteng-rpi/mlmt/blob/master/image/framework.png)
**Figure**: Multi-lingual Multi-task Architecture

## Pre-trained word embeddings

Pre-trained word embeddings for English, Dutch, Spanish, Russian, and Chechen can be found at [this page](http://www.limteng.com/research/2018/05/14/pretrained-word-embeddings.html).

## Single-task Mono-lingual Model

Train a new model:

```
python train_single.py --train <PATH/TO/THE/TRAINING/FILE> --dev <PATH/TO/THE/DEV/FILE>
  --test <PATH/TO/THE/TEST/FILE> --log <LOG/DIRECTORY> --model <MODEL/DIRECTORY>
  --max_epoch 50 --embedding <PATH/TO/THE/PRETRAINED/EMBEDDING/FILE> --embed_skip_first
  --word_embed_dim 100 --char_embed_dim 50
```

Evalute the trained model:

```
python eval_single.py --model <PATH/TO/THE/MODEL/FILE> --file <PATH/TO/THE/DATA/FILE>
  --log <LOG/DIRECTORY>
```

## Multi-task Model

In my original code, I use the `build_tasks_from_file` function in `task.py` to build the whole architecture from a configuration file (see the `Configuration` section). `pipeline.py` shows how to use this function.

I'm writing new scripts similar to `train_single.py` and `eval_single.py`.

## Configuration

For complete configuration, see `example_config.json`.

```json
{
  "training": {
    "eval_freq": 1000,                          # Evaluate the model every <eval_freq> global step
    "max_step": 50000,                          # Maximun training step
    "gpu": true                                 # Use GPU
  },
  "datasets": [                                 # A list of data sets
    {
      "name": "nld_ner",                        # Data set name
      "language": "nld",                        # Data set language 
      "type": "sequence",                       # Data set type; 'sequence' is the only supported value though
      "task": "ner",                            # Task (identical to the 'task' value of the corresponding task)
      "parser": {                               # Data set parser
        "format": "conll",                      # File format
        "token_col": 0,                         # Token column index
        "label_col": 1                          # Label column index
      },
      "sample": 200,                            # Sample number (optional): 'all', int, or float
      "batch_size": 19,                         # Batch size
      "files": {
        "train": "/PATH/TO/ned.train.bioes",    # Path to the training set 
        "dev": "/PATH/TO/ned.testa.bioes",      # Path to the dev set
        "test": "/PATH/TO/ned.testb.bioes"      # Path to the test set (optional)
      }
    },
    ...
  ],
  "tasks": [
    {
      "name": "Dutch NER",                      # Task name
      "language": "nld",                        # Task language
      "task": "ner",                            # Task
      "model": {                                # Components can be shared and are configured in 'components'. Just 
                                                # put their names here.
        "model": "lstm_crf",                    # Model type
        "word_embed": "nld_word_embed",         # Word embedding
        "char_embed": "char_embed",             # Character embedding
        "crf": "ner_crf",                       # CRF layer
        "lstm": "lstm",                         # LSTM layer
        "univ_layer": "ner_univ_linear",        # Universal/shared linear layer
        "spec_layer": "ner_nld_linear",         # Language-specific linear layer
        "embed_dropout": 0.0,                   # Embedding dropout probability
        "lstm_dropout": 0.6,                    # LSTM output dropout probability
        "linear_dropout": 0.0,                  # Linear layer output dropout probability
        "use_char_embedding": true,             # Use character embeddings
        "char_highway": "char_highway"          # Highway networks for character embeddings
      },
      "dataset": "nld_ner",                     # Data set name
      "learning_rate": 0.02,                    # Learning rate
      "decay_rate": 0.9,                        # Decay rate
      "decay_step": 10000,                      # Decay step
      "ref": true                               # Is the target task
    },
    ...
  ],
  "components": [
    {
      "name": "eng_word_embed",
      "model": "embedding",
      "language": "eng",
      "file": "/PATH/TO/enwiki.cbow.50d.txt",
      "stats": true,
      "padding": 2,
      "trainable": true,
      "allow_gpu": false,
      "dimension": 50,
      "padding_idx": 0,
      "sparse": true
    },
    {
      "name": "nld_word_embed",
      "model": "embedding",
      "language": "nld",
      "file": "/PATH/TO/nlwiki.cbow.50d.txt",
      "stats": true,
      "padding": 2,
      "trainable": true,
      "allow_gpu": false,
      "dimension": 50,
      "padding_idx": 0,
      "sparse": true
    },
    {
      "name": "char_embed",
      "model": "char_cnn",
      "dimension": 50,
      "filters": [[2, 20], [3, 20], [4, 20]]
    },
    {
      "name": "lstm",
      "model": "lstm",
      "hidden_size": 171,
      "bidirectional": true,
      "forget_bias": 1.0,
      "batch_first": true,
      "dropout": 0.0                            # Because we use a 1-layer LSTM. This value doesn't have any effect.
    },
    {
      "name": "ner_crf",
      "model": "crf"
    },
    {
      "name": "pos_crf",
      "model": "crf"
    },
    {
      "name": "ner_univ_linear",
      "model": "linear",
      "position": "output"
    },
    {
      "name": "ner_eng_linear",
      "model": "linear",
      "position": "output"
    },
    {
      "name": "ner_nld_linear",
      "model": "linear",
      "position": "output"
    },
    {
      "name": "pos_univ_linear",
      "model": "linear",
      "position": "output"
    },
    {
      "name": "pos_eng_linear",
      "model": "linear",
      "position": "output"
    },
    {
      "name": "pos_nld_linear",
      "model": "linear",
      "position": "output"
    },
    {
      "name": "char_highway",
      "model": "highway",
      "position": "char",
      "num_layers": 2,
      "activation": "selu"
    }
  ]
}
```

## Reference

- Lin, Y., Yang, S., Stoyanov, V., Ji, H. (2018) *A Multi-lingual Multi-task Architecture for Low-resource Sequence Labeling*. Proceedings of The 56th Annual Meeting of the Association for Computational Linguistics. \[[pdf](http://nlp.cs.rpi.edu/paper/multilingualmultitask.pdf)\]

```
@inproceedings{ying2018multi,
    title     = {A Multi-lingual Multi-task Architecture for Low-resource Sequence Labeling},
    author    = {Ying Lin and Shengqi Yang and Veselin Stoyanov and Heng Ji},
    booktitle = {Proceedings of The 56th Annual Meeting of the Association for Computational Linguistics (ACL2018)},
    year      = {2018}
}
```
