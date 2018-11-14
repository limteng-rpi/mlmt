import logging

# UNKNOWN_TOKEN = '$UNK$'
# UNKNOWN_TOKEN_INDEX = 1
# PADDING = '$PAD$'
# PADDING_INDEX = 0
# EMBED_START_IDX = 2
# CHAR_EMBED_START_IDX = 2

PAD = '<$PAD$>'
UNK = '<$UNK$>'
PAD_INDEX = 0
UNK_INDEX = 1
TOKEN_PADS = [
    (PAD, PAD_INDEX),
    (UNK, UNK_INDEX)
]
CHAR_PADS = [
    (PAD, PAD_INDEX),
    (UNK, UNK_INDEX)
]

EVAL_BATCH_SIZE = 200

LOGGING_LEVEL = logging.INFO

PENN_TREEBANK_BRACKETS = {
    '-LRB-': '(',
    '-RRB-': ')',
    '-LSB-': '[',
    '-RSB-': ']',
    '-LCB-': '{',
    '-RCB-': '}',
    '``': '"',
    '\'\'': '"',
    '/.': '.',
}