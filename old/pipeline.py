import traceback

from task import build_tasks_from_file, MultiTask
from util import get_logger
from argparse import ArgumentParser

logger = get_logger(__name__)

arg_parser = ArgumentParser()

# arg_parser.add_argument('-d', '--device',
#                         type=int, default=0, help='GPU index')
# arg_parser.add_argument('-t', '--thread',
#                         type=int, default=5, help='Thread number')
arg_parser.add_argument('-c', '--config', help='Configuration file')

args = arg_parser.parse_args()

# torch.cuda.set_device(args.device)
# torch.set_num_threads(args.thread)
config_file = args.config

tasks, conf, _ = build_tasks_from_file(config_file, options=None)
multitask = MultiTask(tasks, eval_freq=conf.training.eval_freq)

try:
    for step in range(1, conf.training.max_step + 1):
        multitask.step()
except Exception:
    traceback.print_exc()
