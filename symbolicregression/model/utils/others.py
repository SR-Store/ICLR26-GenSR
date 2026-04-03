import os
import torch
import random
import logging
import numpy as np

logger = logging.getLogger('my.utils.others')


def set_seed(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f'set seed to {seed}')


def set_proctitle(title):
    try:
        from setproctitle import setproctitle
        setproctitle(title)
    except ImportError:
        logger.debug('package `setproctitle\' not found, ignore setting process title')


def set_signal():
    import signal
    def handler(signum, frame): raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

