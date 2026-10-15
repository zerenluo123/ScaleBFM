import os
import joblib
from hydra.utils import instantiate
from loguru import logger

_retargeter = None
_formatter = None
_loader = None

def initialize_worker(retargeter_config, formatter_config, loader_config=None):
    global _retargeter, _formatter, _loader
    _retargeter = instantiate(retargeter_config)
    # Per-clip progress bars from many workers only interleave on the shared terminal.
    _retargeter.show_progress = False
    _formatter = instantiate(formatter_config)
    if loader_config is not None:
        _loader = instantiate(loader_config)


def process_single_item(args):
    if len(args) == 2:
        # (save_path, sample_path): the worker parses the file itself.
        save_path, sample_path = args
        if _loader is None:
            raise RuntimeError("Worker loader was not initialized")
        frames, extras = _loader._load_sample(sample_path)
    else:
        save_path, frames, extras = args
    if _retargeter is None or _formatter is None:
        raise RuntimeError("Worker models were not initialized")
            
    _retargeter.update(extras)
    qpos_list = _retargeter.retarget(frames)
    formatted_results = _formatter.format(qpos_list, extras)
            
    dump_atomic(formatted_results, save_path)
    logger.debug(f"Retargeted motion saved to {save_path}")

    return save_path


def dump_atomic(data, save_path):
    """Write to a temporary file and rename, so an interrupted run never leaves a truncated pkl
    that later runs would skip as already converted."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp_path = f"{save_path}.{os.getpid()}.tmp"
    joblib.dump(data, tmp_path)
    os.replace(tmp_path, save_path)


def producer(data_loader, queue, load_in_worker=False):
    try:
        items = data_loader.iter_paths() if load_in_worker else data_loader
        for item in items:
            # The bounded queue blocks here and provides backpressure.
            queue.put(item)
    finally:
        queue.put(None)
