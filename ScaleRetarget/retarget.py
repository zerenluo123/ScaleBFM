import os
import sys
import hydra
import multiprocessing as mp
import threading
from queue import Queue
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from loguru import logger

from scaleretarget.utils.multi_process import (
    dump_atomic,
    initialize_worker,
    process_single_item,
    producer,
)

@hydra.main(
    version_base = None,
    config_path = "./config",
    config_name = "base"
)
def main(config: DictConfig) -> None:
    # add logger
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "retarget.log")
    logger.remove()
    logger.add(hydra_log_path, level='DEBUG')
    console_log_level = os.environ.get('LOGURU_LEVEL', "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logger.info(f"Robot: {config.robot.robot_type}")
    logger.info(f"Data Loader: {config.loader.config.loader_type} loader")
    logger.info(f'Retargeter: {config.retargeter.config.retargeter_type} retargeter')
    logger.info(f"Formatter: {config.formatter.config.formatter_type} formatter")
    
    assert os.path.exists(config.data_path)
    logger.info(f"Retargeting data from {config.data_path}")

    # Load data and align format for retargter
    data_loader = instantiate(config.loader)
    data_loader.load(config.data_path) # lazy loading
    
    # Perform retargeting
    if config.multi_process:
        assert not config.record_video, f"record_video should be set to False when using multi_process!"
        total_motion_num = data_loader.data_num - data_loader.done_num
        configured_workers = config.get("num_workers")
        worker_count = min(configured_workers or mp.cpu_count(), total_motion_num)
        if worker_count < 1:
            logger.info("No motions to retarget.")
            return
        logger.info(f"Using {worker_count} worker processes")
        # Parsing large files in the producer thread serializes the run, so workers can load them instead.
        load_in_worker = config.loader.config.get("load_in_worker", False)
        ctx = mp.get_context('spawn')
        queue = Queue(maxsize=worker_count)
        producer_thread = threading.Thread(
            target=producer,
            args=(data_loader, queue, load_in_worker),
            daemon=True,
        )
        producer_thread.start()
    
        failed_paths = []
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=ctx,
            initializer=initialize_worker,
            initargs=(config.retargeter, config.formatter, config.loader if load_in_worker else None),
        ) as executor:
            futures = {}
            active = True
            success_num = 0

            while active or futures:
                while len(futures) < worker_count * 2 and active:
                    item = queue.get()
                    if item is None:
                        active = False
                        break
                    futures[executor.submit(process_single_item, item)] = item[-1] if load_in_worker else item[0]

                if not futures:
                    continue

                completed, _ = wait(
                    futures,
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    item_path = futures.pop(future)
                    try:
                        future.result()
                        success_num += 1
                        logger.info(f"Success/Total: {success_num}/{total_motion_num}")
                    except Exception as e:
                        failed_paths.append(item_path)
                        logger.exception(f"Processing error for {item_path}: {e}")

        producer_thread.join()
        if failed_paths:
            failed_log = os.path.join(HydraConfig.get().runtime.output_dir, "failed.txt")
            with open(failed_log, "w") as f:
                f.write("\n".join(failed_paths) + "\n")
            logger.warning(f"{len(failed_paths)} motions failed; listed in {failed_log}")

    else:
        # Initialize retargeter and saving formatter
        retargeter = instantiate(config.retargeter)
        formatter = instantiate(config.formatter)

        for save_path, frames, extras in data_loader:
            retargeter.update(extras)
            qpos_list = retargeter.retarget(frames)
            formatted_results = formatter.format(qpos_list, extras)

            dump_atomic(formatted_results, save_path)
            logger.info(f"Retargeted motion saved to {save_path}")

        retargeter.finish()

    logger.info(f"Retargeting Done!")

        
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
