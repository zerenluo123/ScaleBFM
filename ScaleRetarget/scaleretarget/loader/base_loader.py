import os
import glob
from enum import Enum
from pathlib import Path
from omegaconf import DictConfig, ListConfig
from loguru import logger

class Mode(Enum):
    file = 0
    dir = 1

class BaseLoader:
    def __init__(self, config: DictConfig):
        self.config = config
        self.format = self.config.data_format
        self.overwrite = self.config.get('overwrite', False)
        self.target_fps = self.config.get('target_fps', 30)
        self.output_dir = os.path.expanduser(self.config.get('output_dir', 'retargeted_dataset'))
        # Folders laid out like output_dir whose files count as done even with overwrite, e.g. clips taken
        # out of the dataset by hand, so that later runs do not convert them again.
        self.done_dirs = [os.path.expanduser(d) for d in (self.config.get('done_dirs', None) or [])]
        
    def load(self, data_path):
        self.data_path = data_path

        if os.path.isdir(self.data_path):
            self.mode = Mode.dir
            file_list = self.config.get('file_list', None)
            if file_list:
                self.data_list = self._read_file_lists(file_list)
            else:
                self.data_list = glob.glob(f"{self.data_path}/**/*{self.format}", recursive=True)
            exclude = self.config.get('exclude_content', None) or []
            if exclude:
                logger.info(f"[Loader] Removing motions with keywords: {list(exclude)}")
                self.data_list = [p for p in self.data_list if not any(c in p for c in exclude)]
            self.data_num = len(self.data_list)
        elif os.path.isfile(self.data_path):
            suffix = os.path.splitext(self.data_path)[-1]
            assert suffix == self.format, f"You are using the data loader for {self.format} format but the data you give is {suffix} format!"
            self.mode = Mode.file
            self.data_list = [self.data_path]
            self.data_num = 1
        else:
            raise Exception(f"Check your data path: {self.data_path}")

        protected = [self.data_path if self.mode == Mode.dir else os.path.dirname(self.data_path)]
        if self.config.get('read_only_root', None):
            protected.append(os.path.expanduser(self.config.read_only_root))
        output_root = os.path.realpath(self.output_dir)
        for root in protected:
            root = os.path.realpath(root)
            if os.path.commonpath([root, output_root]) == root:
                raise ValueError(f"output_dir {self.output_dir} is inside the read-only source {root}; choose a directory outside it.")
        
        self.done_num = sum(self._is_done(self._save_path(p)) for p in self.data_list)
        logger.info(f"[Loader] Total number of motions: {self.data_num}")
        if self.done_num:
            logger.warning(f"[Loader] {self.done_num} of {self.data_num} motions already have output (or are in done_dirs) and are skipped")

    def _is_done(self, save_path):
        rel = os.path.relpath(save_path, self.output_dir)
        if any(os.path.exists(os.path.join(d, rel)) for d in self.done_dirs):
            return True
        return os.path.exists(save_path) and not self.overwrite

    def _read_file_lists(self, file_list):
        """Text files with one motion per line, relative to data_path (or absolute)."""
        lists = list(file_list) if isinstance(file_list, (list, tuple, ListConfig)) else [file_list]
        data_list = []
        for list_path in lists:
            with open(os.path.expanduser(list_path)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data_list.append(line if os.path.isabs(line) else os.path.join(self.data_path, line))
        missing = [p for p in data_list if not os.path.isfile(p)]
        if missing:
            logger.warning(f"[Loader] {len(missing)} listed files do not exist and are skipped, e.g. {missing[0]}")
            data_list = [p for p in data_list if os.path.isfile(p)]
        logger.info(f"[Loader] Read {len(data_list)} motions from {lists}")
        return data_list

    def _save_path(self, sample_path):
        if self.mode == Mode.file:
            relative_path = os.path.basename(sample_path)
        else:
            data_dir_name = Path(self.data_path).resolve().name
            relative_path = os.path.join(
                data_dir_name,
                os.path.relpath(sample_path, self.data_path),
            )
        relative_path = os.path.splitext(relative_path)[0] + '.pkl'
        save_path = os.path.join(self.output_dir, relative_path)

        if os.path.abspath(save_path) == os.path.abspath(sample_path):
            raise ValueError(
                f"Output path resolves to the source file: {sample_path}. "
                "Choose a different output_dir."
            )
        return save_path

    def iter_paths(self):
        """Yield (save_path, sample_path) for motions still to be retargeted, without loading them."""
        for sample_path in self.data_list:
            save_path = self._save_path(sample_path)
            if self._is_done(save_path):
                logger.debug(f"[Loader] {save_path} already exists!")
                continue
            yield save_path, sample_path
        
    def __iter__(self):
        for save_path, sample_path in self.iter_paths():
            try:
                frames, extras = self._load_sample(sample_path)
            except Exception as e:
                logger.warning(f"[Loader] File {sample_path} is broken! Skip it for error {e}")
                continue
            yield save_path, frames, extras

    def _load_sample(self, sample_path: Path):
        raise NotImplementedError
