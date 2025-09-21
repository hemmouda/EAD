# Just where the config data is stored.
# Plus the args / info needed by the current run...

import yaml
import os
import pickle
from copy import deepcopy


class Config:
    def __init__(self, filepath):
        self.filepath = filepath
        self.info: list[str] = []
        self.config = None
        self.load_config()
        self.apply_inheritance()
        self.validate_config()

    def store(self, key, value):
        """Store something in the config. E.g. output dir."""
        self.__dict__[key] = value

    def add_info(self, any=None):
        if any is None:
            any = ""
        self.info.append(str(any))

    def load_config(self):
        with open(self.filepath, "r") as f:
            self.config = yaml.safe_load(f)

    def apply_inheritance(self):
        label = self.config["label"]
        train_cfg = label["train"]
        for split_name in ["val", "test"]:
            split_cfg = label.get(split_name, {})
            # Inherit missing keys from train
            for key, value in train_cfg.items():
                split_cfg.setdefault(key, deepcopy(value))
            label[split_name] = split_cfg

    def validate_config(self):
        split = self.config["split"]

        # Validate amount
        amount = split["amount"]
        if amount is not None and (not isinstance(amount, int) or amount <= 0):
            raise ValueError("split.amount must be a positive integer or None")

        # Validate as_percentage
        as_percentage = split["as_percentage"]
        if not isinstance(as_percentage, bool):
            raise ValueError("split.as_percentage must be boolean")

        # Validate train/val/test
        train = split["train"]
        val = split["val"]
        test = split["test"]
        for name, value in [("train", train), ("val", val), ("test", test)]:
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"split.{name} must be a non-negative number")

        total = train + val + test
        if as_percentage:
            if total > 100:
                raise ValueError("Sum of split percentages cannot exceed 100")
        else:
            if amount is None:
                raise ValueError("split.amount must be set when as_percentage is False")
            if total > amount:
                raise ValueError("Sum of split amounts cannot exceed split.amount")

        # Validate tolerance
        tol = split["tol"]
        if tol is not None and (not isinstance(tol, (int, float)) or tol < 0):
            raise ValueError("split.tol must be a non-negative number or None")

        # Validate channels
        channels = self.config["channels"]
        if channels is not None and not isinstance(channels, list):
            raise ValueError("channels must be a list or None")

        # Validate cleaning steps
        clean = self.config["clean"]
        steps = clean["steps"]
        if not isinstance(steps, list) or "downsample" not in steps:
            raise ValueError(
                "clean.steps must be a list that must contain `downsample`"
            )

        # Validate new_frequency
        new_freq = clean["new_frequency"]
        if not isinstance(new_freq, int) or new_freq <= 0:
            raise ValueError("clean.new_frequency must be a positive integer")

        # Validate clipping_value
        clip_value = clean["clipping_value"]
        if not isinstance(clip_value, (float, int)) or clip_value <= 0:
            raise ValueError("clean.clipping_value must be a positive integer")

        # Validate label chunk_duration
        label = self.config["label"]
        chunk_duration = label["chunk_duration"]
        if not isinstance(chunk_duration, (int, float)) or chunk_duration <= 0:
            raise ValueError("label.chunk_duration must be a positive number")

        # Validate artifact and balance
        for split_name in ["train", "val", "test"]:
            artifact = label[split_name]["artifact"]
            if artifact is not None and not isinstance(artifact, list):
                raise ValueError("label.{split_name}.artifact must be a list or None")

            balance = label[split_name]["balance"]
            if not isinstance(balance, bool):
                raise ValueError(f"label.{split_name}.balance must be boolean")

    def save_config(self):
        """Save the current configuration to 'config.yaml' in the output dir.
        And returns the full path."""
        file_path = os.path.join(self.output, "config.yaml")
        with open(file_path, "w") as f:
            yaml.safe_dump(self.config, f)
        return file_path

    def save_run_info(self):
        """Save the gathered run info 'run_info.txt' in the output dir.
        And returns the full path."""
        file_path = os.path.join(self.output, "run_info.txt")
        with open(file_path, "w") as f:
            f.write("\n".join(self.info))
        return file_path

    def save_splits(self, dir, splits):
        """Saves the given splits into a `subjects.pickle` file
        in the output dir and returns the full path."""
        file_path = os.path.join(dir, "subjects.pickle")
        with open(file_path, "wb") as f:
            pickle.dump(splits, f)
        return file_path

    def __getitem__(self, key):
        return self.config[key]

    def __contains__(self, key):
        return key in self.config
