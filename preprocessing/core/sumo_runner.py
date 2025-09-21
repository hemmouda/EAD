# Temporally saves the data to run it with
# SUMO, output results, then delete it.


import os
import sys
import shutil
import subprocess
from .config_reader import Config


def run(cfg: Config, splits: dict) -> None:
    """Trains SUMO2 on the data and reports the results."""

    # Create variables needed
    raise AssertionError(f"TODO: FIX THIS TO USE THE CONST FILE.")  # Sowwy.
    SUMO_BASE = os.path.join(os.path.dirname(__file__), "sumo")
    SUMO_TRAIN = os.path.join(SUMO_BASE, "bin", "train.py")
    SUMO_INPUT_DIR = os.path.join(SUMO_BASE, "input")
    SUMO_CONFIG = os.path.join(SUMO_BASE, "config", "default.yaml")
    SUMO_OUTPUT_DIR = os.path.join(SUMO_BASE, "output")

    RUN_SUMO_DIR = os.path.join(cfg.output, "sumo_output")
    os.makedirs(RUN_SUMO_DIR)

    cfg.add_info(f"\nSUMO_BASE: {SUMO_BASE}")
    cfg.add_info(f"SUMO_TRAIN: {SUMO_TRAIN}")
    cfg.add_info(f"SUMO_INPUT_DIR: {SUMO_INPUT_DIR}")
    cfg.add_info(f"SUMO_CONFIG: {SUMO_CONFIG}")
    cfg.add_info(f"SUMO_OUTPUT_DIR: {SUMO_OUTPUT_DIR}")
    cfg.add_info(f"RUN_SUMO_DIR: {RUN_SUMO_DIR}")

    # Start training
    print(f"\nTraining SUMO2 on the data...", flush=True)
    path = os.path.join(cfg.output, "sumo_config.yaml")
    shutil.copy(SUMO_CONFIG, path)
    print(f"The used SUMO2 config has been copied to {path}\n")

    # Save the data
    pickled_data_path = cfg.save_splits(SUMO_INPUT_DIR, splits)
    # Clear the object (we ain't rich on RAM)
    for _, subject in splits.items():
        del subject
    splits.clear()

    # Call sumo train
    result = subprocess.run([sys.executable, SUMO_TRAIN, "-e", "default"])
    cfg.add_info(f"\nSUMO exit code: {result.returncode}")
    print(f"\nSUMO2 finished. Exit code: {result.returncode}")

    # Delete the data
    os.remove(pickled_data_path)

    # Move SUMO output results
    single_dir = True
    if (
        len(
            [
                item
                for item in os.listdir(SUMO_OUTPUT_DIR)
                if os.path.isdir(os.path.join(SUMO_OUTPUT_DIR, item))
            ]
        )
        > 1
    ):
        single_dir = False
        print(f"More than one result directory found in SUMO output dir!")
        cfg.add_info(f"\nMore than one result directory found in SUMO output dir!")
    # Move all regardless
    for item in os.listdir(SUMO_OUTPUT_DIR):
        src_path = os.path.join(SUMO_OUTPUT_DIR, item)
        dst_path = os.path.join(RUN_SUMO_DIR, item)
        if os.path.isdir(src_path):
            if single_dir:
                single_dir = dst_path
            shutil.move(src_path, dst_path)

    # If it was a single dir, output the F1 result
    if not isinstance(single_dir, bool):
        log_file_path = os.path.join(single_dir, os.path.basename(single_dir) + ".log")
        best_f1_mean = None

        with open(log_file_path, "r") as f:
            for line in f:
                if "metrics/val_f1_mean" in line:
                    # Extract the number after the colon and strip whitespace
                    f1_mean = float(line.split(":")[-1].strip())
                    if best_f1_mean is None or best_f1_mean < f1_mean:
                        best_f1_mean = f1_mean

        cfg.add_info(f"\nBest mean F1 score: {best_f1_mean}")
        print(f"Best mean F1 score: {best_f1_mean*100:.2f}%")
