import argparse
import os
import traceback
from datetime import datetime
from core.config_reader import Config


def file_path(path):
    if path is None:
        return None
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"File '{path}' does not exist.")
    return path


parser = argparse.ArgumentParser(
    description="Preprocesses the molded TUAR dataset according to the specified config file, and runs it with SUMO2.\nSaves the configs used but deletes the data once the training is done.",
)

parser.add_argument(
    "-c",
    "--config",
    help="Path to the YAML config file (default: config.yaml)",
    type=file_path,
    default="config.yaml",
)


parser.add_argument(
    "-v",
    "--visualize",
    help="Stop after processing and labeling the data and save it because I want to visualize it.",
    action="store_true",
)


parser.add_argument(
    "-s",
    "--split",
    help="Path to the YAML file containing the datasplit to use, or None to get a new datasplit (default: None)",
    type=file_path,
    default=None,
)

parser.add_argument(
    "-n",
    "--no-sumo",
    help="Don't train SUMO2, just preprocess the data and output it. (default: False)",
    action="store_true",
)

# Parse the args
args = parser.parse_args()

# Parse and validate the config file
cfg = Config(args.config)

# Create the run output file
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)
run_dir = os.path.join(output_dir, datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S"))
os.makedirs(run_dir)
cfg.store("output", run_dir)

print(f"All results will be saved in: {run_dir}")
cfg.save_config()


################################################################################

try:
    # Data splitting
    from core.data_splitting import split_dataset

    splits = split_dataset(cfg, args.split)

    # Filtering and cleaning
    from core.filtering import filter

    splits = filter(cfg, splits)

    # Labeling the data
    from core.labeling import label

    label(cfg, splits)

    if args.visualize:
        path = cfg.save_splits(cfg.output, splits)
        print(
            f"\nThe Patients pickled dict have been saved for visualization in {path}"
        )
        print(f"The down-sampling frequency is {cfg['clean']['new_frequency']}")

    else:

        # Balancing and splitting the data into chunks
        from core.balancing import balance

        balance(cfg, splits)

        # Converting the data into SUMO Subjects
        from core.converting import convert

        splits = convert(cfg, splits)

        # Either train or just save the data
        if args.no_sumo:
            path = cfg.save_splits(cfg.output, splits)
            print(f"\nThe Subjects pickled dict have been saved to {path}")

        else:
            from core.sumo_runner import run

            run(cfg, splits)


except Exception as e:
    traceback.print_exc()

finally:
    path = cfg.save_run_info()
    print(f"\nRun info (if any) were stored in: {path}")
