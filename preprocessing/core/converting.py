# Takes the result of data balancing and splitting
# and converts it to SUMO Subjects

import sys
import os
from tqdm import tqdm
from .config_reader import Config
from .data_splitting import Patient


def convert(
    cfg: Config, splits: dict[str, list[Patient]]
) -> dict[str, list["Subject"]]:
    """Converts the splits into splits that can be used
    with SUMO2."""

    print(f"\nConverting the Patients into SUMO2 Subjects...")

    sys.path.append(os.path.join(os.path.dirname(__file__), "sumo"))
    from sumo.data import Subject

    new_splits = {}

    # For every split
    for split, patients in splits.items():
        print(f"Converting the {split} split:", flush=True)

        new_splits[split] = []

        # For every patient in that split
        pbar = tqdm(patients)
        for patient in pbar:
            pbar.set_description(f"Converting {patient.name}")
            new_splits[split].append(Subject(patient.x, patient.y, 0, patient.name))

    return new_splits
