# Handles data splitting, and has the Patient
# class that holds a patient.
# The data-splitting takes into consideration keeping the data
# representative in terms of gender and age with regards to the
# whole population (all of the dataset in the data dir).

import os
import yaml
import statistics
import random
from .consts import DATA_DIR
from .config_reader import Config


class Patient:
    """Used to hold info about a Patient.
    More info is added later on in the other parts of the run."""

    @staticmethod
    def load_patients() -> list["Patient"]:
        """A list of all the patients that exist in the data dir."""
        patients = []
        for entry in os.listdir(DATA_DIR):
            full_path = os.path.join(DATA_DIR, entry)
            if os.path.isdir(full_path):
                patient = Patient(full_path)
                patients.append(patient)
        return patients

    @staticmethod
    def compute_age_avg_and_m_over_f_ratio(
        patients: list["Patient"],
    ) -> tuple[float, float]:
        """Well that's a mouthful."""

        if len(patients) == 0:
            return (float("nan"), float("nan"))

        age = statistics.mean(p.age for p in patients)

        genders = [p.gender.lower() for p in patients]
        m_count = genders.count("m")
        f_count = genders.count("f")
        if f_count == 0:
            ratio = float("inf")
        else:
            ratio = m_count / f_count

        return (age, ratio)

    def __init__(self, patient_dir):
        """
        Important attributes:
            - name
            - gender
            - age
            - edfs: {edf_file_1_path -> csv_file_1_path, edf_file_2_path -> csv_file_2_path, ...}
        """
        self.dir = patient_dir
        info_file = [f for f in os.listdir(patient_dir) if f.endswith("_info.txt")][0]
        info_path = os.path.join(patient_dir, info_file)

        info = {}
        with open(info_path, "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":", 1)
                    info[key.strip().lower()] = val.strip()

        self.name = info.get("human-readable name")
        self.gender = info.get("gender")
        self.age = int(info.get("age"))

        edf_files = [f for f in os.listdir(patient_dir) if f.endswith(".edf")]
        csv_files = [f for f in os.listdir(patient_dir) if f.endswith(".csv")]

        # This thing just makes sure they are sorted alphabetically (to ensure consistency for the same split and config)
        self.edfs = {
            edf_path: matched_csvs[0]
            for edf_path, matched_csvs in sorted(
                (
                    (
                        os.path.join(patient_dir, edf),
                        [
                            os.path.join(patient_dir, f)
                            for f in csv_files
                            if f.startswith(os.path.splitext(edf)[0])
                        ],
                    )
                    for edf in edf_files
                ),
                key=lambda x: x[0],
            )
        }


def _select_patients(
    patients: list[Patient],
    count: int,
    age_avg: float,
    m_over_f_ratio: float,
    tol: float | None,
) -> tuple[list[Patient], list[Patient]]:
    """Given a list of patients to select `count` patients
    from, this function returns a list of the remaining patients
    that have not been selected, and the ones selected. The randomly
    selected patients gender and age average is within `tol` of
    the given age average and male over female ratio"""

    if count == 0:
        return (patients, [])

    # Attempts to split the data
    for _ in range(len(patients)):
        selected = random.sample(patients, count)

        if tol is not None:
            age, ratio = Patient.compute_age_avg_and_m_over_f_ratio(selected)

            age_ok = abs(age - age_avg) / age_avg <= tol
            gender_ok = abs(ratio - m_over_f_ratio) / m_over_f_ratio <= tol

        if tol is None or (age_ok and gender_ok):
            remaining = [p for p in patients if p not in selected]
            return remaining, selected

    assert (
        False
    ), f"Unable to select {count} valid patients from {len(patients)}... Ratio: {ratio}, Age: {age}"


def split_dataset(cfg: Config, split_file_path: str | None) -> dict[str, list[Patient]]:
    """Splits the dataset into the train, val, and test splits. Either
    a new random split, or following the split given."""

    patients = Patient.load_patients()
    print(f"\nLoaded {len(patients)} patients.")

    splits = {"train": [], "val": [], "test": []}

    # If a split file is given, use it
    if split_file_path is not None:
        print(f"Using the provided datasplit in {split_file_path}")

        with open(split_file_path, "r") as f:
            split_data = yaml.safe_load(f)

        for split_name in splits.keys():
            if split_name in split_data:
                for patient_name in split_data[split_name]:
                    patient = next(
                        (p for p in patients if p.name == patient_name), None
                    )
                    assert (
                        patient is not None
                    ), f"Unable to find this patient: {patient_name}"

                    splits[split_name].append(patient)

        print("The given datasplit specifies:")
        print(f"\t{len(splits['train'])} patients for training.")
        print(f"\t{len(splits['val'])} patients for validation.")
        print(f"\t{len(splits['test'])} patients for testing.")

        cfg.add_info("Using the given data split.")
        cfg.add_info()

    # No split is given, create a new random one
    else:
        amount = cfg["split"]["amount"]
        if amount is None:
            amount = len(patients)
        assert amount <= len(
            patients
        ), f"You wanted to use {amount} Patients, but only {len(patients)} exist."

        train_amount = cfg["split"]["train"]
        val_amount = cfg["split"]["val"]
        test_amount = cfg["split"]["test"]

        if cfg["split"]["as_percentage"]:
            train_amount = round((train_amount / 100) * amount)
            val_amount = round((val_amount / 100) * amount)
            test_amount = round((test_amount / 100) * amount)
        cfg.add_info("Generating a new split.")
        cfg.add_info(f"Total patients: {len(patients)}.")
        cfg.add_info(f"Specified amount to be used: {amount}.")
        cfg.add_info()

        print(
            f"Trying to generate a new data split from {amount} patients ({amount / len(patients) * 100:.2f}% of all patients) with:"
        )
        print(
            f"\tTraining set: {train_amount} ({train_amount / amount * 100:.2f}% of used patients)"
        )
        print(
            f"\tValidation set: {val_amount} ({val_amount / amount * 100:.2f}% of used patients)"
        )
        print(
            f"\tTest set: {test_amount} ({test_amount / amount * 100:.2f}% of used patients)"
        )

        # Compute reference age avg and male/female ratio on full set
        age_avg, m_over_f_ratio = Patient.compute_age_avg_and_m_over_f_ratio(patients)
        cfg.add_info(f"Population age average: {age_avg}.")
        cfg.add_info(f"Population male over female ratio: {m_over_f_ratio}.")
        cfg.add_info()

        # Attempt to get the splits
        exception = None
        for _ in range(len(patients)):
            try:
                # Select count patients from the whole dataset
                _, remaining_patients = _select_patients(
                    patients, amount, age_avg, m_over_f_ratio, cfg["split"]["tol"]
                )

                # Train split
                remaining_patients, train_split = _select_patients(
                    remaining_patients,
                    train_amount,
                    age_avg,
                    m_over_f_ratio,
                    cfg["split"]["tol"],
                )

                # Validation split
                remaining_patients, val_split = _select_patients(
                    remaining_patients,
                    val_amount,
                    age_avg,
                    m_over_f_ratio,
                    cfg["split"]["tol"],
                )

                # Test split
                _, test_split = _select_patients(
                    remaining_patients,
                    test_amount,
                    age_avg,
                    m_over_f_ratio,
                    cfg["split"]["tol"],
                )

                exception = None
                break
            except Exception as e:
                exception = e

        if exception is not None:
            raise exception

        splits["train"] = train_split
        splits["val"] = val_split
        splits["test"] = test_split

    # Save the info about the split and the split it self
    cfg.add_info(
        f"Train amount, age avg, and m/f ratio: {len(splits['train'])} {Patient.compute_age_avg_and_m_over_f_ratio(splits['train'])}."
    )
    cfg.add_info(
        f"Val amount, age avg, and m/f ratio: {len(splits['val'])} {Patient.compute_age_avg_and_m_over_f_ratio(splits['val'])}."
    )
    cfg.add_info(
        f"Test amount, age avg, and m/f ratio: {len(splits['test'])} {Patient.compute_age_avg_and_m_over_f_ratio(splits['test'])}."
    )
    cfg.add_info()

    split_dict = {
        "train": [p.name for p in splits["train"]],
        "val": [p.name for p in splits["val"]],
        "test": [p.name for p in splits["test"]],
    }

    split_path = os.path.join(cfg.output, "data_split.yaml")
    with open(split_path, "w") as f:
        yaml.safe_dump(split_dict, f)

    print(f"The data split has been saved in: {split_path}")

    return splits
