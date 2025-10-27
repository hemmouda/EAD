# Takes the result of labeling and modifies it
# so that data is split into chunks
# then all the patient data is aggregated
# and then balanced. The result is stored
# in x and y in the patient object.

import numpy as np
from tqdm import tqdm
from .config_reader import Config
from .data_splitting import Patient


# So that the balancing is deterministic
# for the same data split.
np.random.seed(42)


def _balance_patient(cfg: Config, patient: Patient) -> bool:
    """Balances a split-ed patients' data by under sampling.
    Balances regardless which class is the majority.
    Returns False if this patient shouldn't be used because
    he is "exclusively" just background data"""

    y = patient.y
    assert np.all(np.isin(np.unique(y), [0, 1])), "What other class have you added?"

    # Get the majority class per chunk
    chunks_labels = (y.mean(axis=1) >= 0.5).astype(int)

    # Then the indices of classes in the chunks
    idx_0 = np.where(chunks_labels == 0)[0]
    idx_1 = np.where(chunks_labels == 1)[0]

    # See which one is more dominant
    # and determine the indices of the
    # chunks that will be kept
    n0, n1 = len(idx_0), len(idx_1)

    # Some patient have "only" background or artifact.
    if n0 == 0 or n1 == 0:
        cfg.add_info(
            f"\nThis patient {patient.name} has \"exclusively\" just {'artifact' if n0 == 0 else 'background'} data. It thus cannot be balanced as that it will have no data left."
        )
        if n0 == 0:
            cfg.add_info(
                f"The patient will however be kept because it's artifact data."
            )
            return True
        else:
            cfg.add_info(
                f"The patient will however be discarded because it's background data. "
            )
            return False

    if n0 > n1:  # Background dominates
        keep_0 = np.random.choice(idx_0, size=n1, replace=False)
        keep_idx = np.concatenate([keep_0, idx_1])
    elif n1 > n0:  # Artifacts dominate (rare)
        keep_1 = np.random.choice(idx_1, size=n0, replace=False)
        keep_idx = np.concatenate([idx_0, keep_1])
    else:  # Already balanced (shiny rare)
        keep_idx = np.arange(len(y))

    # We could go with removing the chunks in
    # a descending order of how dominated by the majority
    # class it is. But since artifacts are so rare
    # if a chunk is an background chunk then it's
    # most likely all background

    patient.x = patient.x[keep_idx]
    patient.y = patient.y[keep_idx]
    return True


def _split_patient(cfg: Config, patient: Patient) -> bool:
    """Splits the patients' data into chunks and
    aggregates it into x and y.
    Returns whether this patient can be used (if
    he or she had enough to data length to split)"""

    CHUNK_LEN = cfg["label"]["chunk_duration"] * cfg["clean"]["new_frequency"]

    x = []
    y = []

    # Go over each individual channels' signal and label in all the edf files
    for edf_file_path, channels in patient.filtered_edfs.items():
        for signal, label in channels.values():
            assert signal.ndim == 1, f"What have you done wrong?"
            assert signal.shape == label.shape, f"Again, what have you done wrong?"

            # Calculate how many cuts can we have
            cuts_amount = len(signal) // CHUNK_LEN
            if cuts_amount == 0:
                cfg.add_info(
                    f"\nThis edf file {edf_file_path} is too short to be cut. {len(signal)} {CHUNK_LEN}"
                )
                break  # All channels have the same duration

            # And so the amount of data that we will use
            data_len = cuts_amount * CHUNK_LEN

            # And then reshape and append
            x.extend(signal[:data_len].reshape((cuts_amount, CHUNK_LEN)))
            y.extend(label[:data_len].reshape((cuts_amount, CHUNK_LEN)))

    # Check if this patient can be used
    if len(x) == 0:
        cfg.add_info(
            f"\nThis patient {patient.name} doesn't have long enough data to be used."
        )
        del patient.filtered_edfs
        return False

    # If so, store the new attributes
    patient.x = np.array(x)
    patient.y = np.array(y)
    assert patient.x.ndim == 2 and patient.x.shape == patient.y.shape, "?"
    del patient.filtered_edfs
    return True


def _store_distribution_info(
    cfg: Config, splits: dict[str, list[Patient]], before: bool
) -> None:
    """Is this before balancing or after?"""

    # Goes over the patients in the splits, aggregates their
    # individual label data, counts instances, and then aggregates
    # at the patients.

    def format_duration(seconds):
        if seconds < 60:
            return f"{seconds:.0f} second{'s' if seconds != 1 else ''}"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes:.0f} minute{'s' if minutes != 1 else ''}"
        else:
            hours = seconds // 3600
            return f"{hours:.0f} hour{'s' if hours != 1 else ''}"

    print(
        f"Extracting the data distribution {'before' if before else 'after'} the balancing...",
        flush=True,
    )

    f = cfg["clean"]["new_frequency"]

    cfg.add_info(
        f"\nArtifacts distribution {'before' if before else 'after'} balancing"
    )

    for split, patients in splits.items():
        cfg.add_info()

        if not before and not cfg["label"][split]["balance"]:
            cfg.add_info(
                f"The {split} split remains untouched (no balancing), but a minor amount of data is probably lost due to splitting into chunks."
            )  # 5 // 2 = 2 with a remained of 1
            continue

        cfg.add_info(f"Patients of the {split} split:")
        split_zeros = 0
        split_ones = 0
        for patient in patients:
            if before:
                label = []
                for channels in patient.filtered_edfs.values():
                    for _, labeled_signal in channels.values():
                        label.extend(labeled_signal)
                label = np.array(label)

            else:
                # Doesn't really matter how it's concatenated
                label = np.concatenate(patient.y)
            assert label.ndim == 1, "?"
            zeros = np.sum(label == 0)
            ones = np.sum(label == 1)
            total = zeros + ones
            assert total == len(label), f"Come again?"
            cfg.add_info(
                f"\t- {patient.name} has {format_duration(total/f)} of data in total ({total:_} data points), of which:"
            )
            cfg.add_info(
                f"\t\t- {zeros/total*100:.2f}% is background ({format_duration(zeros/f)} / {zeros:_} data points)"
            )
            cfg.add_info(
                f"\t\t- {ones/total*100:.2f}% is artifacts ({format_duration(ones/f)} / {ones:_} data points)"
            )
            cfg.add_info()
            split_zeros += zeros
            split_ones += ones

        split_total = split_zeros + split_ones
        cfg.add_info(
            f"- The {split} split itself has {format_duration(split_total/f)} of data in total ({split_total:_} data points), of which:"
        )
        # Empty splits
        if split_total != 0:
            cfg.add_info(
                f"\t- {split_zeros/split_total*100:.2f}% is background ({format_duration(split_zeros/f)} / {split_zeros:_} data points)"
            )
            cfg.add_info(
                f"\t- {split_ones/split_total*100:.2f}% is artifacts ({format_duration(split_ones/f)} / {split_ones:_} data points)"
            )


def balance(cfg: Config, splits: dict[str, list[Patient]]) -> None:
    """Splits the data into chunks and balances it if specified."""

    print()

    _store_distribution_info(cfg, splits, True)

    # For every split
    for split, patients in splits.items():
        balance = cfg["label"][split]["balance"]
        print(
            f"Splitting the data of the patients from the {split} split. Balancing? {balance}"
        )

        # For every patient in that split
        pbar = tqdm(patients[:])
        for patient in pbar:
            pbar.set_description(f"Processing {patient.name}")
            remove = False

            # Split the patients' data
            if _split_patient(cfg, patient):
                # And balance it if specified
                if balance:
                    if not _balance_patient(cfg, patient):
                        remove = True

            else:
                remove = True

            if remove:
                assert (
                    patients.count(patient) == 1
                ), f"This patient {patient.name} is used more than once in the {split} split!"
                patients.remove(patient)

    _store_distribution_info(cfg, splits, False)
