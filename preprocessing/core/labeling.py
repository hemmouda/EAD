# Takes the result of filtering and modifies the
# `filtered_edfs` attribute so that the channel names
# now point to a list of two elements: the original
# filtered signal, and the new annotated signal.
# The annotated signal is a zeros-numpy array
# of same length as the filtered signal
# with ones where the artifact of interest exists.

from tqdm import tqdm
import pandas as pd
import numpy as np
from .config_reader import Config
from .data_splitting import Patient


def _label_channel(
    cfg: Config,
    artifacts: None | list[str],
    csv_as_df: pd.DataFrame,
    channel_name: str,
    channel_length: int,
    csv_file_path: str,
) -> np.ndarray:
    """Returns a labeled signal of the channel"""

    labeled_signal = np.zeros(channel_length)

    # Get the rows of the channel
    rows = csv_as_df[csv_as_df["channel"] == channel_name]

    # Then only keep the ones of the artifacts we care about
    if artifacts is not None:
        assert len(artifacts) != 0, f"HUH?"
        rows = rows[rows["label"].isin(artifacts)]

    # Then extract the durations
    durations = rows[["start_time", "stop_time"]]

    # Finally, simply put a 1 where ever the duration maps to
    frequency = cfg["clean"]["new_frequency"]
    for _, row in durations.iterrows():
        start = row["start_time"]
        stop = row["stop_time"]
        start = round(start * frequency)
        stop = round(stop * frequency)

        # The start is never off bounds
        assert (
            start < channel_length
        ), f"This start time {row['start_time']} is out of bounds. {frequency} {channel_length} {channel_name} {artifacts} {csv_file_path}"

        # The stop in some cases is even 10 seconds off...
        if stop >= channel_length:
            if stop > channel_length:
                cfg.add_info(
                    f"This stop time {row['stop_time']} is out of bounds by {row['stop_time'] - (channel_length/frequency)} seconds. {frequency} {channel_length} {channel_name} {artifacts} {csv_file_path}"
                )
            stop = channel_length - 1

        labeled_signal[start : stop + 1] = 1

    return labeled_signal


def _label_patient(cfg: Config, artifacts: None | list[str], patient: Patient):
    """Labeling at the patient level."""

    # For every edf file of that patient
    for edf_file_path, csv_file_path in patient.edfs.items():

        # Load its corresponding csv file
        df = pd.read_csv(csv_file_path, comment="#")

        # And then for every channel of interest in that edf file
        labeled_signals: dict[str, np.ndarray] = {}  # channel -> labeled_signal
        for channel, filtered_signal in patient.filtered_edfs[edf_file_path].items():

            # Get its labeled signal
            labeled_signals[channel] = _label_channel(
                cfg, artifacts, df, channel, len(filtered_signal), csv_file_path
            )

        # Finally, map the edfs channels to the filtered signal as well as the labeled signal
        for channel, filtered_signal in list(
            patient.filtered_edfs[edf_file_path].items()
        ):
            patient.filtered_edfs[edf_file_path][channel] = [
                filtered_signal,
                labeled_signals[channel],
            ]


def label(cfg: Config, splits: dict[str, list[Patient]]) -> None:
    """Labels the splits."""

    print()

    # For every split
    for split, patients in splits.items():
        artifacts = cfg["label"][split]["artifact"]
        print(
            f"Artifacts of interest for the {split} split are: {'ALL_ARTIFACTS' if artifacts is None else artifacts}",
            flush=True,
        )

        # For every patient in that split
        pbar = tqdm(patients)
        for patient in pbar:
            pbar.set_description(f"Labeling {patient.name}")
            _label_patient(cfg, artifacts, patient)
