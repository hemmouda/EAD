# Takes the result of data splitting and adds
# the `filtered_edfs` attribute to
# the patients. This attribute is a dict that maps
# the edf file path to another dict of channel names
# and their corresponding numpy filtered signal.

import pyedflib
import numpy as np
from scipy.signal import resample_poly, sosfiltfilt, butter, iirnotch, filtfilt
from tqdm import tqdm
from copy import deepcopy
from .config_reader import Config
from .data_splitting import Patient


def _filter_signal(cfg: Config, signal: np.ndarray, signal_freq: float) -> np.ndarray:
    """What actually filters."""

    # Go over the specified cleaning steps
    for step in cfg["clean"]["steps"]:

        if step == "downsample":
            new_freq = cfg["clean"]["new_frequency"]
            assert (
                new_freq <= signal_freq
            ), f"Can only down sample! {new_freq} {signal_freq}"
            signal = resample_poly(signal, new_freq, signal_freq)

        elif step == "butterworth":
            params = cfg["clean"]["butterworth"]
            sos = butter(
                params["order"],
                [params["high_pass"], params["low_pass"]],
                btype="band",
                fs=signal_freq,
                output="sos",
            )
            signal = sosfiltfilt(sos, signal)

        elif step == "fir":
            raise NotImplementedError("Sowwy.")

        elif step == "notch":
            params = cfg["clean"]["notch"]
            b, a = iirnotch(params["freq"], params["quality"], signal_freq)
            signal = filtfilt(b, a, signal)

        elif step == "clip":
            value = abs(cfg["clean"]["clipping_value"])
            signal = np.clip(signal, -value, value)

        else:
            assert False, f"Unknown cleaning step: {step}"

    return signal


def _filter_patient(cfg: Config, patient: Patient) -> None:
    """Mutates the given patient!"""

    filtered_edfs: dict[str, dict[str, np.ndarray]] = {}

    # For every edf file that the patient has
    for edf_file_path in patient.edfs.keys():
        filtered_edfs[edf_file_path] = {}

        # Load the edf file and get the signal labels
        f = pyedflib.EdfReader(edf_file_path)
        labels = f.getSignalLabels()

        # Get the channels (signals) of interest
        channels = cfg["channels"]
        if channels is None:
            channels = labels

        # Then filter every channel (signal)
        for channel in channels:
            assert (
                channel in labels
            ), f"This channel {channel} does not exist in {edf_file_path}!"
            index = labels.index(channel)
            signal = f.readSignal(index).copy()
            frequency = f.getSampleFrequency(index)
            filtered_signal = _filter_signal(cfg, signal, frequency)
            filtered_edfs[edf_file_path][channel] = filtered_signal

        # Close the edf file when you are done
        f.close()

    # And finally add the new attribute
    patient.filtered_edfs = filtered_edfs


def filter(cfg: Config, splits: dict[str, list[Patient]]) -> dict[str, list[Patient]]:
    """Returns the new splits with the necessary filtering and cleaning.
    The patients are copied, so it's fine if the same patient
    is used multiple times."""

    modified_splits = {}

    print(
        f"\nChannels of interest that will be filtered are: {'ALL_CHANNELS' if cfg['channels'] is None else cfg['channels']}",
        flush=True,
    )

    # For every split
    for split, patients in splits.items():
        print(f"Filtering the {split} split:", flush=True)

        # For every patient in that split
        modified_patients = []
        pbar = tqdm(patients)
        for patient in pbar:
            pbar.set_description(f"Processing {patient.name}")
            patient = deepcopy(patient)  # In case it's used multiple times
            _filter_patient(cfg, patient)
            modified_patients.append(patient)

        modified_splits[split] = modified_patients

    return modified_splits
