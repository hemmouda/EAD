# Molds the TUAR dataset so that it can be used with the given preprocessing pipeline.
# It does in three steps:
# -1: It first groups the data patient wise instead of montage wise, while giving the patient human-readable name using the name-mapping file
# -2: It removes all seizure data and patient that only have seizure data.
# -3: It transforms the EDF file from whatever montage they were in, into the TCP montage

import argparse
import os
import shutil
import csv
import glob
import pyedflib
import math
import numpy as np
from tqdm import tqdm


MAPPINGS_CSV_FILE = "name_mappings.csv"


def folder_path(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"Folder '{path}' does not exist.")
    return path


parser = argparse.ArgumentParser(
    description="Molds the TUAR dataset so that it can be used with the preprocessing pipeline (creates a copy, does not delete)."
)

parser.add_argument(
    "-i",
    "--input",
    help="Path to the `v3.0.1` folder of the TUAR dataset that contains the `edf` folder with all the data and the `DOCS` folder with the montages' info.",
    type=folder_path,
    required=True,
)

parser.add_argument(
    "-o",
    "--output",
    help="Where to save the molded dataset. Default is `molded_dataset`.",
    default="molded_dataset",
)

args = parser.parse_args()
data_root = args.input
output_dir = args.output
os.makedirs(output_dir, exist_ok=True)


# -------------------- STEP 1 --------------------
# Group the dataset patient-wise (instead of montage-wise)
# using the human-readable name mapping file.
# Also create a patient info file with code name, human name, age, and gender.

print("Grouping the data patient-wise instead of montage-wise...")

# Read patient name mappings
mappings = []
with open(MAPPINGS_CSV_FILE, "r", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        code_name = row["Original code-name"].strip()
        human_name = row["New human-readable name"].strip()
        gender = row["gender"].strip()
        age = row["age"].strip()
        mappings.append((code_name, human_name, gender, age))

# Get montage folder paths inside the EDF folder
edf_root = os.path.join(data_root, "edf")
montage_folders = [
    os.path.join(edf_root, d)
    for d in os.listdir(edf_root)
    if os.path.isdir(os.path.join(edf_root, d))
]

# Copy over patient files and create info file
pbar = tqdm(mappings)
for code_name, human_name, gender, age in pbar:
    pbar.set_description(f"Copying {code_name:^9} -> {human_name:^10}")
    patient_found = False
    patient_output_path = os.path.join(output_dir, human_name)
    os.makedirs(patient_output_path, exist_ok=True)

    # Search and copy files from montage folders
    for montage_folder in montage_folders:
        for file_name in os.listdir(montage_folder):
            if file_name.startswith(code_name):
                src_path = os.path.join(montage_folder, file_name)
                dst_path = os.path.join(patient_output_path, file_name)
                shutil.copy2(src_path, dst_path)
                patient_found = True

    if not patient_found:
        raise Exception(f"Patient '{code_name}' not found in any montage folder.")

    # Create patient info file
    info_file = os.path.join(patient_output_path, f"{code_name}_info.txt")
    with open(info_file, "w") as f:
        f.write(f"code name: {code_name}\n")
        f.write(f"human name: {human_name}\n")
        f.write(f"age: {age}\n")
        f.write(f"gender: {gender}\n")


# -------------------- STEP 2 --------------------
# Remove all seizure sessions and patients that only have seizure data.

print("\nRemoving seizure data...")

affected_patients = []
deleted_patients = []

for patient in tqdm(os.listdir(output_dir), desc="Processing patients"):
    patient_path = os.path.join(output_dir, patient)
    if not os.path.isdir(patient_path):
        continue

    # Find seizure files
    seiz_files = glob.glob(os.path.join(patient_path, "*_seiz.*"))
    if seiz_files:
        affected_patients.append(patient)
        base_sessions_to_delete = set()

        # Identify base session names
        for seiz_file in seiz_files:
            filename = os.path.basename(seiz_file)
            base_name = filename.replace("_seiz", "").rsplit(".", 1)[0]
            base_sessions_to_delete.add(base_name)

        # Delete base and seizure files
        for base_name in base_sessions_to_delete:
            for ext in [".csv", ".edf"]:
                file_to_delete = os.path.join(patient_path, f"{base_name}{ext}")
                if os.path.exists(file_to_delete):
                    os.remove(file_to_delete)
            # Also remove the seizure CSV if it exists
            seiz_file_path = os.path.join(patient_path, f"{base_name}_seiz.csv")
            if os.path.exists(seiz_file_path):
                os.remove(seiz_file_path)

        # Check if patient dir has no remaining .csv or .edf files
        remaining_data_files = [
            f
            for f in os.listdir(patient_path)
            if f.endswith(".csv") or f.endswith(".edf")
        ]
        if not remaining_data_files:
            shutil.rmtree(patient_path)
            affected_patients.remove(patient)
            deleted_patients.append(patient)

print("Affected patients:", affected_patients)
print("Deleted patients:", deleted_patients)

# Affected patients: ['ulrich', 'joanna', 'ian', 'oscar', 'martin', 'samuel', 'amelia', 'leo', 'owen', 'wendy', 'natalie']
# Deleted patients: ['renate', 'victor', 'gabriella', 'thea', 'elena', 'freya', 'theresa', 'daniel', 'leonard', 'kurt', 'nadine', 'vincent', 'hannah', 'brian', 'martha', 'heidi', 'ethan']


# -------------------- STEP 3 --------------------
# Transform all EDF files into the TCP montage.


def create_new_channel(
    edf_reader, channel_name: str, channel_composition: tuple[str, str]
) -> tuple:
    """Create the specified new channel from the composition, e.g. creates `FP1-F7` from ('EEG FP1-REF', 'EEG F7-REF')."""
    signal_labels = edf_reader.getSignalLabels()
    first, second = channel_composition
    assert (
        first in signal_labels and second in signal_labels
    ), f"Channels not found in EDF file."
    first_idx = signal_labels.index(first)
    second_idx = signal_labels.index(second)

    signal = edf_reader.readSignal(first_idx) - edf_reader.readSignal(second_idx)
    fh = edf_reader.getSignalHeader(first_idx).copy()  # First header
    sh = edf_reader.getSignalHeader(second_idx).copy()  # Second header
    fh.pop("label", None)
    sh.pop("label", None)
    assert fh == sh, f"The two channels differ in header for {channel_composition}"

    fh.pop("sample_rate", None)  # Deprecated, sample_frequency is used instead
    fh.pop(
        "prefilter", None
    )  # Not needed. I don't recall why exactly, maybe it was simply not set
    fh.pop("transducer", None)  # Same
    fh["label"] = channel_name
    fh["physical_min"] = math.floor(min(signal))
    fh["physical_max"] = math.ceil(max(signal))
    return (fh, signal)


def transform_edf_file(edf_file_path: str, montage: dict) -> None:
    """Overrides the given EDF file with a new EDF file that uses the specified montage."""
    headers, signals = [], []
    edf_reader = pyedflib.EdfReader(edf_file_path)

    for channel_name, channel_comp in montage.items():
        header, signal = create_new_channel(edf_reader, channel_name, channel_comp)
        headers.append(header)
        signals.append(signal)

    edf_reader._close()
    os.remove(edf_file_path)

    edf_writer = pyedflib.EdfWriter(
        edf_file_path, len(headers), file_type=pyedflib.FILETYPE_EDFPLUS
    )
    edf_writer.setSignalHeaders(headers)
    edf_writer.writeSamples(signals)
    edf_writer.close()


def load_montages(docs_path: str) -> dict:
    """Loads all montage files from DOCS and returns a dict of montage definitions."""
    montages = {}
    for file_name in os.listdir(docs_path):
        if not file_name.endswith("montage.txt"):
            continue
        montage = {}
        with open(os.path.join(docs_path, file_name), "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("montage ="):
                    parts = line.split(",")
                    name_desc = parts[1].split(":")
                    assert len(name_desc) == 2, f"Unexpected format in {docs_path}"
                    name = name_desc[0].strip()
                    if name == "EKG":
                        continue
                    channels = [ch.strip() for ch in name_desc[1].split("--")]
                    assert len(channels) == 2, f"Unexpected format in {docs_path}"
                    montage[name] = tuple(channels)
        montages[file_name] = montage
    return montages


def get_montage_for_file(edf_file_path: str, montages: dict) -> dict:
    """Finds which montage to use for a given EDF file based on its CSV metadata."""
    csv_file_path = edf_file_path.replace(".edf", ".csv")
    with open(csv_file_path, "r") as f:
        for line in f:
            if line.startswith("# montage_file"):
                montage_name = line.split("/")[-1].strip()
                assert montage_name in montages, f"Unknown montage {montage_name}"
                return montages[montage_name]
    raise Exception(f"No montage found for {edf_file_path}")


def transform_patient_to_tcp(patient_dir: str, montages: dict) -> None:
    """Transforms all EDF files in a patient dir into the TCP montage."""
    for file_name in os.listdir(patient_dir):
        if file_name.endswith(".edf"):
            edf_path = os.path.join(patient_dir, file_name)
            montage = get_montage_for_file(edf_path, montages)
            transform_edf_file(edf_path, montage)


print("\nTransforming EDFs into the TCP montage...")

montages_dir = os.path.join(data_root, "DOCS")
MONTAGES = load_montages(montages_dir)

pbar = tqdm(os.listdir(output_dir))
for patient in pbar:
    pbar.set_description(f"Transforming {patient:^10}")
    patient_path = os.path.join(output_dir, patient)
    if os.path.isdir(patient_path):
        transform_patient_to_tcp(patient_path, MONTAGES)

print("Done! You are ready to go :)")
print(
    f"The dataset that you will use for the preprocessing pipeline is this: {os.path.abspath(output_dir)}"
)
