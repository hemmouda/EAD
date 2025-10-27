# Molding the dataset once

The given preprocessing pipeline expects the dataset to be in a specific format. Namely:
- it expects the dataset to be grouped Patient-wise instead of Montage-wise,
- that there are no Seizure sessions or data,
- and that the EDF file are in the TCP montage.

The `mold.py` script takes care of this. You only need to run it once on the TUAR dataset, and it will create a new copy with the needed transformations. The resulting "new" dataset is what you will use with the preprocessing pipeline.

## Usage example
First don't forget to have the dependencies specified installed. Then you can run the script as follows:

```python
$ python mold.py -i /path/to/original/tuar/v3.0.1/folder 
```
