# Molding the dataset and Preprocessing

The provided [preprocessing pipeline](preprocessing/) expects the dataset to be in a certain format / shape. Thus, to be able to use it, you first need to mold the TUAR dataset once by running the script [here](molding/). But before that, you first need to install the needed Python packages. This [requirements.txt](requirements.txt) file has them, you simply need to:

```console
$ python -m pip install -r requirements.txt
```

However, it is recommended to first install PyTorch with your system resources in mind: [link](https://pytorch.org/get-started/locally/).

In case something doesn't work, this [detailed_requirements.txt](detailed_requirements.txt) file has the exact versions I used with the packages (with Python 3.10.2).