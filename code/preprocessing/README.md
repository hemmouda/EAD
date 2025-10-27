# Preprocessing pipeline

The `main.py` script not only can preprocess the data, it can also train SUMO2 on it in one go, as that it has the SUMO2 code "integrated" with it. You can read more about that by running:

```console
$ python main.py -h
```

To use the script to produce the SUMO2 pickled subject dict that was used for the presented model, first make sure to point out the location of your molded dataset in this [file](core/data_splitting.py). After that you can run:

```console
$ python main.py -c config.yaml -s data_split.yaml --no-sumo
```

The run info are always saved in a new folder in an output dir.

More information about the values in the config can be found in the [config file](config.yaml) it self.