# Project Structure

## configs
yaml files. debug and trainer config.
- demo
- path
- experiments

## dataloader
### main
get train_dataloader and val_dataloader.
- dataloader.py
### dataset
how to process original data. Including other data processing and loading method or class.
- dataset.py
- label_map.py
- ...
### other
- collate_fn.py
- ...


## model
### main
get nn.Module subclass model.
- model.py
### modules
The modules that are used in model.
- ...
### other
composition of main model.

## utils


## train
### trainer
based on lightning.

