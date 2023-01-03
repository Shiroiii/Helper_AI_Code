# A collection of python scripts to help with ML pipelining.

* `data_setup.py` :  Contains functions for creating datasets and dataloaders
* `engine.py`: Contains functions to train a model given an optimizer and loss function with an option to write tensorboard logs.
* `model_builder.py`: Contains the class definitions of any model being used.
* `utils.py`: Contains functions for defining `SummaryWriters` for logging for tensorboard. Also contains functions for saving the model and plotting loss and accuracy results.