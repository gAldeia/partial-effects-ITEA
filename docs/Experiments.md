How experiment scripts work
=====

To run the experiment, you first need to have all dependencies installed, preferably in versions specified in README (you can create a conda environment to avoid messing with your local python libraries).

Then, you just call the script with the name of the data sets separated by a space in the terminal, for example:

```shell
$ python3 itea_experiment.py I.10.7 I.11.19 I.12.1 I.12.11 I.12.2
```

the code above will create a process that sequentially will run the informed number of executions ```n_runs``` for each dataset.

The script saves the progress in two local CSV files inside ```results/tabular_raw``` and can restart from the last checkpoint if interrupted. If you change something on the experiments scripts, it is advised to delete the files inside ```docs/results/tabular_raw```, since the script will restart from last saved execution.

Using the Filelock library, you can create multiple processes executing at the same time for the same regressor, **as long as they do not operate over the save data set**.
