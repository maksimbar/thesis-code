This package is structured as follows:

- `original/`: Contains training and testing results specifically for the original SDnCNN model
- `proposed/`: Contains training and testing results specifically for the modified (proposed) SDnCNN model
- `experiments/`: Contains results from experimental training runs

Individual experiment runs are stored in subfolders. These subfolders are named with a timestamp (e.g., `yyyy-mm-dd_hh-mm-ss`), and each contains:

- `.hydra/config.yaml`: The configuration used for that specific run
- `best_model.pt`: The saved model weights that achieved the lowest Mean Squared Error (MSE)
- `train.log`: All log statements generated during training
- `train_log.csv`: Training metrics in CSV format, useful for creating diagrams with [tikz](https://tikz.net/) for the report

> [!NOTE]  
> For the _experiments_ `SI-SDR` and `STOI` values were not calculated on the validation set, only PESQ was.

