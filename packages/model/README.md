This package is structured as follow:

- `conf/` Contains Hydra configuration
- `models/` Contains weights of the models
- `scripts/splitter` Contains script for splitting dataset
- `scripts/playground` Contains script for restoring audio

## Project setup

Clone the repository

```
git clone https://github.com/maksimbar/thesis-code.git
```

Switch to this package

```
cd packages/model
```

Install the dependencies using [uv](https://docs.astral.sh/uv/) 

```
uv sync
```

## Generating dataset
Dataset with `60/20/20` splits is already included as `data.zip`, you just need to extract it.

However, if you want to generate one from scratch, you can merge `test`, `train` and `valid` directories and run this command:

```
uv run scripts/splitter/main.py data --seed 42
```

## Running experiments
This project uses [Hydra](https://github.com/facebookresearch/hydra) package to handle different parameter configurations.

### Sanity check
To make sure everything works as expected, run the command below. It sets pretty small computational requirements and must work just fine even with a CPU. 

```
uv run train.py \
    data.base_dir='${hydra:runtime.cwd}/data_small' \
    data.sample_rate=8000 \
    train.epochs=5 \
    train.sgd.lr_decay_epochs=5 \
    stft.window_type=hamming \
    model.activation=relu
```

### Window functions

```
uv run train.py stft.window_type=hamming
```

```
uv run train.py stft.window_type=hann
```

```
uv run train.py stft.window_type=blackman
```

### Activation functions

```
uv run train.py model.activation=relu
```

```
uv run train.py model.activation=prelu
```

```
uv run train.py model.activation=leaky_relu
```

### Dilated convolutions

```
uv run train.py model.dilation_rates=[1] 
```

```
uv run train.py model.dilation_rates=[2]
```

```
uv run train.py model.dilation_rates=[1,2,4]
```

## Empirical testing

If you wish to verify performance of a trained model empirically, you can use `scripts/playground/playground.py` script.

1. Put saved model checkpoint under `models/` directory  
2. Specify architectural details (e.g. window function, activation function, etc.) in the constants of the playground  
3. Provide path to noisy and clean file  
4. Execute the script
