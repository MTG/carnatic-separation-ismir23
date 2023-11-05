# Carnatic singing voice separation trained with in-domain data with leakage
This is the official repository for:

- Carnatic Singing Voice Separation Using Cold Diffusion on Training Data with Bleeding, G. Plaja-Roglans, M. Miron, A. Shankar and X. Serra, 2023 (accepted for presentation at ISMIR 2023, Milan, Italy).

**IMPORTANT NOTE:** The code structure and an important part of the data loader and training code is an adaptation of the unofficial Tensorflow implementation of DiffWave (Zhifeng Kong et al., 2020). [Link to original repo](https://github.com/revsic/tf-diffwave).

**ANOTHER IMPORTANT NOTE:** The model in this repo can also be used for hassle-free inference through the Python library [compIAM](https://github.com/MTG/compIAM), a centralized repository of tools, models, and datasets for the computational analysis of Carnatic and Hindustani Music. With a few commands, you can easily download and run the separation model. Refer to `compIAM` to use these model (and many others!) out-of-the-box. `compIAM` is installed with `pip install compiam``, make sure to install `v0.3.0` to use the separation model in this repo.

## Requirements

The repository is based on Tensorflow 2. [See a complete list of requirements here](./requirements.txt).

## Run separation inference

To run separation inference you can use `separate.py` file.

```bash
python3 separate.py --input-signal /path/to/file.wav --clusters 5 --scheduler 4
```

Additional arguments can be passed to use a different model (`--model-name`), modify the batch size (i.e. chunk size processed by the model for optimized inference, `--batch-size`), and also specify to which GPU the process should be routed (`--gpu`).

## Train the model

To train your own model, you should first prepare the data. See [how we do process Saraga](./dataset/prepare_saraga.py) before the training process detailed in the paper. The key idea is to have the chunked and aligned audio samples of the dataset with a naming like: `<unique_id>_<source>.wav`, where `<source>` corresponds to `mixture` and `vocals`.

Then, run model training in [train.py](./train.py). Checkpoints will be stored every X training steps, X is defined by user in the [config.py](./config.py) file.

To start to train from previous checkpoint, `--load-step` is available.

```bash
python3 train.py --load-step 416 --config ./ckpt/<model_name>.json
```

Download the pre-trained weights for the feature extraction U-Net [here](https://drive.google.com/uc?export=download&id=1yj9iHTY7nCh2qrIM2RIUOXhLXt1K8WcE).

Unzip and store the weights into the [ckpt folder](./ckpt/). There should be .json file with the configuration, and a folder with the model weight checkpoint inside. Here's an example:

```py
with open('./ckpt/saraga-8.json') as f:
    config = Config.load(json.load(f))

diffwave = DiffWave(config.model)
diffwave.restore('./ckpt/saraga-8/saraga-8.ckpt-1').expect_partial()
```

[Write us](mailto:genis.plaja@upf.edu) or open an issue if you have any issues or questions!
