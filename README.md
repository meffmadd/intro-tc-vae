
# Intro-TC-VAE

An implementation of the Soft-Intro VAE with the beta-TC-VAE disentanglement term.


## Files in this repository

| File       | Description                                            |
|------------|--------------------------------------------------------|
| [main.py](https://github.com/meffmadd/ukiyo_e_project/blob/main/main.py)    | Entry point for training script, parses command line arguments     |
| [train.py](https://github.com/meffmadd/ukiyo_e_project/blob/main/train.py)   | Train loop with various parameters to change behaviour |
| [models.py](https://github.com/meffmadd/ukiyo_e_project/blob/main/models.py)  | Implements autoencoder with encoder and decoder modules        |
| [dataset.py](https://github.com/meffmadd/ukiyo_e_project/blob/main/dataset.py) | Implements the data loader with preprocessing and downsampling  |
| [utils.py](https://github.com/meffmadd/ukiyo_e_project/blob/main/utils.py)   | Loss functions and miscellaneous helper functions      |
| [ModelVis.ipynb](https://github.com/meffmadd/ukiyo_e_project/blob/main/ModelVis.ipynb)    | Visualizes the output of a specified model
| [tests](https://github.com/meffmadd/ukiyo_e_project/tree/main/tests) | Test cases, including a test run of the training loop
| [README.md](https://github.com/meffmadd/ukiyo_e_project/blob/main/README.md) | This file


## Running the training loop

```
python main.py --dataset ukiyo_e64 --device 0 --lr 2e-4 --num_epochs 250 --beta_kl 0.5 --beta_rec 0.75 --beta_neg 512 --z_dim 128 --batch_size 64 --amp --arch conv
```
Note that this requires a GPU to work. To train on the CPU use ```--device -1```.

To run the tests use pytest:

```
pytest ./tests
```
