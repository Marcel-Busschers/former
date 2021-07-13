# GitHub README - FVAE

Bachelor Thesis implementation. Vrije Universiteit Amsterdam. BSc Computer Science. 2018-2021

**Supervisor**: [Peter Bloem](https://github.com/pbloem)

# former

---

This repo has been forked from [the following repo](https://github.com/pbloem/former) which contains a simple transformer implementation from scratch; from which all the work conducted gets built on:

- `experiments/generate.py`
- `former/transformers.py`

There are two branches, **FVAE** and **master**.

1. **FVAE** is the final model to accommodate the financial data.
2. **master** contains the model accommodating the language data.

# FVAE

---

To use the model on default settings, run it from the root directory:

```python
python experiments/generate.py
```

Hyperparameters can be passed in via the command line as well. Type `-h` to list all hyperparameters:

```python
python experiments/generate.py -h
```

For instance, to run the model on a depth of 2 for 150 epochs, run the following:

```python
python experiments/generate.py -d 2 -e 150 --log True
```

# Requirements

---

Python 3.6+ is required.

The following should install all requirements `pip install torch tqdm numpy`

You may also need `pip install future` depending on the exact python version.

## conda environment

The file `environment.yml` describes a complete conda environment with all dependencies. After cloning or downloading the project, you create the environment as follows:

```python
conda env create -f environment.yml --name former
conda activate former
```

# Logging

---

To log tensorboards and reconstructions, include `—-log True` in the command:

```python
python experiments/generate.py --log True
```

This will log the loss curves and gradient norms to Tensorboard, as well as, create a file containing the reconstructed sentence per epoch. A new directory will be created in `runs` with the name of the current date and time, containing:

- Reconstruction pdf file
- Model save points
- Tensorboard logs

You can add a description message via `-m` that will get added to the reconstruction pdf file:

```python
python experiments/generate.py --log True -m "Some description here"
```

To see the tensorboard logs, for example, the run of `2021-06-22_15:28` - run the command:

```python
tensorboard --logdir 2021-06-22_15:28
```

If you are running this on a server instance, you might have to bind it to a port:

```python
tensorboard --logdir 2021-06-22_15:28 --bind_all
```

**NOTE**: the gitignore will ignore all any model checkpoint files and tensorboard logging files, therefore these commands won't work on any existing runs.

# Extra Hyperparameters

---

This branch includes two extra hyperparameters than the model within **master:**

- Latent Size `-L <int>` that sets the size of the embedded latent vector, allowing you to compress the Encoder's output accordingly.
- Single Batching `—-single-batch <bool>` that allows you to train on a single batch, for overfitting purposes.

# Latent Visualisation

---

The `experiments/latent.py` allows you to visualise the latent space for any given sequence in the [data](https://github.com/Marcel-Busschers/former/blob/FVAE/data/data.pdf). Three hyperparameters are available via `-h`.

To run the file, the directory of the model needs to be passed in with a direct path to the `model.pt` file, as well as which sequences you want to see in latent space, for example, the run of `2021-06-16_14:00` - run the command:

```python
python experiments/latent.py --model-dir 2021-06-16_14:00 --from-sequence 1 4 6
```

Running `--from-sequence 0` will allow you to visualise the latent space of all sequences within the data.

This will create a pdf file within the run's directory. You can add a description via `-d` that will get added to the pdf:

```python
python experiments/latent.py --model-dir 2021-06-16_14:00 --from-sequence 0 -d "All sequences"
```

Visualisation gets done using [tSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).