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

# master

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

To log tensorboards and text generation, include `—log True` in the command:

```python
python experiments/generate.py --log True
```

This will log the loss curves and gradient norms to Tensorboard, as well as, create a file containing the generated sentence per epoch. A new directory will be created in `runs` with the name of the current date and time, containing:

- Generated text file
- Model save points
- Tensorboard logs

You can add a description message via `-m` that will get added to the generated text file:

```python
python experiments/generate.py --log True -m "Some description here"
```

To see the tensorboard logs, for example, the run of `[2021-06-22_15:28](https://github.com/Marcel-Busschers/former/tree/master/runs/2021-06-22_15:28)` - run the command:

```python
tensorboard --logdir 2021-06-22_15:28
```

If you are running this on a server instance, you might have to bind it to a port:

```python
tensorboard --logdir 2021-06-22_15:28 --bind_all
```