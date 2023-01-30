# Reproducibility Challenge 2022 :recycle:

*Author:* **Luka Žontar**

*Link to Reproducibility Article*: [here](./article/Reproducibility_Challenge_2022.pdf)

*Reproduced Article:* **Bandit Theory and Thompson Sampling-guided Directed Evolution for Sequence Optimization**

*Description*: This repository contains the code re-implementation of the reproduced article along with thorough
documentation, reproducibility instructions and implemented unit tests. This repository was developed to participate in
the Reproducibility Challenge 2022.

# Repo structure :blue_book:

This repository contains folders:

* ```article/``` - contains the original and the Reproducibility Challenge article.
* ```docs/``` - contains additional documentation of the repository.
* ```util/``` - contains Python helper files that are used when running the experiments in `notebooks/*.ipynb`.
* ```notebooks/``` - contains Jupyter notebooks with all the experiments.
  * ```plots/``` - contains plots that resulted from the main experiments in Jupyter notebooks.
* ```test/``` - contains unit tests for all the functions that were defined in the original article. Additionally, we
  added the implementation of the basic DE that is missing in the original article.

# Computer specifications :computer:

Here are the specifications of the computer that was used to reproduce the original article:

- PyCharm Professional
- Windows 11 Pro
- Processor: 11th Gen Intel(R) Core(TM) i7-1165G7, 2.80GHz
- RAM: 32GB

# Reproducing results :snake:

To reproduce results, you will need to fork this repository and install Python dependencies using `virtualenv`
and `pip`.

We used PyCharm to create virtual environment. Alternatively, find instructions to do it via your
terminal [here](https://docs.python.org/3/library/venv.html).

Next, use `pip` to install requirements from `requirements.txt`:

```
pip install -r requirements.txt
```

To run unit tests, execute:

```
python -m unittest discover test
```

Now your environment is ready to go. Running unit tests validates if all went well in environment setup.

Runing experiments in `notebooks` files produces some interesting results.:partying_face: :clinking_glasses:
