# PD-Rank
This repository contains the code for the paper *Ranking with confidence for large scale comparison data* ([Pre-print](https://arxiv.org/abs/2202.01670))

PD-Rank is a method for pairwise ranking aggregation from noisy labels. Given a set of (possibly) noise pairwise comparisons between a set of items, PD-Rank returns a ranked list of the entire item set. PD-Rank scales well with the number of items due to the optimization method employed.

## How to Use

To use PD-Rank follow the following steps

1. **Installation**: Clone the repository and install the required dependencies.
    
    ```bash
    git clone https://github.com/FilVa/PD-Rank.git
    cd PD-Rank
    pip install -r requirements.txt
    ```

2. **Usage**: For an example usage, see `PDRank-demo.py`.


## Citation

If you use PD-Rank in your research, please cite the following paper:

```

@article{doi:10.1137/22M1495494,
author = {Valdeira, Filipa M. and Ferreira, Ricardo and Micheletti, Alessandra and Soares, Cl\'{a}udia},
title = {Probabilistic Registration for Gaussian Process Three-Dimensional Shape Modelling in the Presence of Extensive Missing Data},
journal = {SIAM Journal on Mathematics of Data Science},
volume = {5},
number = {2},
pages = {502-527},
year = {2023},
doi = {10.1137/22M1495494},
URL = {https://doi.org/10.1137/22M1495494},
}

```
