# ktest

This package implements kernel tests such as kFDA and MMD. 

The results presented in [[1]](#1) are available [here](https://github.com/AnthoOzier/ktest_experiment_genome_biology_2024). The source code for experimental_data and squair scripts are based on the branch [publication_genome_biology](https://github.com/LMJL-Alea/ktest/tree/publication_genome_biology). The reversion scripts are based on the branch [publication_genome_biology_reversion](https://github.com/LMJL-Alea/ktest/tree/publication_genome_biology_reversion). See https://github.com/AnthoOzier/ktest_experiment_genome_biology_2024 for more details.


## Python package

See [`python`](./python) directory.

```python
pip install ktest@git+https://github.com/LMJL-Alea/ktest@main#subdirectory=python
```

## R package

See [`r-pkg`](./r-pkg) directory.

```r
remotes::install_github("AnthoOzier/ktest", ref = "r-ktest", subdir = "r-pkg")
```

## References
<a id="1">[1]</a> 
Anthony Ozier-Lafontaine, Camille Fourneaux, Ghislain Durif, Polina Arsenteva, Céline Vallot, Olivier Gandrillon, Sandrine Gonin-Giraud, Bertrand Michel, Franck Picard. (2024). Kernel-Based Testing for Single-Cell Differential Analysis. Preprint, [10.48550/arXiv.2307.08509](
https://doi.org/10.48550/arXiv.2307.08509).
