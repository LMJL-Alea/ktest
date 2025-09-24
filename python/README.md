# ktest

This package implements kernel tests such as kernel Fisher Discriminant Analysis (kFDA) and Minimum Mean Discrepency (MMD) introduced in [[1]](#1).

## Note regarding code used in publication

The code for the analysis presented in [[1]](#1) are available in a dedicated repository [here](https://github.com/AnthoOzier/ktest_experiment_genome_biology_2024).

The source code for `experimental_data` and `squair` scripts are based on the branch [`publication_genome_biology`](https://github.com/LMJL-Alea/ktest/tree/publication_genome_biology). The `reversion` scripts are based on the branch [`publication_genome_biology_reversion`](https://github.com/LMJL-Alea/ktest/tree/publication_genome_biology_reversion).

See <https://github.com/AnthoOzier/ktest_experiment_genome_biology_2024> for more details.

## Licensing and authorship

See the dedicated [`LICENSE.md`](./LICENSE.md) and [`AUTHORS.md`](./AUTHORS.md) files.

## Python package

### Install

<!--Latest release:
```
pip install ktest
```-->

Latest development version:
```
pip install ktest@git+https://github.com/LMJL-Alea/ktest@main#subdirectory=python
```

### Tutorials

See the dedicated [`tutorials`](https://github.com/LMJL-Alea/ktest/tree/main/tutorials) folder in the project root directory for notebook tutorials.

### Usage

Example with a dataset used in [[1]](#1):

```python
import pandas as pd
from ktest.tester import Ktest

# data
url = "https://raw.githubusercontent.com/LMJL-Alea/ktest/main/tutorials/v5_data/RTqPCR_reversion_logcentered.csv"
data = pd.read_csv(url, index_col=0)

meta = pd.Series(data = pd.Series(data.index).apply(lambda x : x.split(sep='.')[1]))
meta.index = data.index

# init ktest object
kt_1 = Ktest(
    data=data, metadata=meta, sample_names=['48HREV','48HDIFF'], nystrom=True
)
# run kfda test
kt_1.test()
print(kt_1)

# save object
kt_1.save("ktest_RTqPCR_reversion_logcentered.pkl", compress=True)

# load saved
kt_1 = Ktest.load("ktest_RTqPCR_reversion_logcentered.pkl", compressed=True)
```

## References

<a id="1">[1]</a> Anthony Ozier-Lafontaine, Camille Fourneaux, Ghislain Durif, Polina Arsenteva, Céline Vallot, Olivier Gandrillon, Sandrine Gonin-Giraud, Bertrand Michel, Franck Picard. Kernel-based testing for single-cell differential analysis. Genome Biol 25, 114 (2024). [doi:10.1186/s13059-024-03255-1](https://doi.org/10.1186/s13059-024-03255-1), [hal-04214858](https://hal.science/hal-04214858)
