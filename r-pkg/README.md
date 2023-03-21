
<!-- README.md is generated from README.Rmd. Please edit that file -->
<!-- Python configuration -->

# ktest R package

<!-- badges: start -->
<!-- badges: end -->

Kernel based statistical testing

# Installation

## Requirements

- R version 4+
- Python version 3+

> **Note:** if you don’t have Python on your system: when configuring
> the `ktest` R package (c.f. [below](#additional-setup)), the
> `reticulate` R package will offer to install Python on your system.

The `ktest` R package is using the `ktest` python package under the hood
thanks to the
[`reticulate`](https://CRAN.R-project.org/package=reticulate) R package
that provides an “R Interface to Python”.

### Install `ktest` R package

You can install the development version of `ktest` with the following
command:

``` r
remotes::install_github("AnthoOzier/ktest", ref = "rktest_dev", subdir = "r-pkg")
```

> **Note:** `ktest` is not available on CRAN at the moment but will be
> shortly.

### Additional setup

After installing the `ktest` R package, you need to run the following
command (**once**) to complete the setup (and install the `ktest` Python
package on your system):

> :warning: to avoid messing with your Python system environment, we
> recommend to use a dedicated Python virtual environment (c.f. [next
> section](#using-python-virtual-environment)) :warning:

``` r
library(ktest)
install_pyktest()
```

### Managing Python

If you need more **details** about which **version of Python** you are
using through `reticulate`, you can run:

``` r
reticulate::py_config()
reticulate::py_discover_config()
```

> **Note:** To get more information about **managing** which **version
> of Python** you are using, you can refer to `reticulate`
> [documentation](https://rstudio.github.io/reticulate/articles/versions.html)
> about “Python Version Configuration”.

#### Using Python virtual environment

1.  To create a dedicated Python environment, run the following command
    (once):

``` r
# create dedicated virtual environment
reticulate::virtualenv_create("ktest")
```

2.  To load the Python environment, you should run the following
    command:

``` r
# activate python environment
reticulate::use_virtualenv(virtualenv = "ktest", required = TRUE)
```

3.  Setup `ktest` Python dependency inside the dedicated environment (to
    be run once):

``` r
library(ktest)
install_pyktest(method = "virtualenv", envname = "ktest")
```

> **Note:** you can also use a Conda environment if you use a
> Miniconda/Anaconda Python distribution, c.f. `reticulate`
> [documentation](https://rstudio.github.io/reticulate/articles/versions.html)
> about “Python Version Configuration”.

### Usage

Once the `ktest` package is installed and configured, to use it, you
only need to load it like any other R package:

``` r
library(ktest)
kt <- ktest(...) # see other vignettes
```

If you use a dedicated Python environment, you also need to load it
before using `ktest`:

``` r
reticulate::use_virtualenv(virtualenv = "ktest", required = TRUE)
library(ktest)
kt <- ktest(...) # see other vignettes
```

# Example: multivariate testing

``` r
library(ktest)
```

## Load data

``` r
# cell expression data
sc_df <- read.table("data/data2.csv", row.names = 1, sep = ",", header = TRUE)
# cell metadata
meta_sc_df <- read.table("data/metadata2.csv", row.names = 1, sep = ",", header = TRUE)
```

## Ktest definition

Load data and metadata, and initialize ktest object - `data` is the
dataframe of data. - `metadata` is the dataframe of metadata. -
`condition` is the column of `metadata` containing the labels to test. -
`samples` is the couple of samples to test in the column `condition` of
`metadata` - Set `nystrom` to `TRUE` to use the Nystrom approximation. -
Set `verbose >0` to follow the steps of the initialization of the
`ktest` object.

Init test:

``` r
kt <- ktest(
    sc_df, meta_sc_df,
    condition='condition', samples=c('0H','48HREV'), verbose=1
)
```

## Multivariate testing

Perform a multivariate test on the data:

``` r
# R call
multivariate_test(kt, verbose=1)
# equivalent Python call through reticulate
kt$multivariate_test(verbose=1)
```

### Print test results

- `long` : if `TRUE`, the results are printed for several truncations
- `t` : truncation associated to the printed p-value if `long` is
  `FALSE`.
- `ts` : list of truncations associated to the printed p-values if
  `long` is `TRUE`.

``` r
# R call
print_multivariate_test_results(kt, long = TRUE, ts = c(1,2,3))
# equivalent Python call through reticulate
kt$print_multivariate_test_results(long = TRUE, ts = c(1,2,3))
```

### Get p-values table with respect to the truncation parameter.

- `contrib`: if true, returns the p-value associated to each principal
  component individually. Returns the p-value associated to the `kfda`
  statistic otherwise.
- `log`: if `TRUE`, returns the log p-values.

``` r
# R call
pval <- get_pvalue(kt, contrib = TRUE, log = FALSE)
# equivalent Python call through reticulate
pval <- kt$get_pvalue(contrib = TRUE, log = FALSE)

head(pval)
#>          1            2            3            4            5            6   
#> 6.117177e-01 1.246529e-08 2.552708e-08 1.884499e-07 1.101084e-01 4.584441e-01
```

### Plot p-values with respect to truncation

FIX ME!

### Other tests

The default p-value is the asymptotic p-value of the `kfda` statistic
associated to the truncation parameter `t=10`. To change the default
truncation parameter, use the class function `set_truncation`.

``` r
# Python call through reticulate
kt$set_truncation(5L)
kt$multivariate_test(verbose=1)
```

The asymptotic p-value is available for the `kfda` statistic only. The
permutation p-value is available for the `kfda` statistic and the `mmd`
statistic.

Compute the permutation p-value associated to the chosen statistic : -
`stat` : chosen statistic among (`kfda`,`mmd`) - `permutation`: if True,
compute the permutation p-value (automatically True if `stat` is
`mmd`) - `n_permutations` : set the number of permutations. -
`seed_permutation` : define the number of random seeds. -
`n_jobs_permutation` : number of CPU to use for parallelized computation

``` r
# Python call through reticulate
kt$multivariate_test(
    stat='mmd',
    permutation=TRUE,
    n_permutations=2000L,
    seed_permutation=123L,
    n_jobs_permutation=3L,
    verbose=1L)
```

Results:

``` r
# Python call through reticulate
kt$stat
#> [1] "mmd"
kt$get_pvalue_name()
#> [1] "standard_perm2000_seed123_datacondition0H48HREV"
kt$dict_pval_mmd[kt$get_pvalue_name()]
#> $standard_perm2000_seed123_datacondition0H48HREV
#> [1] 0
kt$get_pvalue()
#> [1] 0
```

### Nystrom approximation

Use the Nystrom approximation to reduce the computational cost

``` r
# Python call through reticulate
kt$set_test_params(nystrom=TRUE)
kt$multivariate_test(verbose=1)
```

Tune each parameter of the nystrom approximation with function
`set_test_params` : - `nystrom` : if True, use the Nystrom
approximation - `n_landmarks` : number of landmarks to use (subsample
used to compute the anchors). - `n_anchors` : number of anchors to use
(eigenvectors associated to a matrix defined with the landmarks). -
`landmark_method` : how to choose the landmarks among
(`random`,`kmeans`) - `anchor_basis` : matrix used to define the anchors
among - `k` : second moment of the landmarks. - `s` : total covariance
of the landmarks. - `w` : within group covariance of the landmarks.

``` r
# Python call through reticulate
kt$set_test_params(
    nystrom=TRUE,
    n_landmarks=50L,
    n_anchors=50L,
    landmark_method='random',
    anchor_basis='s'
)
kt$multivariate_test(verbose=1)
```

### Kernel function choice

**TODO**

The default kernel is the RBF kernel with median bandwidth. Use
`init_kernel_params()` to specify the kernel function with a `dict` of
specifications. - `function` is either a string among
\[`gauss`,`linear`\] or a user specified kernel function. + if
`function` is `gauss`: - `bandwidth` is either the string `median` or a
float. - if `bandwidth` is `median`: + `median_coef` is the coefficient
such that `bandwidth=median_coef x median` (default is `1`) -
`kernel_name` is the name of the kernel used.

``` python
import torch
from pyktest.base import init_kernel_params

# gauss kernel with median/2 bandwidth
kernel1 = init_kernel_params(function = 'gauss', bandwidth = 'median', median_coef = 1/2)

# linear kernel 
kernel2 = init_kernel_params(function='linear')

# user specified kernel
kernel3 = init_kernel_params(function=lambda x,y: torch.cdist(x,y).exp()+1,kernel_name='useless_kernel')
```

``` python
kt = Ktest(data,
           metadata,
           condition='condition',
           samples=['0H','48HREV'],
#            kernel=kernel3,
           verbose=1)

kt.multivariate_test(verbose=1)
```

    - Add data 'data' to Ktest, dimensions=(685, 83)
    - Set test data info (0H,48HREV from condition)
    - Define kernel function (gauss)
    - Initialize kfdat
        cov : standard 
        mmd : standard
    - Diagonalize within covariance centered gram
    - Compute within covariance centered gram
    - Compute kfda statistic

    ___Multivariate kfda test results___
    Asymptotic p-value(truncation) for multivariate testing : 
        p-value(10) = 3.9e-23 (kfda=130.41)

# Example: univariate testing
