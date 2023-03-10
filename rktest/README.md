
<!-- README.md is generated from README.Rmd. Please edit that file -->

# rktest

<!-- badges: start -->
<!-- badges: end -->

Kernal based statistical testing

## Installation

You can install the development version of `rktest` with the following
command:

``` r
remotes::install_github("AnthoOzier/ktest", ref = "rktest_dev", subdir = "rktest")
```

## Example

More to come

``` r
library(rktest)
```

    #> ℹ Loading rktest

### Multivariate testing

Load data:

``` r
sc_df <- read.table("data/data2.csv", row.names = 1, sep = ",", header = TRUE)
str(sc_df)
#> 'data.frame':    685 obs. of  83 variables:
#>  $ AACS       : num  -0.302 1.369 1.956 2.949 -3.015 ...
#>  $ ACSL6      : num  -0.938 0.443 -0.911 1.303 -1.458 ...
#>  $ ACSS1      : num  -0.506 -0.506 2.7 -0.506 -0.506 ...
#>  $ ALAS1      : num  1.656 1.307 -0.605 0.952 0.402 ...
#>  $ AMDHD2     : num  -0.768 0.297 1.105 0.838 -2.263 ...
#>  $ ARHGEF2    : num  -0.315 1.219 1.549 1.717 0.59 ...
#>  $ BATF       : num  -0.101 1.119 1.383 1.45 -2.147 ...
#>  $ BCL11A     : num  -0.108 -0.236 0.523 1.59 -1.476 ...
#>  $ betaglobin : num  -0.503 -0.37 -2.218 -2.831 -6.748 ...
#>  $ BPI        : num  0.504 1.777 0.871 0.91 1.691 ...
#>  $ CD151      : num  0.0805 0.4015 0.9873 0.4245 -1.0048 ...
#>  $ CD44       : num  3.51 4.25 1.29 -1.13 1.08 ...
#>  $ CREG1      : num  -2.64 -0.608 0.206 0.715 -1.024 ...
#>  $ CRIP2      : num  0.696 2.65 -1.233 0.595 2.358 ...
#>  $ CTCF       : num  -0.407 -0.677 0.668 0.758 -0.33 ...
#>  $ CTSA       : num  -0.7327 -1.0107 0.6701 0.0927 -0.0454 ...
#>  $ CYP51A1    : num  0.545 0.163 2.326 2.292 0.463 ...
#>  $ DCP1A      : num  -2.155 -1.083 -0.23 0.153 -0.559 ...
#>  $ DCTD       : num  -1.633 -1.527 1.275 0.94 0.306 ...
#>  $ DHCR24     : num  0.583 1.187 2.592 2.433 0.776 ...
#>  $ DHCR7      : num  0.715 0.355 1.215 2.216 0.331 ...
#>  $ DPP7       : num  -0.1907 -0.0919 0.7108 0.6894 -2.2918 ...
#>  $ EGFR       : num  1.173 0.846 -0.667 -0.234 -0.229 ...
#>  $ EMB        : num  0.176 -2.858 0.367 0.299 -0.426 ...
#>  $ FAM208B    : num  0.404 0.746 0.855 -0.191 -0.828 ...
#>  $ FHL3       : num  -0.0796 0.4341 0.4527 1.2118 0.8014 ...
#>  $ FNIP1.F1.R1: num  -1.2343 -0.0549 1.3443 1.2734 0.457 ...
#>  $ GLRX5      : num  -0.712 -0.439 -0.199 -0.256 -0.676 ...
#>  $ GPT2       : num  1.6508 0.0534 1.8715 -0.6208 -0.4948 ...
#>  $ GSN        : num  1.19 1.36 1.16 1.14 1.31 ...
#>  $ HMGCS1     : num  1.526 -1.123 2.081 2.051 0.969 ...
#>  $ HRAS1      : num  -2.931 -2.931 1.382 1.102 0.513 ...
#>  $ HSD17B7    : num  2.69 1.46 2.76 2.39 1.81 ...
#>  $ HSP90AA1   : num  -1.484 -0.847 0.474 0.307 -0.754 ...
#>  $ HYAL1      : num  -1.603 1.412 -0.566 -1.603 -1.603 ...
#>  $ LCP1       : num  1.31 1.62 2.16 1.38 1.55 ...
#>  $ LDHA       : num  -0.039 1.097 0.164 1.222 0.249 ...
#>  $ MAPK12     : num  -2.2538 0.0355 -2.2538 -2.2538 1.7362 ...
#>  $ MFSD2B     : num  -0.737 -0.737 2.253 -0.737 -0.737 ...
#>  $ MID2       : num  1.31 1.51 1.33 1.51 1.75 ...
#>  $ MKNK2      : num  -1.1043 0.0214 -0.0623 -0.2849 -0.6815 ...
#>  $ MTFR1      : num  -0.0668 0.1087 -0.0495 0.3421 -2.1225 ...
#>  $ MVD        : num  0.935 0.684 0.666 2.403 -3.487 ...
#>  $ MYO1G      : num  1.87 1.73 1.85 3.33 2.06 ...
#>  $ NCOA4      : num  -0.78 -0.513 -0.172 0.174 -4.84 ...
#>  $ NSDHL      : num  1.22 -2.75 3.51 3.64 2.12 ...
#>  $ PDLIM7     : num  -1.236 0.137 0.361 1.266 0.255 ...
#>  $ PIK3CG     : num  -0.8855 -0.1217 -0.8588 0.0323 -0.8855 ...
#>  $ PLAG1      : num  0.671 -3.278 1.052 -0.919 -2.698 ...
#>  $ PLS1       : num  -0.859 -0.315 -0.251 -0.383 -1.031 ...
#>  $ PLS3       : num  -0.3427 -1.5608 -0.0526 0.5956 -0.808 ...
#>  $ PPP1R15B   : num  -0.933 0.176 0.366 0.149 -1.327 ...
#>  $ PTPRC      : num  -1.01 2.95 1.58 2.05 -1.07 ...
#>  $ RBM38      : num  -2.9 -2.9 1.44 1.56 1 ...
#>  $ REXO2      : num  -0.1057 0.6702 0.3218 0.0447 -0.7767 ...
#>  $ RFFL       : num  -0.53 0.5703 0.0444 -1.6615 -1.6777 ...
#>  $ RPL22L1    : num  -1.647 0.179 0.182 -0.384 -1.892 ...
#>  $ RSFR       : num  0.512 0.386 1.94 1.778 3.086 ...
#>  $ RUNX2      : num  0.974 2.046 0.651 1.775 0.886 ...
#>  $ sca2       : num  0.557 0.762 0.661 0.722 1.66 ...
#>  $ SCD        : num  1.18 1.068 1.889 1.754 0.341 ...
#>  $ SERPINI1   : num  -0.108 -0.108 1.187 -0.108 -0.108 ...
#>  $ SLC25A37   : num  -3.589 -3.589 -0.478 -0.187 -1.21 ...
#>  $ SLC6A9     : num  -0.97 0.487 -2.005 -1.131 -4.279 ...
#>  $ SLC9A3R2   : num  0.0279 -0.6926 -0.9606 0.6326 -3.9259 ...
#>  $ SMPD1      : num  -0.529 -0.516 0.939 0.702 -0.713 ...
#>  $ SNX22      : num  -2.22452 -0.08351 -0.00224 -1.06277 -2.22452 ...
#>  $ SNX27      : num  -0.488 -0.926 0.764 0.658 -1.488 ...
#>  $ SQLE       : num  0.293 1.438 1.697 2.361 0.332 ...
#>  $ SQSTM1     : num  -3.6827 -0.2549 -0.9829 0.4216 0.0358 ...
#>  $ STARD4     : num  -0.256 0.267 0.498 2.543 -0.87 ...
#>  $ STX12      : num  -1.08 -0.101 0.358 -0.922 -1.13 ...
#>  $ SULF2      : num  0.852 -0.13 2.262 0.644 2.058 ...
#>  $ SULT1E1    : num  0.1692 -0.0565 0.7738 -0.7774 -0.7994 ...
#>  $ TADA2L     : num  -0.427 -0.282 0.934 -1.348 0.758 ...
#>  $ TBC1D7     : num  -2.415 -1.95 1.046 -0.176 -1.044 ...
#>  $ TNFRSF21   : num  -0.1333 0.5429 0.5746 0.2517 0.0427 ...
#>  $ TPP1       : num  0.274 -0.992 1.217 -0.152 0.179 ...
#>  $ TTYH2      : num  -1.94 -1.94 -1.94 2.39 -1.94 ...
#>  $ UCK1       : num  -1.444 -1.024 -0.384 0.419 -4 ...
#>  $ VDAC3      : num  -0.228 -1.147 -0.273 0.211 -0.448 ...
#>  $ WDR91      : num  -0.3695 -0.6212 0.3039 0.3657 -0.0581 ...
#>  $ XPNPEP1    : num  -2.0073 -1.9719 -0.0374 1.0846 -1.0859 ...
rownames(sc_df)
#>   [1] "REV1.0H.1"       "REV1.0H.2"       "REV1.0H.3"       "REV1.0H.4"      
#>   [5] "REV1.0H.5"       "REV1.0H.6"       "REV1.0H.7"       "REV1.0H.8"      
#>   [9] "REV1.0H.9"       "REV1.0H.10"      "REV1.0H.11"      "REV1.0H.12"     
#>  [13] "REV1.0H.13"      "REV1.0H.14"      "REV1.0H.15"      "REV1.0H.16"     
#>  [17] "REV1.0H.17"      "REV1.0H.18"      "REV1.0H.19"      "REV1.0H.20"     
#>  [21] "REV1.0H.21"      "REV1.0H.22"      "REV1.0H.23"      "REV1.24H.1"     
#>  [25] "REV1.24H.2"      "REV1.24H.3"      "REV1.24H.4"      "REV1.24H.5"     
#>  [29] "REV1.24H.6"      "REV1.24H.7"      "REV1.24H.8"      "REV1.24H.9"     
#>  [33] "REV1.24H.10"     "REV1.24H.11"     "REV1.24H.12"     "REV1.24H.13"    
#>  [37] "REV1.24H.14"     "REV1.24H.15"     "REV1.24H.16"     "REV1.24H.17"    
#>  [41] "REV1.24H.18"     "REV1.24H.19"     "REV1.24H.20"     "REV1.24H.21"    
#>  [45] "REV1.48HDIFF.1"  "REV1.48HDIFF.2"  "REV1.48HDIFF.3"  "REV1.48HDIFF.4" 
#>  [49] "REV1.48HDIFF.5"  "REV1.48HDIFF.6"  "REV1.48HDIFF.7"  "REV1.48HDIFF.8" 
#>  [53] "REV1.48HDIFF.9"  "REV1.48HDIFF.10" "REV1.48HDIFF.11" "REV1.48HDIFF.12"
#>  [57] "REV1.48HDIFF.13" "REV1.48HDIFF.14" "REV1.48HDIFF.15" "REV1.48HDIFF.16"
#>  [61] "REV1.48HDIFF.17" "REV1.48HDIFF.18" "REV1.48HDIFF.19" "REV1.48HDIFF.20"
#>  [65] "REV1.48HDIFF.21" "REV1.48HDIFF.22" "REV1.48HDIFF.23" "REV1.48HDIFF.24"
#>  [69] "REV1.48HREV.1"   "REV1.48HREV.2"   "REV1.48HREV.3"   "REV1.48HREV.4"  
#>  [73] "REV1.48HREV.5"   "REV1.48HREV.6"   "REV1.48HREV.7"   "REV1.48HREV.8"  
#>  [77] "REV1.48HREV.9"   "REV1.48HREV.10"  "REV1.48HREV.11"  "REV1.48HREV.12" 
#>  [81] "REV1.48HREV.13"  "REV1.48HREV.14"  "REV1.48HREV.15"  "REV1.48HREV.16" 
#>  [85] "REV1.48HREV.17"  "REV1.48HREV.18"  "REV1.48HREV.19"  "REV1.48HREV.20" 
#>  [89] "REV1.48HREV.21"  "REV2.0H.1"       "REV2.0H.2"       "REV2.0H.3"      
#>  [93] "REV2.0H.4"       "REV2.0H.5"       "REV2.0H.6"       "REV2.0H.7"      
#>  [97] "REV2.0H.8"       "REV2.0H.9"       "REV2.0H.10"      "REV2.0H.11"     
#> [101] "REV2.0H.12"      "REV2.0H.13"      "REV2.0H.14"      "REV2.0H.15"     
#> [105] "REV2.0H.16"      "REV2.0H.17"      "REV2.0H.18"      "REV2.0H.19"     
#> [109] "REV2.0H.20"      "REV2.0H.21"      "REV2.0H.22"      "REV2.0H.23"     
#> [113] "REV2.24H.1"      "REV2.24H.2"      "REV2.24H.3"      "REV2.24H.4"     
#> [117] "REV2.24H.5"      "REV2.24H.6"      "REV2.24H.7"      "REV2.24H.8"     
#> [121] "REV2.24H.9"      "REV2.24H.10"     "REV2.24H.11"     "REV2.24H.12"    
#> [125] "REV2.24H.13"     "REV2.24H.14"     "REV2.24H.15"     "REV2.24H.16"    
#> [129] "REV2.24H.17"     "REV2.24H.18"     "REV2.24H.19"     "REV2.24H.20"    
#> [133] "REV2.24H.21"     "REV2.24H.22"     "REV2.24H.23"     "REV2.24H.24"    
#> [137] "REV2.48HDIFF.1"  "REV2.48HDIFF.2"  "REV2.48HDIFF.3"  "REV2.48HDIFF.4" 
#> [141] "REV2.48HDIFF.5"  "REV2.48HDIFF.6"  "REV2.48HDIFF.7"  "REV2.48HDIFF.8" 
#> [145] "REV2.48HDIFF.9"  "REV2.48HDIFF.10" "REV2.48HDIFF.11" "REV2.48HDIFF.12"
#> [149] "REV2.48HDIFF.13" "REV2.48HDIFF.14" "REV2.48HDIFF.15" "REV2.48HDIFF.16"
#> [153] "REV2.48HDIFF.17" "REV2.48HDIFF.18" "REV2.48HDIFF.19" "REV2.48HDIFF.20"
#> [157] "REV2.48HDIFF.21" "REV2.48HDIFF.22" "REV2.48HREV.1"   "REV2.48HREV.2"  
#> [161] "REV2.48HREV.3"   "REV2.48HREV.4"   "REV2.48HREV.5"   "REV2.48HREV.6"  
#> [165] "REV2.48HREV.7"   "REV2.48HREV.8"   "REV2.48HREV.9"   "REV2.48HREV.10" 
#> [169] "REV2.48HREV.11"  "REV2.48HREV.12"  "REV2.48HREV.13"  "REV2.48HREV.14" 
#> [173] "REV2.48HREV.15"  "REV2.48HREV.16"  "REV2.48HREV.17"  "REV2.48HREV.18" 
#> [177] "REV2.48HREV.19"  "REV2.48HREV.20"  "REV2.48HREV.21"  "REV3.0H.1"      
#> [181] "REV3.0H.2"       "REV3.0H.3"       "REV3.0H.4"       "REV3.0H.5"      
#> [185] "REV3.0H.6"       "REV3.0H.7"       "REV3.0H.8"       "REV3.0H.9"      
#> [189] "REV3.0H.10"      "REV3.0H.11"      "REV3.0H.12"      "REV3.0H.13"     
#> [193] "REV3.0H.14"      "REV3.0H.15"      "REV3.0H.16"      "REV3.0H.17"     
#> [197] "REV3.0H.18"      "REV3.0H.19"      "REV3.0H.20"      "REV3.0H.21"     
#> [201] "REV3.24H.1"      "REV3.24H.2"      "REV3.24H.3"      "REV3.24H.4"     
#> [205] "REV3.24H.5"      "REV3.24H.6"      "REV3.24H.7"      "REV3.24H.8"     
#> [209] "REV3.24H.9"      "REV3.24H.10"     "REV3.24H.11"     "REV3.24H.12"    
#> [213] "REV3.24H.13"     "REV3.24H.14"     "REV3.24H.15"     "REV3.24H.16"    
#> [217] "REV3.24H.17"     "REV3.24H.18"     "REV3.24H.19"     "REV3.24H.20"    
#> [221] "REV3.24H.21"     "REV3.24H.22"     "REV3.24H.23"     "REV3.24H.24"    
#> [225] "REV3.48HDIFF.1"  "REV3.48HDIFF.2"  "REV3.48HDIFF.3"  "REV3.48HDIFF.4" 
#> [229] "REV3.48HDIFF.5"  "REV3.48HDIFF.6"  "REV3.48HDIFF.7"  "REV3.48HDIFF.8" 
#> [233] "REV3.48HDIFF.9"  "REV3.48HDIFF.10" "REV3.48HDIFF.11" "REV3.48HDIFF.12"
#> [237] "REV3.48HDIFF.13" "REV3.48HDIFF.14" "REV3.48HDIFF.15" "REV3.48HDIFF.16"
#> [241] "REV3.48HDIFF.17" "REV3.48HDIFF.18" "REV3.48HDIFF.19" "REV3.48HDIFF.20"
#> [245] "REV3.48HDIFF.21" "REV3.48HDIFF.22" "REV3.48HREV.1"   "REV3.48HREV.2"  
#> [249] "REV3.48HREV.3"   "REV3.48HREV.4"   "REV3.48HREV.5"   "REV3.48HREV.6"  
#> [253] "REV3.48HREV.7"   "REV3.48HREV.8"   "REV3.48HREV.9"   "REV3.48HREV.10" 
#> [257] "REV3.48HREV.11"  "REV3.48HREV.12"  "REV3.48HREV.13"  "REV3.48HREV.14" 
#> [261] "REV3.48HREV.15"  "REV3.48HREV.16"  "REV3.48HREV.17"  "REV3.48HREV.18" 
#> [265] "REV3.48HREV.19"  "REV3.48HREV.20"  "REV3.48HREV.21"  "REV3.48HREV.22" 
#> [269] "REV4.0H.1"       "REV4.0H.2"       "REV4.0H.3"       "REV4.0H.4"      
#> [273] "REV4.0H.5"       "REV4.0H.6"       "REV4.0H.7"       "REV4.0H.8"      
#> [277] "REV4.0H.9"       "REV4.0H.10"      "REV4.0H.11"      "REV4.0H.12"     
#> [281] "REV4.0H.13"      "REV4.0H.14"      "REV4.0H.15"      "REV4.0H.16"     
#> [285] "REV4.0H.17"      "REV4.0H.18"      "REV4.24H.1"      "REV4.24H.2"     
#> [289] "REV4.24H.3"      "REV4.24H.4"      "REV4.24H.5"      "REV4.24H.6"     
#> [293] "REV4.24H.7"      "REV4.24H.8"      "REV4.24H.9"      "REV4.24H.10"    
#> [297] "REV4.24H.11"     "REV4.24H.12"     "REV4.24H.13"     "REV4.24H.14"    
#> [301] "REV4.24H.15"     "REV4.24H.16"     "REV4.24H.17"     "REV4.24H.18"    
#> [305] "REV4.24H.19"     "REV4.24H.20"     "REV4.24H.21"     "REV4.48HDIFF.1" 
#> [309] "REV4.48HDIFF.2"  "REV4.48HDIFF.3"  "REV4.48HDIFF.4"  "REV4.48HDIFF.5" 
#> [313] "REV4.48HDIFF.6"  "REV4.48HDIFF.7"  "REV4.48HDIFF.8"  "REV4.48HDIFF.9" 
#> [317] "REV4.48HDIFF.10" "REV4.48HDIFF.11" "REV4.48HDIFF.12" "REV4.48HDIFF.13"
#> [321] "REV4.48HDIFF.14" "REV4.48HDIFF.15" "REV4.48HDIFF.16" "REV4.48HDIFF.17"
#> [325] "REV4.48HDIFF.18" "REV4.48HDIFF.19" "REV4.48HREV.1"   "REV4.48HREV.2"  
#> [329] "REV4.48HREV.3"   "REV4.48HREV.4"   "REV4.48HREV.5"   "REV4.48HREV.6"  
#> [333] "REV4.48HREV.7"   "REV4.48HREV.8"   "REV4.48HREV.9"   "REV4.48HREV.10" 
#> [337] "REV4.48HREV.11"  "REV4.48HREV.12"  "REV4.48HREV.13"  "REV4.48HREV.14" 
#> [341] "REV4.48HREV.15"  "REV4.48HREV.16"  "REV4.48HREV.17"  "REV4.48HREV.18" 
#> [345] "REV4.48HREV.19"  "REV5.0H.1"       "REV5.0H.2"       "REV5.0H.3"      
#> [349] "REV5.0H.4"       "REV5.0H.5"       "REV5.0H.6"       "REV5.0H.7"      
#> [353] "REV5.0H.8"       "REV5.0H.9"       "REV5.0H.10"      "REV5.0H.11"     
#> [357] "REV5.0H.12"      "REV5.0H.13"      "REV5.0H.14"      "REV5.0H.15"     
#> [361] "REV5.0H.16"      "REV5.0H.17"      "REV5.0H.18"      "REV5.0H.19"     
#> [365] "REV5.0H.20"      "REV5.0H.21"      "REV5.0H.22"      "REV5.0H.23"     
#> [369] "REV5.24H.1"      "REV5.24H.2"      "REV5.24H.3"      "REV5.24H.4"     
#> [373] "REV5.24H.5"      "REV5.24H.6"      "REV5.24H.7"      "REV5.24H.8"     
#> [377] "REV5.24H.9"      "REV5.24H.10"     "REV5.24H.11"     "REV5.24H.12"    
#> [381] "REV5.24H.13"     "REV5.24H.14"     "REV5.24H.15"     "REV5.24H.16"    
#> [385] "REV5.24H.17"     "REV5.24H.18"     "REV5.24H.19"     "REV5.24H.20"    
#> [389] "REV5.24H.21"     "REV5.48HDIFF.1"  "REV5.48HDIFF.2"  "REV5.48HDIFF.3" 
#> [393] "REV5.48HDIFF.4"  "REV5.48HDIFF.5"  "REV5.48HDIFF.6"  "REV5.48HDIFF.7" 
#> [397] "REV5.48HDIFF.8"  "REV5.48HDIFF.9"  "REV5.48HDIFF.10" "REV5.48HDIFF.11"
#> [401] "REV5.48HDIFF.12" "REV5.48HDIFF.13" "REV5.48HDIFF.14" "REV5.48HDIFF.15"
#> [405] "REV5.48HDIFF.16" "REV5.48HDIFF.17" "REV5.48HDIFF.18" "REV5.48HDIFF.19"
#> [409] "REV5.48HDIFF.20" "REV5.48HDIFF.21" "REV5.48HREV.1"   "REV5.48HREV.2"  
#> [413] "REV5.48HREV.3"   "REV5.48HREV.4"   "REV5.48HREV.5"   "REV5.48HREV.6"  
#> [417] "REV5.48HREV.7"   "REV5.48HREV.8"   "REV5.48HREV.9"   "REV5.48HREV.10" 
#> [421] "REV5.48HREV.11"  "REV5.48HREV.12"  "REV5.48HREV.13"  "REV5.48HREV.14" 
#> [425] "REV5.48HREV.15"  "REV5.48HREV.16"  "REV5.48HREV.17"  "REV5.48HREV.18" 
#> [429] "REV5.48HREV.19"  "REV5.48HREV.20"  "REV5.48HREV.21"  "REV5.48HREV.22" 
#> [433] "REV5.48HREV.23"  "REV6.0H.1"       "REV6.0H.2"       "REV6.0H.3"      
#> [437] "REV6.0H.4"       "REV6.0H.5"       "REV6.0H.6"       "REV6.0H.7"      
#> [441] "REV6.0H.8"       "REV6.0H.9"       "REV6.0H.10"      "REV6.0H.11"     
#> [445] "REV6.0H.12"      "REV6.0H.13"      "REV6.0H.14"      "REV6.0H.15"     
#> [449] "REV6.0H.16"      "REV6.0H.17"      "REV6.0H.18"      "REV6.0H.19"     
#> [453] "REV6.0H.20"      "REV6.24H.1"      "REV6.24H.2"      "REV6.24H.3"     
#> [457] "REV6.24H.4"      "REV6.24H.5"      "REV6.24H.6"      "REV6.24H.7"     
#> [461] "REV6.24H.8"      "REV6.24H.9"      "REV6.24H.10"     "REV6.24H.11"    
#> [465] "REV6.24H.12"     "REV6.24H.13"     "REV6.24H.14"     "REV6.24H.15"    
#> [469] "REV6.24H.16"     "REV6.24H.17"     "REV6.24H.18"     "REV6.24H.19"    
#> [473] "REV6.24H.20"     "REV6.24H.21"     "REV6.24H.22"     "REV6.48HDIFF.1" 
#> [477] "REV6.48HDIFF.2"  "REV6.48HDIFF.3"  "REV6.48HDIFF.4"  "REV6.48HDIFF.5" 
#> [481] "REV6.48HDIFF.6"  "REV6.48HDIFF.7"  "REV6.48HDIFF.8"  "REV6.48HDIFF.9" 
#> [485] "REV6.48HDIFF.10" "REV6.48HDIFF.11" "REV6.48HDIFF.12" "REV6.48HDIFF.13"
#> [489] "REV6.48HDIFF.14" "REV6.48HDIFF.15" "REV6.48HDIFF.16" "REV6.48HDIFF.17"
#> [493] "REV6.48HDIFF.18" "REV6.48HDIFF.19" "REV6.48HDIFF.20" "REV6.48HDIFF.21"
#> [497] "REV6.48HDIFF.22" "REV6.48HREV.1"   "REV6.48HREV.2"   "REV6.48HREV.3"  
#> [501] "REV6.48HREV.4"   "REV6.48HREV.5"   "REV6.48HREV.6"   "REV6.48HREV.7"  
#> [505] "REV6.48HREV.8"   "REV6.48HREV.9"   "REV6.48HREV.10"  "REV6.48HREV.11" 
#> [509] "REV6.48HREV.12"  "REV6.48HREV.13"  "REV6.48HREV.14"  "REV6.48HREV.15" 
#> [513] "REV6.48HREV.16"  "REV6.48HREV.17"  "REV6.48HREV.18"  "REV6.48HREV.19" 
#> [517] "REV6.48HREV.20"  "REV6.48HREV.21"  "REV6.48HREV.22"  "REV7.0H.1"      
#> [521] "REV7.0H.2"       "REV7.0H.3"       "REV7.0H.4"       "REV7.0H.5"      
#> [525] "REV7.0H.6"       "REV7.0H.7"       "REV7.0H.8"       "REV7.0H.9"      
#> [529] "REV7.0H.10"      "REV7.0H.11"      "REV7.0H.12"      "REV7.0H.13"     
#> [533] "REV7.0H.14"      "REV7.0H.15"      "REV7.0H.16"      "REV7.0H.17"     
#> [537] "REV7.0H.18"      "REV7.0H.19"      "REV7.0H.20"      "REV7.0H.21"     
#> [541] "REV7.24H.1"      "REV7.24H.2"      "REV7.24H.3"      "REV7.24H.4"     
#> [545] "REV7.24H.5"      "REV7.24H.6"      "REV7.24H.7"      "REV7.24H.8"     
#> [549] "REV7.24H.9"      "REV7.24H.10"     "REV7.24H.11"     "REV7.24H.12"    
#> [553] "REV7.24H.13"     "REV7.24H.14"     "REV7.24H.15"     "REV7.24H.16"    
#> [557] "REV7.24H.17"     "REV7.24H.18"     "REV7.24H.19"     "REV7.24H.20"    
#> [561] "REV7.48HDIFF.1"  "REV7.48HDIFF.2"  "REV7.48HDIFF.3"  "REV7.48HDIFF.4" 
#> [565] "REV7.48HDIFF.5"  "REV7.48HDIFF.6"  "REV7.48HDIFF.7"  "REV7.48HDIFF.8" 
#> [569] "REV7.48HDIFF.9"  "REV7.48HDIFF.10" "REV7.48HDIFF.11" "REV7.48HDIFF.12"
#> [573] "REV7.48HDIFF.13" "REV7.48HDIFF.14" "REV7.48HDIFF.15" "REV7.48HDIFF.16"
#> [577] "REV7.48HDIFF.17" "REV7.48HDIFF.18" "REV7.48HDIFF.19" "REV7.48HDIFF.20"
#> [581] "REV7.48HDIFF.21" "REV7.48HDIFF.22" "REV7.48HDIFF.23" "REV7.48HREV.1"  
#> [585] "REV7.48HREV.2"   "REV7.48HREV.3"   "REV7.48HREV.4"   "REV7.48HREV.5"  
#> [589] "REV7.48HREV.6"   "REV7.48HREV.7"   "REV7.48HREV.8"   "REV7.48HREV.9"  
#> [593] "REV7.48HREV.10"  "REV7.48HREV.11"  "REV7.48HREV.12"  "REV7.48HREV.13" 
#> [597] "REV7.48HREV.14"  "REV7.48HREV.15"  "REV7.48HREV.16"  "REV7.48HREV.17" 
#> [601] "REV7.48HREV.18"  "REV7.48HREV.19"  "REV7.48HREV.20"  "REV8.0H.1"      
#> [605] "REV8.0H.2"       "REV8.0H.3"       "REV8.0H.4"       "REV8.0H.5"      
#> [609] "REV8.0H.6"       "REV8.0H.7"       "REV8.0H.8"       "REV8.0H.9"      
#> [613] "REV8.0H.10"      "REV8.0H.11"      "REV8.0H.12"      "REV8.0H.13"     
#> [617] "REV8.0H.14"      "REV8.0H.15"      "REV8.0H.16"      "REV8.0H.17"     
#> [621] "REV8.0H.18"      "REV8.0H.19"      "REV8.0H.20"      "REV8.0H.21"     
#> [625] "REV8.0H.22"      "REV8.0H.23"      "REV8.0H.24"      "REV8.24H.1"     
#> [629] "REV8.24H.2"      "REV8.24H.3"      "REV8.24H.4"      "REV8.24H.5"     
#> [633] "REV8.24H.6"      "REV8.24H.7"      "REV8.24H.8"      "REV8.24H.9"     
#> [637] "REV8.24H.10"     "REV8.24H.11"     "REV8.24H.12"     "REV8.24H.13"    
#> [641] "REV8.24H.14"     "REV8.24H.15"     "REV8.24H.16"     "REV8.24H.17"    
#> [645] "REV8.24H.18"     "REV8.24H.19"     "REV8.24H.20"     "REV8.48HDIFF.1" 
#> [649] "REV8.48HDIFF.2"  "REV8.48HDIFF.3"  "REV8.48HDIFF.4"  "REV8.48HDIFF.5" 
#> [653] "REV8.48HDIFF.6"  "REV8.48HDIFF.7"  "REV8.48HDIFF.8"  "REV8.48HDIFF.9" 
#> [657] "REV8.48HDIFF.10" "REV8.48HDIFF.11" "REV8.48HDIFF.12" "REV8.48HDIFF.13"
#> [661] "REV8.48HDIFF.14" "REV8.48HDIFF.15" "REV8.48HREV.1"   "REV8.48HREV.2"  
#> [665] "REV8.48HREV.3"   "REV8.48HREV.4"   "REV8.48HREV.5"   "REV8.48HREV.6"  
#> [669] "REV8.48HREV.7"   "REV8.48HREV.8"   "REV8.48HREV.9"   "REV8.48HREV.10" 
#> [673] "REV8.48HREV.11"  "REV8.48HREV.12"  "REV8.48HREV.13"  "REV8.48HREV.14" 
#> [677] "REV8.48HREV.15"  "REV8.48HREV.16"  "REV8.48HREV.17"  "REV8.48HREV.18" 
#> [681] "REV8.48HREV.19"  "REV8.48HREV.20"  "REV8.48HREV.21"  "REV8.48HREV.22" 
#> [685] "REV8.48HREV.23"

meta_sc_df <- read.table("data/metadata2.csv", row.names = 1, sep = ",", header = TRUE)
str(meta_sc_df)
#> 'data.frame':    685 obs. of  3 variables:
#>  $ index    : chr  "REV1.0H.1" "REV1.0H.2" "REV1.0H.3" "REV1.0H.4" ...
#>  $ batch    : chr  "REV1" "REV1" "REV1" "REV1" ...
#>  $ condition: chr  "0H" "0H" "0H" "0H" ...
rownames(meta_sc_df)
#>   [1] "REV1.0H.1"       "REV1.0H.2"       "REV1.0H.3"       "REV1.0H.4"      
#>   [5] "REV1.0H.5"       "REV1.0H.6"       "REV1.0H.7"       "REV1.0H.8"      
#>   [9] "REV1.0H.9"       "REV1.0H.10"      "REV1.0H.11"      "REV1.0H.12"     
#>  [13] "REV1.0H.13"      "REV1.0H.14"      "REV1.0H.15"      "REV1.0H.16"     
#>  [17] "REV1.0H.17"      "REV1.0H.18"      "REV1.0H.19"      "REV1.0H.20"     
#>  [21] "REV1.0H.21"      "REV1.0H.22"      "REV1.0H.23"      "REV1.24H.1"     
#>  [25] "REV1.24H.2"      "REV1.24H.3"      "REV1.24H.4"      "REV1.24H.5"     
#>  [29] "REV1.24H.6"      "REV1.24H.7"      "REV1.24H.8"      "REV1.24H.9"     
#>  [33] "REV1.24H.10"     "REV1.24H.11"     "REV1.24H.12"     "REV1.24H.13"    
#>  [37] "REV1.24H.14"     "REV1.24H.15"     "REV1.24H.16"     "REV1.24H.17"    
#>  [41] "REV1.24H.18"     "REV1.24H.19"     "REV1.24H.20"     "REV1.24H.21"    
#>  [45] "REV1.48HDIFF.1"  "REV1.48HDIFF.2"  "REV1.48HDIFF.3"  "REV1.48HDIFF.4" 
#>  [49] "REV1.48HDIFF.5"  "REV1.48HDIFF.6"  "REV1.48HDIFF.7"  "REV1.48HDIFF.8" 
#>  [53] "REV1.48HDIFF.9"  "REV1.48HDIFF.10" "REV1.48HDIFF.11" "REV1.48HDIFF.12"
#>  [57] "REV1.48HDIFF.13" "REV1.48HDIFF.14" "REV1.48HDIFF.15" "REV1.48HDIFF.16"
#>  [61] "REV1.48HDIFF.17" "REV1.48HDIFF.18" "REV1.48HDIFF.19" "REV1.48HDIFF.20"
#>  [65] "REV1.48HDIFF.21" "REV1.48HDIFF.22" "REV1.48HDIFF.23" "REV1.48HDIFF.24"
#>  [69] "REV1.48HREV.1"   "REV1.48HREV.2"   "REV1.48HREV.3"   "REV1.48HREV.4"  
#>  [73] "REV1.48HREV.5"   "REV1.48HREV.6"   "REV1.48HREV.7"   "REV1.48HREV.8"  
#>  [77] "REV1.48HREV.9"   "REV1.48HREV.10"  "REV1.48HREV.11"  "REV1.48HREV.12" 
#>  [81] "REV1.48HREV.13"  "REV1.48HREV.14"  "REV1.48HREV.15"  "REV1.48HREV.16" 
#>  [85] "REV1.48HREV.17"  "REV1.48HREV.18"  "REV1.48HREV.19"  "REV1.48HREV.20" 
#>  [89] "REV1.48HREV.21"  "REV2.0H.1"       "REV2.0H.2"       "REV2.0H.3"      
#>  [93] "REV2.0H.4"       "REV2.0H.5"       "REV2.0H.6"       "REV2.0H.7"      
#>  [97] "REV2.0H.8"       "REV2.0H.9"       "REV2.0H.10"      "REV2.0H.11"     
#> [101] "REV2.0H.12"      "REV2.0H.13"      "REV2.0H.14"      "REV2.0H.15"     
#> [105] "REV2.0H.16"      "REV2.0H.17"      "REV2.0H.18"      "REV2.0H.19"     
#> [109] "REV2.0H.20"      "REV2.0H.21"      "REV2.0H.22"      "REV2.0H.23"     
#> [113] "REV2.24H.1"      "REV2.24H.2"      "REV2.24H.3"      "REV2.24H.4"     
#> [117] "REV2.24H.5"      "REV2.24H.6"      "REV2.24H.7"      "REV2.24H.8"     
#> [121] "REV2.24H.9"      "REV2.24H.10"     "REV2.24H.11"     "REV2.24H.12"    
#> [125] "REV2.24H.13"     "REV2.24H.14"     "REV2.24H.15"     "REV2.24H.16"    
#> [129] "REV2.24H.17"     "REV2.24H.18"     "REV2.24H.19"     "REV2.24H.20"    
#> [133] "REV2.24H.21"     "REV2.24H.22"     "REV2.24H.23"     "REV2.24H.24"    
#> [137] "REV2.48HDIFF.1"  "REV2.48HDIFF.2"  "REV2.48HDIFF.3"  "REV2.48HDIFF.4" 
#> [141] "REV2.48HDIFF.5"  "REV2.48HDIFF.6"  "REV2.48HDIFF.7"  "REV2.48HDIFF.8" 
#> [145] "REV2.48HDIFF.9"  "REV2.48HDIFF.10" "REV2.48HDIFF.11" "REV2.48HDIFF.12"
#> [149] "REV2.48HDIFF.13" "REV2.48HDIFF.14" "REV2.48HDIFF.15" "REV2.48HDIFF.16"
#> [153] "REV2.48HDIFF.17" "REV2.48HDIFF.18" "REV2.48HDIFF.19" "REV2.48HDIFF.20"
#> [157] "REV2.48HDIFF.21" "REV2.48HDIFF.22" "REV2.48HREV.1"   "REV2.48HREV.2"  
#> [161] "REV2.48HREV.3"   "REV2.48HREV.4"   "REV2.48HREV.5"   "REV2.48HREV.6"  
#> [165] "REV2.48HREV.7"   "REV2.48HREV.8"   "REV2.48HREV.9"   "REV2.48HREV.10" 
#> [169] "REV2.48HREV.11"  "REV2.48HREV.12"  "REV2.48HREV.13"  "REV2.48HREV.14" 
#> [173] "REV2.48HREV.15"  "REV2.48HREV.16"  "REV2.48HREV.17"  "REV2.48HREV.18" 
#> [177] "REV2.48HREV.19"  "REV2.48HREV.20"  "REV2.48HREV.21"  "REV3.0H.1"      
#> [181] "REV3.0H.2"       "REV3.0H.3"       "REV3.0H.4"       "REV3.0H.5"      
#> [185] "REV3.0H.6"       "REV3.0H.7"       "REV3.0H.8"       "REV3.0H.9"      
#> [189] "REV3.0H.10"      "REV3.0H.11"      "REV3.0H.12"      "REV3.0H.13"     
#> [193] "REV3.0H.14"      "REV3.0H.15"      "REV3.0H.16"      "REV3.0H.17"     
#> [197] "REV3.0H.18"      "REV3.0H.19"      "REV3.0H.20"      "REV3.0H.21"     
#> [201] "REV3.24H.1"      "REV3.24H.2"      "REV3.24H.3"      "REV3.24H.4"     
#> [205] "REV3.24H.5"      "REV3.24H.6"      "REV3.24H.7"      "REV3.24H.8"     
#> [209] "REV3.24H.9"      "REV3.24H.10"     "REV3.24H.11"     "REV3.24H.12"    
#> [213] "REV3.24H.13"     "REV3.24H.14"     "REV3.24H.15"     "REV3.24H.16"    
#> [217] "REV3.24H.17"     "REV3.24H.18"     "REV3.24H.19"     "REV3.24H.20"    
#> [221] "REV3.24H.21"     "REV3.24H.22"     "REV3.24H.23"     "REV3.24H.24"    
#> [225] "REV3.48HDIFF.1"  "REV3.48HDIFF.2"  "REV3.48HDIFF.3"  "REV3.48HDIFF.4" 
#> [229] "REV3.48HDIFF.5"  "REV3.48HDIFF.6"  "REV3.48HDIFF.7"  "REV3.48HDIFF.8" 
#> [233] "REV3.48HDIFF.9"  "REV3.48HDIFF.10" "REV3.48HDIFF.11" "REV3.48HDIFF.12"
#> [237] "REV3.48HDIFF.13" "REV3.48HDIFF.14" "REV3.48HDIFF.15" "REV3.48HDIFF.16"
#> [241] "REV3.48HDIFF.17" "REV3.48HDIFF.18" "REV3.48HDIFF.19" "REV3.48HDIFF.20"
#> [245] "REV3.48HDIFF.21" "REV3.48HDIFF.22" "REV3.48HREV.1"   "REV3.48HREV.2"  
#> [249] "REV3.48HREV.3"   "REV3.48HREV.4"   "REV3.48HREV.5"   "REV3.48HREV.6"  
#> [253] "REV3.48HREV.7"   "REV3.48HREV.8"   "REV3.48HREV.9"   "REV3.48HREV.10" 
#> [257] "REV3.48HREV.11"  "REV3.48HREV.12"  "REV3.48HREV.13"  "REV3.48HREV.14" 
#> [261] "REV3.48HREV.15"  "REV3.48HREV.16"  "REV3.48HREV.17"  "REV3.48HREV.18" 
#> [265] "REV3.48HREV.19"  "REV3.48HREV.20"  "REV3.48HREV.21"  "REV3.48HREV.22" 
#> [269] "REV4.0H.1"       "REV4.0H.2"       "REV4.0H.3"       "REV4.0H.4"      
#> [273] "REV4.0H.5"       "REV4.0H.6"       "REV4.0H.7"       "REV4.0H.8"      
#> [277] "REV4.0H.9"       "REV4.0H.10"      "REV4.0H.11"      "REV4.0H.12"     
#> [281] "REV4.0H.13"      "REV4.0H.14"      "REV4.0H.15"      "REV4.0H.16"     
#> [285] "REV4.0H.17"      "REV4.0H.18"      "REV4.24H.1"      "REV4.24H.2"     
#> [289] "REV4.24H.3"      "REV4.24H.4"      "REV4.24H.5"      "REV4.24H.6"     
#> [293] "REV4.24H.7"      "REV4.24H.8"      "REV4.24H.9"      "REV4.24H.10"    
#> [297] "REV4.24H.11"     "REV4.24H.12"     "REV4.24H.13"     "REV4.24H.14"    
#> [301] "REV4.24H.15"     "REV4.24H.16"     "REV4.24H.17"     "REV4.24H.18"    
#> [305] "REV4.24H.19"     "REV4.24H.20"     "REV4.24H.21"     "REV4.48HDIFF.1" 
#> [309] "REV4.48HDIFF.2"  "REV4.48HDIFF.3"  "REV4.48HDIFF.4"  "REV4.48HDIFF.5" 
#> [313] "REV4.48HDIFF.6"  "REV4.48HDIFF.7"  "REV4.48HDIFF.8"  "REV4.48HDIFF.9" 
#> [317] "REV4.48HDIFF.10" "REV4.48HDIFF.11" "REV4.48HDIFF.12" "REV4.48HDIFF.13"
#> [321] "REV4.48HDIFF.14" "REV4.48HDIFF.15" "REV4.48HDIFF.16" "REV4.48HDIFF.17"
#> [325] "REV4.48HDIFF.18" "REV4.48HDIFF.19" "REV4.48HREV.1"   "REV4.48HREV.2"  
#> [329] "REV4.48HREV.3"   "REV4.48HREV.4"   "REV4.48HREV.5"   "REV4.48HREV.6"  
#> [333] "REV4.48HREV.7"   "REV4.48HREV.8"   "REV4.48HREV.9"   "REV4.48HREV.10" 
#> [337] "REV4.48HREV.11"  "REV4.48HREV.12"  "REV4.48HREV.13"  "REV4.48HREV.14" 
#> [341] "REV4.48HREV.15"  "REV4.48HREV.16"  "REV4.48HREV.17"  "REV4.48HREV.18" 
#> [345] "REV4.48HREV.19"  "REV5.0H.1"       "REV5.0H.2"       "REV5.0H.3"      
#> [349] "REV5.0H.4"       "REV5.0H.5"       "REV5.0H.6"       "REV5.0H.7"      
#> [353] "REV5.0H.8"       "REV5.0H.9"       "REV5.0H.10"      "REV5.0H.11"     
#> [357] "REV5.0H.12"      "REV5.0H.13"      "REV5.0H.14"      "REV5.0H.15"     
#> [361] "REV5.0H.16"      "REV5.0H.17"      "REV5.0H.18"      "REV5.0H.19"     
#> [365] "REV5.0H.20"      "REV5.0H.21"      "REV5.0H.22"      "REV5.0H.23"     
#> [369] "REV5.24H.1"      "REV5.24H.2"      "REV5.24H.3"      "REV5.24H.4"     
#> [373] "REV5.24H.5"      "REV5.24H.6"      "REV5.24H.7"      "REV5.24H.8"     
#> [377] "REV5.24H.9"      "REV5.24H.10"     "REV5.24H.11"     "REV5.24H.12"    
#> [381] "REV5.24H.13"     "REV5.24H.14"     "REV5.24H.15"     "REV5.24H.16"    
#> [385] "REV5.24H.17"     "REV5.24H.18"     "REV5.24H.19"     "REV5.24H.20"    
#> [389] "REV5.24H.21"     "REV5.48HDIFF.1"  "REV5.48HDIFF.2"  "REV5.48HDIFF.3" 
#> [393] "REV5.48HDIFF.4"  "REV5.48HDIFF.5"  "REV5.48HDIFF.6"  "REV5.48HDIFF.7" 
#> [397] "REV5.48HDIFF.8"  "REV5.48HDIFF.9"  "REV5.48HDIFF.10" "REV5.48HDIFF.11"
#> [401] "REV5.48HDIFF.12" "REV5.48HDIFF.13" "REV5.48HDIFF.14" "REV5.48HDIFF.15"
#> [405] "REV5.48HDIFF.16" "REV5.48HDIFF.17" "REV5.48HDIFF.18" "REV5.48HDIFF.19"
#> [409] "REV5.48HDIFF.20" "REV5.48HDIFF.21" "REV5.48HREV.1"   "REV5.48HREV.2"  
#> [413] "REV5.48HREV.3"   "REV5.48HREV.4"   "REV5.48HREV.5"   "REV5.48HREV.6"  
#> [417] "REV5.48HREV.7"   "REV5.48HREV.8"   "REV5.48HREV.9"   "REV5.48HREV.10" 
#> [421] "REV5.48HREV.11"  "REV5.48HREV.12"  "REV5.48HREV.13"  "REV5.48HREV.14" 
#> [425] "REV5.48HREV.15"  "REV5.48HREV.16"  "REV5.48HREV.17"  "REV5.48HREV.18" 
#> [429] "REV5.48HREV.19"  "REV5.48HREV.20"  "REV5.48HREV.21"  "REV5.48HREV.22" 
#> [433] "REV5.48HREV.23"  "REV6.0H.1"       "REV6.0H.2"       "REV6.0H.3"      
#> [437] "REV6.0H.4"       "REV6.0H.5"       "REV6.0H.6"       "REV6.0H.7"      
#> [441] "REV6.0H.8"       "REV6.0H.9"       "REV6.0H.10"      "REV6.0H.11"     
#> [445] "REV6.0H.12"      "REV6.0H.13"      "REV6.0H.14"      "REV6.0H.15"     
#> [449] "REV6.0H.16"      "REV6.0H.17"      "REV6.0H.18"      "REV6.0H.19"     
#> [453] "REV6.0H.20"      "REV6.24H.1"      "REV6.24H.2"      "REV6.24H.3"     
#> [457] "REV6.24H.4"      "REV6.24H.5"      "REV6.24H.6"      "REV6.24H.7"     
#> [461] "REV6.24H.8"      "REV6.24H.9"      "REV6.24H.10"     "REV6.24H.11"    
#> [465] "REV6.24H.12"     "REV6.24H.13"     "REV6.24H.14"     "REV6.24H.15"    
#> [469] "REV6.24H.16"     "REV6.24H.17"     "REV6.24H.18"     "REV6.24H.19"    
#> [473] "REV6.24H.20"     "REV6.24H.21"     "REV6.24H.22"     "REV6.48HDIFF.1" 
#> [477] "REV6.48HDIFF.2"  "REV6.48HDIFF.3"  "REV6.48HDIFF.4"  "REV6.48HDIFF.5" 
#> [481] "REV6.48HDIFF.6"  "REV6.48HDIFF.7"  "REV6.48HDIFF.8"  "REV6.48HDIFF.9" 
#> [485] "REV6.48HDIFF.10" "REV6.48HDIFF.11" "REV6.48HDIFF.12" "REV6.48HDIFF.13"
#> [489] "REV6.48HDIFF.14" "REV6.48HDIFF.15" "REV6.48HDIFF.16" "REV6.48HDIFF.17"
#> [493] "REV6.48HDIFF.18" "REV6.48HDIFF.19" "REV6.48HDIFF.20" "REV6.48HDIFF.21"
#> [497] "REV6.48HDIFF.22" "REV6.48HREV.1"   "REV6.48HREV.2"   "REV6.48HREV.3"  
#> [501] "REV6.48HREV.4"   "REV6.48HREV.5"   "REV6.48HREV.6"   "REV6.48HREV.7"  
#> [505] "REV6.48HREV.8"   "REV6.48HREV.9"   "REV6.48HREV.10"  "REV6.48HREV.11" 
#> [509] "REV6.48HREV.12"  "REV6.48HREV.13"  "REV6.48HREV.14"  "REV6.48HREV.15" 
#> [513] "REV6.48HREV.16"  "REV6.48HREV.17"  "REV6.48HREV.18"  "REV6.48HREV.19" 
#> [517] "REV6.48HREV.20"  "REV6.48HREV.21"  "REV6.48HREV.22"  "REV7.0H.1"      
#> [521] "REV7.0H.2"       "REV7.0H.3"       "REV7.0H.4"       "REV7.0H.5"      
#> [525] "REV7.0H.6"       "REV7.0H.7"       "REV7.0H.8"       "REV7.0H.9"      
#> [529] "REV7.0H.10"      "REV7.0H.11"      "REV7.0H.12"      "REV7.0H.13"     
#> [533] "REV7.0H.14"      "REV7.0H.15"      "REV7.0H.16"      "REV7.0H.17"     
#> [537] "REV7.0H.18"      "REV7.0H.19"      "REV7.0H.20"      "REV7.0H.21"     
#> [541] "REV7.24H.1"      "REV7.24H.2"      "REV7.24H.3"      "REV7.24H.4"     
#> [545] "REV7.24H.5"      "REV7.24H.6"      "REV7.24H.7"      "REV7.24H.8"     
#> [549] "REV7.24H.9"      "REV7.24H.10"     "REV7.24H.11"     "REV7.24H.12"    
#> [553] "REV7.24H.13"     "REV7.24H.14"     "REV7.24H.15"     "REV7.24H.16"    
#> [557] "REV7.24H.17"     "REV7.24H.18"     "REV7.24H.19"     "REV7.24H.20"    
#> [561] "REV7.48HDIFF.1"  "REV7.48HDIFF.2"  "REV7.48HDIFF.3"  "REV7.48HDIFF.4" 
#> [565] "REV7.48HDIFF.5"  "REV7.48HDIFF.6"  "REV7.48HDIFF.7"  "REV7.48HDIFF.8" 
#> [569] "REV7.48HDIFF.9"  "REV7.48HDIFF.10" "REV7.48HDIFF.11" "REV7.48HDIFF.12"
#> [573] "REV7.48HDIFF.13" "REV7.48HDIFF.14" "REV7.48HDIFF.15" "REV7.48HDIFF.16"
#> [577] "REV7.48HDIFF.17" "REV7.48HDIFF.18" "REV7.48HDIFF.19" "REV7.48HDIFF.20"
#> [581] "REV7.48HDIFF.21" "REV7.48HDIFF.22" "REV7.48HDIFF.23" "REV7.48HREV.1"  
#> [585] "REV7.48HREV.2"   "REV7.48HREV.3"   "REV7.48HREV.4"   "REV7.48HREV.5"  
#> [589] "REV7.48HREV.6"   "REV7.48HREV.7"   "REV7.48HREV.8"   "REV7.48HREV.9"  
#> [593] "REV7.48HREV.10"  "REV7.48HREV.11"  "REV7.48HREV.12"  "REV7.48HREV.13" 
#> [597] "REV7.48HREV.14"  "REV7.48HREV.15"  "REV7.48HREV.16"  "REV7.48HREV.17" 
#> [601] "REV7.48HREV.18"  "REV7.48HREV.19"  "REV7.48HREV.20"  "REV8.0H.1"      
#> [605] "REV8.0H.2"       "REV8.0H.3"       "REV8.0H.4"       "REV8.0H.5"      
#> [609] "REV8.0H.6"       "REV8.0H.7"       "REV8.0H.8"       "REV8.0H.9"      
#> [613] "REV8.0H.10"      "REV8.0H.11"      "REV8.0H.12"      "REV8.0H.13"     
#> [617] "REV8.0H.14"      "REV8.0H.15"      "REV8.0H.16"      "REV8.0H.17"     
#> [621] "REV8.0H.18"      "REV8.0H.19"      "REV8.0H.20"      "REV8.0H.21"     
#> [625] "REV8.0H.22"      "REV8.0H.23"      "REV8.0H.24"      "REV8.24H.1"     
#> [629] "REV8.24H.2"      "REV8.24H.3"      "REV8.24H.4"      "REV8.24H.5"     
#> [633] "REV8.24H.6"      "REV8.24H.7"      "REV8.24H.8"      "REV8.24H.9"     
#> [637] "REV8.24H.10"     "REV8.24H.11"     "REV8.24H.12"     "REV8.24H.13"    
#> [641] "REV8.24H.14"     "REV8.24H.15"     "REV8.24H.16"     "REV8.24H.17"    
#> [645] "REV8.24H.18"     "REV8.24H.19"     "REV8.24H.20"     "REV8.48HDIFF.1" 
#> [649] "REV8.48HDIFF.2"  "REV8.48HDIFF.3"  "REV8.48HDIFF.4"  "REV8.48HDIFF.5" 
#> [653] "REV8.48HDIFF.6"  "REV8.48HDIFF.7"  "REV8.48HDIFF.8"  "REV8.48HDIFF.9" 
#> [657] "REV8.48HDIFF.10" "REV8.48HDIFF.11" "REV8.48HDIFF.12" "REV8.48HDIFF.13"
#> [661] "REV8.48HDIFF.14" "REV8.48HDIFF.15" "REV8.48HREV.1"   "REV8.48HREV.2"  
#> [665] "REV8.48HREV.3"   "REV8.48HREV.4"   "REV8.48HREV.5"   "REV8.48HREV.6"  
#> [669] "REV8.48HREV.7"   "REV8.48HREV.8"   "REV8.48HREV.9"   "REV8.48HREV.10" 
#> [673] "REV8.48HREV.11"  "REV8.48HREV.12"  "REV8.48HREV.13"  "REV8.48HREV.14" 
#> [677] "REV8.48HREV.15"  "REV8.48HREV.16"  "REV8.48HREV.17"  "REV8.48HREV.18" 
#> [681] "REV8.48HREV.19"  "REV8.48HREV.20"  "REV8.48HREV.21"  "REV8.48HREV.22" 
#> [685] "REV8.48HREV.23"
```

Init test:

``` r
kt <- ktest(
    sc_df, meta_sc_df,
    condition='condition',
    samples=c('0H','48HREV'),
    verbose=1)
```

Perform multivariate testing

``` r
kt$multivariate_test(verbose=1)
multivariate_test(kt, verbose=1)
```

Multivariate testing result:

``` r
kt$print_multivariate_test_results(long = TRUE, ts = c(1,2,3))
print_multivariate_test_results(kt, long = TRUE, ts = c(1,2,3))
```

Get p-values:

``` r
kt$get_pvalue(contrib = TRUE, log = FALSE)
#>           1             2             3             4             5   
#>  6.117177e-01  1.246529e-08  2.552708e-08  1.884499e-07  1.101084e-01 
#>           6             7             8             9             10  
#>  4.584441e-01  9.999931e-01  9.998929e-01  1.000000e+00  9.741743e-01 
#>           11            12            13            14            15  
#>  9.999404e-01  7.447209e-01  4.697539e-01  9.999987e-01  1.000000e+00 
#>           16            17            18            19            20  
#>  1.000000e+00  9.953514e-01  1.000000e+00  9.989000e-01  1.000000e+00 
#>           21            22            23            24            25  
#>  1.000000e+00  9.975742e-01  9.998560e-01  3.919685e-01  1.000000e+00 
#>           26            27            28            29            30  
#>  1.000000e+00  1.000000e+00  1.213503e-01  1.000000e+00  1.000000e+00 
#>           31            32            33            34            35  
#>  1.000000e+00  1.000000e+00  1.000000e+00  4.444550e-03  1.000000e+00 
#>           36            37            38            39            40  
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           41            42            43            44            45  
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           46            47            48            49            50  
#>  1.000000e+00  1.000000e+00  9.999997e-01  1.000000e+00  1.000000e+00 
#>           51            52            53            54            55  
#>  9.999880e-01  1.000000e+00  9.999879e-01  1.000000e+00  1.000000e+00 
#>           56            57            58            59            60  
#>  9.999717e-01  1.000000e+00  1.000000e+00  9.999966e-01  9.999999e-01 
#>           61            62            63            64            65  
#>  1.000000e+00  9.935994e-01  1.000000e+00  1.000000e+00  1.000000e+00 
#>           66            67            68            69            70  
#>  9.837601e-05  1.000000e+00  4.588575e-04  1.000000e+00  1.000000e+00 
#>           71            72            73            74            75  
#>  9.999891e-01  2.529133e-01  1.000000e+00  1.000000e+00  1.000000e+00 
#>           76            77            78            79            80  
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           81            82            83            84            85  
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           86            87            88            89            90  
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           91            92            93            94            95  
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           96            97            98            99            100 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           101           102           103           104           105 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           106           107           108           109           110 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           111           112           113           114           115 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           116           117           118           119           120 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           121           122           123           124           125 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           126           127           128           129           130 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           131           132           133           134           135 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           136           137           138           139           140 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           141           142           143           144           145 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           146           147           148           149           150 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           151           152           153           154           155 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           156           157           158           159           160 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           161           162           163           164           165 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           166           167           168           169           170 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           171           172           173           174           175 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           176           177           178           179           180 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           181           182           183           184           185 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           186           187           188           189           190 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           191           192           193           194           195 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           196           197           198           199           200 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           201           202           203           204           205 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           206           207           208           209           210 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           211           212           213           214           215 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           216           217           218           219           220 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           221           222           223           224           225 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           226           227           228           229           230 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           231           232           233           234           235 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           236           237           238           239           240 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           241           242           243           244           245 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           246           247           248           249           250 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           251           252           253           254           255 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           256           257           258           259           260 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           261           262           263           264           265 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           266           267           268           269           270 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           271           272           273           274           275 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           276           277           278           279           280 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           281           282           283           284           285 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           286           287           288           289           290 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           291           292           293           294           295 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           296           297           298           299           300 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           301           302           303           304           305 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           306           307           308           309           310 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           311           312           313           314           315 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           316           317           318           319           320 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           321           322           323           324           325 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           326           327           328           329           330 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           331           332           333           334           335 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  9.996041e-01 
#>           336           337           338           339           340 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           341           342           343           344 
#>  2.100686e-01 7.242913e-242  0.000000e+00  1.000000e+00
get_pvalue(kt, contrib = TRUE, log = FALSE)
#>           1             2             3             4             5   
#>  6.117177e-01  1.246529e-08  2.552708e-08  1.884499e-07  1.101084e-01 
#>           6             7             8             9             10  
#>  4.584441e-01  9.999931e-01  9.998929e-01  1.000000e+00  9.741743e-01 
#>           11            12            13            14            15  
#>  9.999404e-01  7.447209e-01  4.697539e-01  9.999987e-01  1.000000e+00 
#>           16            17            18            19            20  
#>  1.000000e+00  9.953514e-01  1.000000e+00  9.989000e-01  1.000000e+00 
#>           21            22            23            24            25  
#>  1.000000e+00  9.975742e-01  9.998560e-01  3.919685e-01  1.000000e+00 
#>           26            27            28            29            30  
#>  1.000000e+00  1.000000e+00  1.213503e-01  1.000000e+00  1.000000e+00 
#>           31            32            33            34            35  
#>  1.000000e+00  1.000000e+00  1.000000e+00  4.444550e-03  1.000000e+00 
#>           36            37            38            39            40  
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           41            42            43            44            45  
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           46            47            48            49            50  
#>  1.000000e+00  1.000000e+00  9.999997e-01  1.000000e+00  1.000000e+00 
#>           51            52            53            54            55  
#>  9.999880e-01  1.000000e+00  9.999879e-01  1.000000e+00  1.000000e+00 
#>           56            57            58            59            60  
#>  9.999717e-01  1.000000e+00  1.000000e+00  9.999966e-01  9.999999e-01 
#>           61            62            63            64            65  
#>  1.000000e+00  9.935994e-01  1.000000e+00  1.000000e+00  1.000000e+00 
#>           66            67            68            69            70  
#>  9.837601e-05  1.000000e+00  4.588575e-04  1.000000e+00  1.000000e+00 
#>           71            72            73            74            75  
#>  9.999891e-01  2.529133e-01  1.000000e+00  1.000000e+00  1.000000e+00 
#>           76            77            78            79            80  
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           81            82            83            84            85  
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           86            87            88            89            90  
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           91            92            93            94            95  
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           96            97            98            99            100 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           101           102           103           104           105 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           106           107           108           109           110 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           111           112           113           114           115 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           116           117           118           119           120 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           121           122           123           124           125 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           126           127           128           129           130 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           131           132           133           134           135 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           136           137           138           139           140 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           141           142           143           144           145 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           146           147           148           149           150 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           151           152           153           154           155 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           156           157           158           159           160 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           161           162           163           164           165 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           166           167           168           169           170 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           171           172           173           174           175 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           176           177           178           179           180 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           181           182           183           184           185 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           186           187           188           189           190 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           191           192           193           194           195 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           196           197           198           199           200 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           201           202           203           204           205 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           206           207           208           209           210 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           211           212           213           214           215 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           216           217           218           219           220 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           221           222           223           224           225 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           226           227           228           229           230 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           231           232           233           234           235 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           236           237           238           239           240 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           241           242           243           244           245 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           246           247           248           249           250 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           251           252           253           254           255 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           256           257           258           259           260 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           261           262           263           264           265 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           266           267           268           269           270 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           271           272           273           274           275 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           276           277           278           279           280 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           281           282           283           284           285 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           286           287           288           289           290 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           291           292           293           294           295 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           296           297           298           299           300 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           301           302           303           304           305 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           306           307           308           309           310 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           311           312           313           314           315 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           316           317           318           319           320 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           321           322           323           324           325 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           326           327           328           329           330 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           331           332           333           334           335 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  9.996041e-01 
#>           336           337           338           339           340 
#>  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00 
#>           341           342           343           344 
#>  2.100686e-01 7.242913e-242  0.000000e+00  1.000000e+00
```
