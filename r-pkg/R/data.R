#' Load example dataset shipped with the package
#' 
#' @description
#' Dataset containing single cell gene expression data and metadata 
#' (i.e. corresponding experimental conditions).
#' 
#' @details
#' This dataset originates from a study that investigated the molecular 
#' mechanisms underlying cell differentiation and reversion, by measuring cell 
#' transcriptomes at four time points: undifferentiated T2EC maintained in a 
#' self-renewal medium (condition `"0H"`), then put in a 
#' differentiation-inducing medium for 24h (condition `"24H"`). This population 
#' was then split into a first population maintained in the same medium for 
#' another 24h to achieve differentiation (condition `"48HDIFF"`), and the 
#' second population was put back in the self-renewal medium to investigate 
#' potential reversion (condition `"48HREV"`). Cell transcriptomes were 
#' measured using scRT-qPCR on 83 genes selected to be involved in the 
#' differentiation process.
#' 
#' See Zreika et al (2022) and Ozier-Lafontaine et al (2024) for more details.
#' 
#' The example dataset contains the samples for the condition `"0H"` and 
#' `"48HREV"`.
#' 
#' See <https://github.com/LMJL-Alea/ktest/tree/main/tutorials/v5_data> for 
#' the entire dataset. 
#' 
#' @return a list containing two data tables (`data.frame`):
#' - `data_tab` with the gene expression measurements (in columns) for the 
#' different cells (in rows);
#' - `metadata_tab` with the corresponding condition labels.
#' 
#' @export
#' 
#' @importFrom dplyr %>% select
#' @importFrom fs path_package
#' @importFrom readr read_csv
#' @importFrom tibble lst
#' 
#' @references
#' Zreika S., Fourneaux C., Vallin E., Modolo L., Seraphin R., Moussy A., 
#' Ventre E., Bouvier M., Ozier-Lafontaine A., Bonnaffoux A., Picard F., 
#' Gandrillon O., Gonin-Giraud S. (2022 Jul 6). 
#' Evidence for close molecular proximity between reverting and 
#' undifferentiated cells. BMC Biol. 20(1):155.
#' [doi:10.1186/s12915-022-01363-7](https://dx.doi.org/10.1186/s12915-022-01363-7);
#' [PMID: 35794592](https://pubmed.ncbi.nlm.nih.gov/35794592/);
#' [PMCID: PMC9258043](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9258043/);
#' [hal-04134084v1](https://hal.science/hal-04134084v1).
#' 
#' Ozier-Lafontaine A., Fourneaux C., Durif G., Arsenteva P., Vallot C., 
#' Gandrillon O., Gonin-Giraud S., Michel B., Picard F. (2024). 
#' Kernel-Based Testing for Single-Cell Differential Analysis. Preprint. 
#' \doi{10.48550/arXiv.2307.08509};
#' [arXiv.2307.08509](https://arxiv.org/abs/2307.08509);
#' [hal-04214858](https://hal.science/hal-04214858).
#'
#' @examples
#' # data loading
#' tmp <- load_example_data()
#' # gene expression data table (344 cells and 83 genes)
#' data_tab <- tmp$data_tab
#' # metadata table with sampling conditions (for the 344 cells)
#' metadata_tab <- tmp$metadata_tab
load_example_data <- function() {
    # data directory
    data_dir <- file.path(fs::path_package("ktest"), "extdata")
    # expression data table
    data_tab <- readr::read_csv(file.path(data_dir, "data.csv")) %>%
        dplyr::select(!1)
    # metadata table
    metadata_tab <- readr::read_csv(file.path(data_dir, "metadata.csv")) %>%
        dplyr::select(condition)
    # output
    return(tibble::lst(data_tab, metadata_tab))
}