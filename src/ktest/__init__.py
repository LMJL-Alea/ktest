from .utils import ordered_eigsy

from .kernel_operations import \
    compute_gram, \
    center_gram_matrix_with_respect_to_some_effects, \
    compute_kmn, \
    center_kmn_matrix_with_respect_to_some_effects,\
    diagonalize_within_covariance_centered_gram,\
    compute_within_covariance_centered_gram

from .centering_operations import \
    compute_centering_matrix_with_respect_to_some_effects, \
    compute_omega, \
    compute_covariance_centering_matrix

from .nystrom_operations import \
    compute_nystrom_anchors, \
    compute_nystrom_landmarks,\
    compute_quantization_weights,\
    reinitialize_landmarks,\
    reinitialize_anchors

from .statistics import \
    get_trace,\
    compute_kfdat,\
    compute_kfdat_with_different_order,\
    get_explained_variance,\
    compute_pkm,\
    compute_upk,\
    initialize_kfdat,\
    kfdat,\
    kpca,\
    initialize_mmd,\
    mmd,\
    compute_mmd

from .projection_operations import \
    compute_proj_kfda,\
    compute_proj_kpca,\
    init_df_proj,\
    compute_proj_mmd

from .correlation_operations import \
    compute_corr_proj_var,\
    find_correlated_variables

from .visualizations import \
    plot_kfdat,\
    init_plot_kfdat,\
    init_plot_pvalue,\
    plot_pvalue,\
    plot_kfdat_contrib,\
    plot_spectrum,\
    density_proj,\
    scatter_proj,\
    init_axes_projs,\
    density_projs,\
    scatter_projs,\
    get_color_for_scatter,\
    get_plot_properties,\
    plot_correlation_proj_var,\
    plot_pval_with_respect_to_within_covariance_reconstruction_error,\
    plot_pval_with_respect_to_between_covariance_reconstruction_error,\
    plot_relative_reconstruction_errors,\
    plot_ratio_reconstruction_errors,\
    plot_within_covariance_reconstruction_error_with_respect_to_t,\
    plot_between_covariance_reconstruction_error_with_respect_to_t,\
    plot_pval_and_errors,\
    what_if_we_ignored_cells_by_condition,\
    what_if_we_ignored_cells_by_outliers_list,\
    prepare_visualization,\
    visualize_patient_celltypes_CRCL,\
    visualize_quality_CRCL,\
    visualize_effect_graph_CRCL
    
from .initializations import \
    init_data,\
    init_model,\
    init_kernel,\
    set_center_by,\
    init_xy,\
    init_index_xy,\
    init_variables,\
    init_metadata,\
    init_data_from_dataframe,\
    verbosity


from .residuals import \
    compute_discriminant_axis_qh,\
    project_on_discriminant_axis,\
    compute_proj_on_discriminant_orthogonal,\
    compute_residual_covariance,\
    diagonalize_residual_covariance,\
    proj_residus,\
    get_between_covariance_projection_error,\
    get_between_covariance_projection_error_associated_to_t,\
    get_ordered_spectrum_wrt_between_covariance_projection_error
from .truncation_selection import \
    select_trunc_by_between_reconstruction_ratio,\
    select_trunc_by_between_reconstruction_ressaut,\
    select_trunc

from .univariate_testing import \
    univariate_kfda,\
    parallel_univariate_kfda,\
    update_var_from_dataframe,\
    save_univariate_test_results_in_var,\
    load_univariate_test_results_in_var,\
    visualize_univariate_test_CRCL,\
    plot_density_of_variable,\
    get_zero_proportions_of_variable,\
    add_zero_proportions_to_var,\
    volcano_plot,\
    volcano_plot_zero_pvals_and_non_zero_pvals,\
    color_volcano_plot

from .pvalues import \
    compute_pval,\
    correct_BenjaminiHochberg_pval,\
    correct_BenjaminiHochberg_pval_univariate,\
    get_rejected_variables_univariate
    