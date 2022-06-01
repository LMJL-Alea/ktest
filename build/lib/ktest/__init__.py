from .utils import ordered_eigsy

from .kernel_operations import \
    compute_gram, \
    center_gram_matrix_with_respect_to_some_effects, \
    compute_kmn, \
    diagonalize_centered_gram,\
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
    get_trunc,\
    get_95variance_trunc,\
    compute_kfdat,\
    compute_kfdat_with_different_order,\
    get_explained_variance,\
    compute_pval,\
    correct_BenjaminiHochberg_pval,\
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
    set_color_for_scatter,\
    find_cells_from_proj,\
    plot_correlation_proj_var,\
    plot_pval_with_respect_to_within_covariance_reconstruction_error,\
    plot_pval_with_respect_to_between_covariance_reconstruction_error,\
    plot_relative_reconstruction_errors,\
    plot_ratio_reconstruction_errors,\
    plot_within_covariance_reconstruction_error_with_respect_to_t,\
    plot_between_covariance_reconstruction_error_with_respect_to_t

from .initializations import \
    init_data,\
    init_model,\
    init_kernel,\
    set_center_by,\
    init_xy,\
    init_index_xy,\
    init_variables,\
    init_masks,\
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
    get_ordered_spectrum_wrt_between_covariance_projection_error