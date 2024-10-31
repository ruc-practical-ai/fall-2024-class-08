import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA


def compute_gaussianity_measure(x_features_transformed):
    gaussianity_measures = np.abs(1 / kurtosis(x_features_transformed, axis=0))
    return gaussianity_measures


def compute_peak_to_average_gaussianity_measure(
    compute_gaussianity_measure, x_features_transformed
):
    gaussianity_measures = compute_gaussianity_measure(x_features_transformed)
    peak_to_avg_gaussianity_measure = np.max(
        gaussianity_measures
    ) / np.average(gaussianity_measures)

    return peak_to_avg_gaussianity_measure


def perform_n_components_search(x_features_scaled):
    """Help search for right number of components using a Gaussianity metric.

    When one component is much more Gaussian than the others, this indicates
    that we have the correct number of components specified.
    """
    n_components_candidates = np.arange(2, 30)
    peak_to_avg_gaussianity_measures = []
    for n_components in n_components_candidates:
        ica = FastICA(
            n_components=n_components,
            whiten="arbitrary-variance",
            tol=1e-7,
            max_iter=500,
            random_state=42,
        )
        x_features_transformed = ica.fit_transform(x_features_scaled)

        peak_to_avg_gaussianity_measure = (
            compute_peak_to_average_gaussianity_measure(
                compute_gaussianity_measure, x_features_transformed
            )
        )
        peak_to_avg_gaussianity_measures.append(
            peak_to_avg_gaussianity_measure
        )
        print(
            "Checked n = {}, peak to avg. Gaussianity ratio = {}".format(
                n_components, peak_to_avg_gaussianity_measure
            )
        )
    best_n_components = np.argmax(peak_to_avg_gaussianity_measures)
    return best_n_components


def get_noise_component(x_features_transformed):
    gaussianity_measures = compute_gaussianity_measure(x_features_transformed)
    noise_component_index = np.argmax(gaussianity_measures)
    return noise_component_index


def rebuild_features_without_noise(
    ica, scaler, x_features, x_features_transformed
):
    noise_component_index = get_noise_component(x_features_transformed)

    x_features_transformed[:, noise_component_index] = 0

    x_features_reconstructed = ica.inverse_transform(x_features_transformed)
    x_smooth_original_scale = scaler.inverse_transform(
        x_features_reconstructed
    )

    delta = x_features - x_smooth_original_scale
    return x_smooth_original_scale, delta
