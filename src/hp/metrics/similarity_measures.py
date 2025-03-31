import numpy as np
from itertools import combinations
from scipy.spatial.transform import Rotation

from hp.utils.rotation_tools import batch_geodesic_distance_matrix


def get_correlation_from_rv_on_p_sphere(
    X: np.ndarray, Y: np.ndarray, p: int = 3, large_sample_distribution: bool = False
) -> float:
    """Get correlation coefficient from realizations of two random variables on p-sphere.
    X.shape = (n,p+1)
    Y.shape = (n,p+1)
    X,Y: parametrized in higherdim. space. E.g. for p=1: [cos(x), sin(x)]
    """
    X, Y = X.T, Y.T
    if X.shape[0] - 1 is not p:
        raise ValueError(
            f"Shape of X is not correct: Is {X.shape} but should be ({p+1},n)."
        )
    if Y.shape[0] - 1 is not p:
        raise ValueError(
            f"Shape of Y is not correct: Is {Y.shape} but should be ({p+1},n)."
        )
    if not X.shape == Y.shape:
        raise ValueError(f"X and Y do not have the same shape: {X.shape}, {Y.shape}")

    n = X.shape[1]

    if not large_sample_distribution:
        detX = np.maximum(np.linalg.det(1 / n * X @ X.T), 1e-20)
        detY = np.maximum(np.linalg.det(1 / n * Y @ Y.T), 1e-20)
        detXY = np.maximum(np.linalg.det(1 / n * X @ Y.T), 1e-20)
        if (detX == detY) and (detX == detXY):
            rho = 1
        else:
            rho = (detXY) / (np.sqrt(detX * detY))
    else:
        narr = np.arange(n)
        ik_s = np.array(list(combinations(narr, p + 1)))
        detX = np.linalg.det(X[:, ik_s].swapaxes(0, 1))
        detY = np.linalg.det(Y[:, ik_s].swapaxes(0, 1))
        nom = detX @ detY
        denom = np.sqrt(np.sum(detX**2) * np.sum(detY**2))
        rho = nom / denom

    return rho


def wasserstein_distance_so3(
    samples_1: np.ndarray, samples_2: np.ndarray
) -> np.ndarray:
    try:
        import ot
    except ImportError:
        raise ImportError(
            "The ot library is required for computing the Wasserstein distance."
        )

    n1, n2 = len(samples_1), len(samples_2)
    p1, p2 = np.ones(n1) / n1, np.ones(n2) / n2

    distance_matrix = batch_geodesic_distance_matrix(samples_1, samples_2)

    # Compute Wasserstein distance using the POT library
    wasserstein_distance = ot.emd2(
        p1, p2, distance_matrix, numItermax=1e7, numThreads=1
    )
    return wasserstein_distance


def wasserstein_distance_nd_so3(
    samples_1: np.ndarray, samples_2: np.ndarray
) -> np.ndarray:
    try:
        import ot
    except ImportError:
        raise ImportError(
            "The ot library is required for computing the Wasserstein distance."
        )

    n1, n2 = len(samples_1), len(samples_2)
    p1, p2 = np.ones(n1) / n1, np.ones(n2) / n2

    distance_matrices_l2 = np.zeros((n1, n2))
    for i in range(samples_1.shape[1]):
        distance_matrix = batch_geodesic_distance_matrix(
            samples_1[:, i], samples_2[:, i]
        )
        distance_matrices_l2 += distance_matrix

    wasserstein_distance = ot.emd2(
        p1, p2, distance_matrices_l2, numItermax=1e7, numThreads=1
    )
    return wasserstein_distance


def compute_concentration_of_sample_set_SO3(rot_mats: np.ndarray):
    rot_obj = Rotation.from_matrix(rot_mats)
    mean_direction = rot_obj.mean()
    mean_direction = mean_direction.as_matrix()[None]
    distances = batch_geodesic_distance_matrix(rot_obj.as_matrix(), mean_direction)
    concentration = distances.std()
    return concentration
