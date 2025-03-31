from scipy.spatial.transform import Rotation
import numpy as np
import matplotlib.pyplot as plt


def visualize_like_implicit_pdf(
    rotations, canonical_rotation=np.eye(3), fig=None, return_fig=False
):
    # from https://github.com/google-research/google-research/blob/master/implicit_pdf/evaluation.py
    # rotations: matrix
    def _show_single_marker(
        ax, rotation, euler, marker, edgecolors=True, facecolors=False, annotate=None
    ):
        xyz = rotation[:, 0]
        tilt_angle = euler[0]
        longitude = np.arctan2(xyz[0], -xyz[1])
        latitude = np.arcsin(xyz[2])

        color = cmap(0.5 + tilt_angle / 2 / np.pi)
        ax.scatter(
            longitude,
            latitude,
            s=2500 / 200,
            edgecolors=color if edgecolors else "none",
            facecolors=facecolors if facecolors else "none",
            marker=marker,
            linewidth=1,
        )
        if annotate is not None:
            plt.annotate(annotate, (longitude, latitude))

    rotations = rotations @ canonical_rotation
    eulers = Rotation.from_matrix(rotations).as_euler("xyz")

    cmap = plt.cm.hsv
    if fig is None:
        fig = plt.figure(figsize=(8, 4), dpi=100)

    show_color_wheel = True
    if show_color_wheel:
        # Add a color wheel showing the tilt angle to color conversion.
        ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection="polar")
        theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
        radii = np.linspace(0.4, 0.5, 2)
        _, theta_grid = np.meshgrid(radii, theta)
        colormap_val = 0.5 + theta_grid / np.pi / 2.0
        ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
        ax.set_yticklabels([])
        ax.set_xticks(
            [0, np.pi / 2, np.pi, 3 * np.pi / 2],
            labels=[r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2} \pi$", r"$0$"],
        )

        ax.spines["polar"].set_visible(False)
        plt.text(
            0.5,
            0.5,
            "Tilt",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    ax = fig.add_subplot(111, projection="mollweide")

    # for rotation in rotations:
    for i in range(len(rotations)):
        rotation, euler = rotations[i], eulers[i]
        if i < 48:
            _show_single_marker(ax, rotation, euler, "o")
        else:
            _show_single_marker(ax, rotation, euler, "o")
        _show_single_marker(
            ax, rotation, euler, "o", edgecolors=False, facecolors="#ffffff"
        )

    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if return_fig:
        return fig


def visualize_so3_probabilities(
    rotations,
    probabilities,
    rotations_gt=None,
    ax=None,
    fig=None,
    display_threshold_probability=0.0,
    to_image=True,
    show_color_wheel=True,
    canonical_rotation=np.eye(3),
    title=None,
    scatterpoint_scaling=2e1,
):
    """Plot a single distribution on SO(3) using the tilt-colored method.
    Args:
    rotations: [N, 3, 3] tensor of rotation matrices
    probabilities: [N] tensor of probabilities
    rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
    ax: The matplotlib.pyplot.axis object to paint
    fig: The matplotlib.pyplot.figure object to paint
    display_threshold_probability: The probability threshold below which to omit
        the marker
    to_image: If True, return a tensor containing the pixels of the finished
        figure; if False return the figure itself
    show_color_wheel: If True, display the explanatory color wheel which matches
        color on the plot with tilt angle
    canonical_rotation: A [3, 3] rotation matrix representing the 'display
        rotation', to change the view of the distribution.  It rotates the
        canonical axes so that the view of SO(3) on the plot is different, which
        can help obtain a more informative view.
    Returns:
    A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
    """

    def _show_single_marker(ax, rotation, marker, edgecolors=True, facecolors=False):
        eulers = Rotation.from_matrix(rotation).as_euler("xyz")
        xyz = rotation[:, 0]
        tilt_angle = eulers[0]
        longitude = np.arctan2(xyz[0], -xyz[1])
        latitude = np.arcsin(xyz[2])

        color = cmap(0.5 + tilt_angle / 2 / np.pi)
        ax.scatter(
            longitude,
            latitude,
            s=2500,
            edgecolors=color if edgecolors else "none",
            facecolors=facecolors if facecolors else "none",
            marker=marker,
            linewidth=4,
        )

    if ax is None:
        fig = plt.figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111, projection="mollweide")
        plt.title(title)
    if rotations_gt is not None and len(rotations_gt.shape) == 2:
        rotations_gt = rotations_gt[None]

    display_rotations = rotations @ canonical_rotation
    cmap = plt.cm.hsv
    eulers_queries = Rotation.from_matrix(display_rotations).as_euler("xyz")
    xyz = display_rotations[:, :, 0]
    tilt_angles = eulers_queries[:, 0]

    longitudes = np.arctan2(xyz[:, 0], -xyz[:, 1])
    latitudes = np.arcsin(xyz[:, 2])

    which_to_display = probabilities > display_threshold_probability

    if rotations_gt is not None:
        # The visualization is more comprehensible if the GT
        # rotation markers are behind the output with white filling the interior.
        display_rotations_gt = rotations_gt @ canonical_rotation

        for rotation in display_rotations_gt:
            _show_single_marker(ax, rotation, "o")
        # Cover up the centers with white markers
        for rotation in display_rotations_gt:
            _show_single_marker(
                ax, rotation, "o", edgecolors=False, facecolors="#ffffff"
            )

    # Display the distribution
    ax.scatter(
        longitudes[which_to_display],
        latitudes[which_to_display],
        s=scatterpoint_scaling * probabilities[which_to_display],
        c=cmap(0.5 + tilt_angles[which_to_display] / 2.0 / np.pi),
    )

    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if title is not None:
        ax.set_title(title)

    if show_color_wheel:
        # Add a color wheel showing the tilt angle to color conversion.
        ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection="polar")
        theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
        radii = np.linspace(0.4, 0.5, 2)
        _, theta_grid = np.meshgrid(radii, theta)
        colormap_val = 0.5 + theta_grid / np.pi / 2.0
        ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
        ax.set_yticklabels([])
        ax.set_xticks(
            [0, np.pi / 2, np.pi, 3 * np.pi / 2],
            labels=[r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2} \pi$", r"$0$"],
            fontsize=12,
        )
        ax.spines["polar"].set_visible(False)
        plt.text(
            0.5,
            0.5,
            "Tilt",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    if to_image:
        return None

    else:
        print("Returning fig")
        return fig
