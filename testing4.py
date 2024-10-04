import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import least_squares
from utils import (
    transform_from_scanner_frame,
    transform_to_scanner_frame,
    scan_line_residuals,
    scan_point_residuals,
)
import argparse
from pathlib import Path


def f(x: float) -> float:
    return np.sin(x)


def f_range(
    x_min: float,
    x_max: float,
    num_points: int,
) -> np.ndarray:
    x = np.linspace(x_min, x_max, num_points)
    y = f(x)
    return x, y


def hits_f(x0: float, y0: float, theta: float) -> np.ndarray:
    """
    Find the intersection points of a line with the function f(x) = sin(x).

    Args:
        x0: x-coordinate of the laser emission point
        y0: y-coordinate of the laser emission point
        theta: angle of the laser path in radians

    Returns:
        x_intersections: x-coordinates of the intersection points
        y_intersections: y-coordinates of the intersection points
    """

    def laser_path(x):
        return y0 + np.tan(theta) * (x - x0)

    def intersection(x):
        return f(x) - laser_path(x)

    # Check the direction of the laser based on the angle theta
    search_direction = np.sign(np.cos(theta))
    step_size = 0.1
    step = search_direction * step_size

    for i in range(100):
        x_a = x0 + i * step
        x_b = x0 + (i + 1) * step
        try:
            root = scipy.optimize.root_scalar(
                intersection, bracket=[x_a, x_b], method="brentq"
            )
        except ValueError:
            continue
        if root.converged:
            return root.root, f(root.root)

    # If the laser does not intersect the function, return the point after 100 steps

    return x0 + 100 * step, laser_path(x0 + 100 * step)


def create_scan(
    x_scanner: float,
    y_scanner: float,
    theta_scanner: float,
    angle_range: float,
    num_points: int,
) -> np.ndarray:
    """
    Create a simulated scan of the function f(x) = sin(x).

    Args:
        x_scanner: x-coordinate of the scanner
        y_scanner: y-coordinate of the scanner
        theta_scanner: angle of the scanner in radians
        angle_range: total angle range of the scan in radians
        num_points: number of points in the scan

    Returns:
        scan: Nx2 array of scan points
    """
    scan = []
    for angle in np.linspace(
        theta_scanner - angle_range / 2, theta_scanner + angle_range / 2, num_points
    ):
        x, y = hits_f(x_scanner, y_scanner, angle)
        scan.append((x, y))

    return np.array(scan)


def create_scans(
    x_scanner_1: float,
    y_scanner_1: float,
    theta_scanner_1: float,
    x_scanner_2: float,
    y_scanner_2: float,
    theta_scanner_2: float,
    num_points: int = 100,
    angle_range: float = np.pi / 4,
) -> tuple:
    scan_1 = create_scan(
        x_scanner_1, y_scanner_1, theta_scanner_1, angle_range, num_points
    )
    scan_2 = create_scan(
        x_scanner_2, y_scanner_2, theta_scanner_2, angle_range, num_points
    )

    return scan_1, scan_2


def objective_function(
    params,
    initial_frame: np.ndarray,
    points: list[np.ndarray],
    dx: float,
    dy: float,
    m_shape: tuple[int],
    x_offset: float,
    y_offset: float,
    number_of_points_scan_line: int,
    both_directions: bool,
    weight_points: float,
    weight_lines: float,
):
    """Objective function to minimize for the optimization problem.

    Args:
        initial_frame: Frame of the first scan
        params: array of parameters to optimize
        points: list of scan points
        dx: grid spacing along the x-axis
        dy: grid spacing along the y-axis
        m_shape: shape of the map grid
        x_offset: offset of the map along the x-axis
        y_offset: offset of the map along the y-axis
        number_of_points_scan_line: number of points to consider in each scan line
        both_directions: whether to consider both directions of the scan line
        weight_points: weight for the point residuals
        weight_lines: weight for the line residuals

    Returns:
        Array of residuals for the optimization problem
    """

    residuals_points = weight_points * scan_point_residuals(
        params, initial_frame, points, dx, dy, m_shape, x_offset, y_offset
    )
    residuals_lines = weight_lines * scan_line_residuals(
        params,
        initial_frame,
        points,
        dx,
        dy,
        m_shape,
        x_offset,
        y_offset,
        number_of_points_scan_line,
        both_directions,
    )
    residuals = np.concatenate([residuals_points, residuals_lines])
    return residuals


def plot_scans_in_global_frame(
    scan_1_local: np.ndarray,
    scan_2_local: np.ndarray,
    initial_frame: np.ndarray,
    optimized_theta_tx_ty: np.ndarray,
    file_name: Path | None = None,
):
    """Transform the scan points to global frame and plot them.

    Args:
        scan_1_local: Scan points for scanner 1 in the local frame
        scan_2_local: Scan points for scanner 2 in the local frame
        initial_frame: Frame of the first scan
        optimized_theta_tx_ty: Optimized parameters for scanner 2
        file_name: Path to save the plot. If None, the plot is displayed.
    """
    # Now transform back to the global frame and verify if the points match
    scan_1_global = transform_from_scanner_frame(
        scan_1_local,
        initial_frame[0],
        initial_frame[1],
        initial_frame[2],
    )
    scan_2_global = transform_from_scanner_frame(
        scan_2_local,
        optimized_theta_tx_ty[0][0],
        optimized_theta_tx_ty[0][1],
        optimized_theta_tx_ty[0][2],
    )

    # Plot the transformed back scan points
    for x, y in scan_1_global:
        plt.scatter(x, y, color="orange", label="Transformed back 1")

    for x, y in scan_2_global:
        plt.scatter(x, y, color="cyan", label="Transformed back 2")

    if file_name:
        plt.savefig(file_name)
        return

    plt.show()


def plot_map(
    m: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    scan_1_local: np.ndarray,
    scan_2_local: np.ndarray,
    initial_frame: np.ndarray,
    optimized_theta_tx_ty: np.ndarray,
    file_name: Path | None,
) -> None:
    """
    Plot the map and the transformed scan points.

    Args:
        m: 2D map grid
        x_min: Minimum value on the x-axis
        x_max: Maximum value on the x-axis
        y_min: Minimum value on the y-axis
        y_max: Maximum value on the y-axis
        scan_1_local: Scan points for scanner 1 in the local frame
        scan_2_local: Scan points for scanner 2 in the local frame
        initial_frame: Frame of the first scan
        optimized_theta_tx_ty: Optimized parameters for scanner 2
        file_name: Path to save the plot. If None, the plot is displayed.
    """
    m_clipped = np.clip(m, -3, 3)
    plt.imshow(
        m_clipped.T,
        origin="lower",
        extent=[
            x_min,
            x_max,
            y_min,
            y_max,
        ],
    )
    plt.title("Map")
    plt.colorbar()
    x = np.linspace(x_min, x_max, 1000)
    y = f(x)
    plt.plot(x, y, color="red", label="f(x) = sin(x)")
    plt.legend()

    # Now transform back to the global frame and verify if the points match
    scan_1_global = transform_from_scanner_frame(
        scan_1_local,
        initial_frame[0],
        initial_frame[1],
        initial_frame[2],
    )
    scan_2_global = transform_from_scanner_frame(
        scan_2_local,
        optimized_theta_tx_ty[0][0],
        optimized_theta_tx_ty[0][1],
        optimized_theta_tx_ty[0][2],
    )

    # Plot the transformed back scan points
    for x, y in scan_1_global:
        plt.scatter(x, y, color="orange", label="Transformed back 1")

    for x, y in scan_2_global:
        plt.scatter(x, y, color="cyan", label="Transformed back 2")

    if file_name:
        plt.savefig(file_name)
        return

    plt.show()


def init_map(
    x_min: float, x_max: float, y_min: float, y_max: float, num_x: int, num_y: int
) -> np.ndarray:
    """
    Initialize a map with specified grid extents and resolution.

    Args:
        x_min: Minimum value on the x-axis
        x_max: Maximum value on the x-axis
        y_min: Minimum value on the y-axis
        y_max: Maximum value on the y-axis
        num_x: Number of grid points along the x-axis
        num_y: Number of grid points along the y-axis

    Returns:
        m: 2D map grid initialized to zeros.
    """
    m = np.zeros((num_x, num_y))

    # Compute the grid spacing (dx and dy)
    dx = (x_max - x_min) / (num_x - 1)
    dy = (y_max - y_min) / (num_y - 1)

    return m, dx, dy


def print_guess(optimized_theta_tx_ty, true_theta_tx_ty):
    print("Optimized parameters:")
    print(optimized_theta_tx_ty)
    print("True parameters:")
    print(true_theta_tx_ty)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Simulate and optimize laser scans.")
    parser.add_argument(
        "--max_nfev",
        type=int,
        default=15,
        help="Maximum number of function evaluations.",
    )
    parser.add_argument(
        "--num_points", type=int, default=50, help="Number of points in the scan."
    )
    parser.add_argument(
        "--noise_level_degrees", type=float, default=5, help="Noise level in degrees."
    )
    parser.add_argument(
        "--noise_level_translation",
        type=float,
        default=0.15,
        help="Noise level in translation.",
    )
    parser.add_argument(
        "--number_of_points_scan_line",
        type=int,
        default=10,
        help="Number of points to consider in each scan line.",
    )
    parser.add_argument(
        "--both_directions",
        type=bool,
        default=True,
        help="Whether to consider both directions of the scan line.",
    )
    parser.add_argument(
        "--weight_points", type=float, default=1, help="Weight for the point residuals."
    )
    parser.add_argument(
        "--weight_lines", type=float, default=1, help="Weight for the line residuals."
    )
    parser.add_argument(
        "--map_size_x",
        type=int,
        default=50,
        help="Number of grid points along the x-axis.",
    )
    parser.add_argument(
        "--map_size_y",
        type=int,
        default=50,
        help="Number of grid points along the y-axis.",
    )
    return parser.parse_args()


def plot_scan_lines(
    x_scanner_1: float,
    y_scanner_1: float,
    scan_1: np.ndarray,
    x_scanner_2: float,
    y_scanner_2: float,
    scan_2: np.ndarray,
    file_name: Path | None,
):
    """Plot the scanner locations and the scan points with the scan lines.

    Args:
        x_scanner_1: x-coordinate of scanner 1
        y_scanner_1: y-coordinate of scanner 1
        scan_1: Scan points for scanner 1
        x_scanner_2: x-coordinate of scanner 2
        y_scanner_2: y-coordinate of scanner 2
        scan_2: Scan points for scanner 2
        file_name: Path to save the plot. If None, the plot is displayed.
    """
    # Plot the locations of the scanners
    plt.scatter(x_scanner_1, y_scanner_1, color="green", marker="x", label="Scanner 1")
    plt.scatter(x_scanner_2, y_scanner_2, color="purple", marker="x", label="Scanner 2")

    # Plot the scan points
    for x, y in scan_1:
        plt.scatter(x, y, color="red", label="Hits")

    for x, y in scan_2:
        plt.scatter(x, y, color="blue", label="Hits")

    # Plot the scan lines
    for x, y in scan_1:
        plt.plot(
            [x_scanner_1, x],
            [y_scanner_1, y],
            color="red",
            linestyle="--",
            linewidth=0.5,
        )

    for x, y in scan_2:
        plt.plot(
            [x_scanner_2, x],
            [y_scanner_2, y],
            color="blue",
            linestyle="--",
            linewidth=0.5,
        )

    if file_name:
        plt.savefig(file_name)
        return

    plt.show()


if __name__ == "__main__":
    args = parse_arguments()
    max_nfev = args.max_nfev
    num_points = args.num_points

    # Noise parameters
    noise_level_degrees = args.noise_level_degrees
    noise_level_translation = args.noise_level_translation

    number_of_points_scan_line = args.number_of_points_scan_line
    both_directions = args.both_directions

    weight_points = args.weight_points
    weight_lines = args.weight_lines

    # Map parameters
    map_size_x = args.map_size_x
    map_size_y = args.map_size_y

    #
    x_min = -1
    x_max = 2 * np.pi + 1
    y_min = -6
    y_max = 2

    x_s, y_s = f_range(0, 2 * np.pi, 100)
    plt.plot(x_s, y_s, label="f(x)")

    x_scanner_1 = 3.5
    y_scanner_1 = -5
    theta_scanner_1 = 9 * np.pi / 16

    x_scanner_2 = 4
    y_scanner_2 = -5
    theta_scanner_2 = 8 * np.pi / 16

    scan_1, scan_2 = create_scans(
        x_scanner_1,
        y_scanner_1,
        theta_scanner_1,
        x_scanner_2,
        y_scanner_2,
        theta_scanner_2,
        num_points,
    )

    plot_scan_lines(
        x_scanner_1, y_scanner_1, scan_1, x_scanner_2, y_scanner_2, scan_2, None
    )

    # Transform the scans into the frame of the respective scanner
    transformed_scan_1 = transform_to_scanner_frame(
        scan_1, theta_scanner_1, x_scanner_1, y_scanner_1
    )
    transformed_scan_2 = transform_to_scanner_frame(
        scan_2, theta_scanner_2, x_scanner_2, y_scanner_2
    )

    ######## Here we begin with the optimization ########

    # Add noise to the initial guess of all the parameters
    noise_level_rad = np.deg2rad(noise_level_degrees)

    theta_scanner_2_initial_guess = theta_scanner_2 + np.random.normal(
        -noise_level_rad, noise_level_rad
    )
    x_scanner_2_initial_guess = x_scanner_2 + np.random.normal(
        -noise_level_translation, noise_level_translation
    )
    y_scanner_2_initial_guess = y_scanner_2 + np.random.normal(
        -noise_level_translation, noise_level_translation
    )

    initial_frame = np.array(
        [
            theta_scanner_1,
            x_scanner_1,
            y_scanner_1,
        ]
    )

    initial_theta_tx_ty = [
        theta_scanner_2_initial_guess,
        x_scanner_2_initial_guess,
        y_scanner_2_initial_guess,
    ]

    print_guess(
        initial_theta_tx_ty,
        [
            theta_scanner_2,
            x_scanner_2,
            y_scanner_2,
        ],
    )
    # Initial guess for the map
    m_initial, dx, dy = init_map(x_min, x_max, y_min, y_max, map_size_x, map_size_y)
    m_shape = m_initial.shape

    initial_params = np.concatenate([initial_theta_tx_ty, m_initial.ravel()])

    plot_scans_in_global_frame(
        transformed_scan_1,
        transformed_scan_2,
        initial_frame,
        np.array(initial_theta_tx_ty).reshape((1, 3)),
    )
    plot_map(
        m_initial,
        x_min,
        x_max,
        y_min,
        y_max,
        transformed_scan_1,
        transformed_scan_2,
        initial_frame,
        np.array(initial_theta_tx_ty).reshape((1, 3)),
    )

    result = least_squares(
        objective_function,
        initial_params,
        args=(
            initial_frame,
            [transformed_scan_1, transformed_scan_2],
            dx,
            dy,
            m_shape,
            x_min,
            y_min,
            number_of_points_scan_line,
            both_directions,
            weight_points,
            weight_lines,
        ),
        verbose=2,
        max_nfev=max_nfev,
        # callback=callback,
    )

    # Extract the optimized parameters
    optimized_theta_tx_ty = result.x[:3].reshape((1, 3))
    optimized_map = result.x[3:].reshape(m_shape)

    print_guess(
        optimized_theta_tx_ty,
        [
            theta_scanner_2,
            x_scanner_2,
            y_scanner_2,
        ],
    )
    plot_scans_in_global_frame(
        transformed_scan_1, transformed_scan_2, initial_frame, optimized_theta_tx_ty
    )

    plot_map(
        optimized_map,
        x_min,
        x_max,
        y_min,
        y_max,
        transformed_scan_1,
        transformed_scan_2,
        initial_frame,
        optimized_theta_tx_ty,
    )
