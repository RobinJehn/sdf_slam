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
    points,
    dx,
    dy,
    M_shape,
    x_offset: float,
    y_offset: float,
):
    residuals_points = scan_point_residuals(
        params, points, dx, dy, M_shape, x_offset, y_offset
    )
    residuals_lines = scan_line_residuals(
        params, points, dx, dy, M_shape, x_offset, y_offset
    )
    residuals = np.concatenate([residuals_points, residuals_lines])
    return residuals


def plot_transformed_scans(
    transformed_scan_1, transformed_scan_2, optimized_theta_tx_ty
):
    # Now transform back to the global frame and verify if the points match
    transformed_back_scan_1 = transform_from_scanner_frame(
        transformed_scan_1,
        optimized_theta_tx_ty[0][0],
        optimized_theta_tx_ty[0][1],
        optimized_theta_tx_ty[0][2],
    )
    transformed_back_scan_2 = transform_from_scanner_frame(
        transformed_scan_2,
        optimized_theta_tx_ty[1][0],
        optimized_theta_tx_ty[1][1],
        optimized_theta_tx_ty[1][2],
    )

    # Plot the transformed back scan points
    for x, y in transformed_back_scan_1:
        plt.scatter(x, y, color="orange", label="Transformed back 1")

    for x, y in transformed_back_scan_2:
        plt.scatter(x, y, color="cyan", label="Transformed back 2")

    plt.show()


def plot_map(
    m: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    transformed_scan_1,
    transformed_scan_2,
    optimized_theta_tx_ty,
):
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
    transformed_back_scan_1 = transform_from_scanner_frame(
        transformed_scan_1,
        optimized_theta_tx_ty[0][0],
        optimized_theta_tx_ty[0][1],
        optimized_theta_tx_ty[0][2],
    )
    transformed_back_scan_2 = transform_from_scanner_frame(
        transformed_scan_2,
        optimized_theta_tx_ty[1][0],
        optimized_theta_tx_ty[1][1],
        optimized_theta_tx_ty[1][2],
    )

    # Plot the transformed back scan points
    for x, y in transformed_back_scan_1:
        plt.scatter(x, y, color="orange", label="Transformed back 1")

    for x, y in transformed_back_scan_2:
        plt.scatter(x, y, color="cyan", label="Transformed back 2")

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


if __name__ == "__main__":
    max_nfev = 30
    num_points = 50

    # Noise parameters
    noise_level_degrees = 5
    noise_level_translation = 0.15

    # Map parameters
    map_size_x = 50
    map_size_y = 50
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

    plt.show()

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

    theta_scanner_1_initial_guess = theta_scanner_1 + np.random.normal(
        -noise_level_rad, noise_level_rad
    )
    x_scanner_1_initial_guess = x_scanner_1 + np.random.normal(
        -noise_level_translation, noise_level_translation
    )
    y_scanner_1_initial_guess = y_scanner_1 + np.random.normal(
        -noise_level_translation, noise_level_translation
    )

    theta_scanner_2_initial_guess = theta_scanner_2 + np.random.normal(
        -noise_level_rad, noise_level_rad
    )
    x_scanner_2_initial_guess = x_scanner_2 + np.random.normal(
        -noise_level_translation, noise_level_translation
    )
    y_scanner_2_initial_guess = y_scanner_2 + np.random.normal(
        -noise_level_translation, noise_level_translation
    )

    initial_theta_tx_ty = [
        theta_scanner_1_initial_guess,
        x_scanner_1_initial_guess,
        y_scanner_1_initial_guess,
        theta_scanner_2_initial_guess,
        x_scanner_2_initial_guess,
        y_scanner_2_initial_guess,
    ]

    print_guess(
        initial_theta_tx_ty,
        [
            theta_scanner_1,
            x_scanner_1,
            y_scanner_1,
            theta_scanner_2,
            x_scanner_2,
            y_scanner_2,
        ],
    )
    # Initial guess for the map
    m_initial, dx, dy = init_map(x_min, x_max, y_min, y_max, map_size_x, map_size_y)
    m_shape = m_initial.shape

    initial_params = np.concatenate([initial_theta_tx_ty, m_initial.ravel()])

    plot_transformed_scans(
        transformed_scan_1,
        transformed_scan_2,
        np.array(initial_theta_tx_ty).reshape((2, 3)),
    )
    plot_map(
        m_initial,
        x_min,
        x_max,
        y_min,
        y_max,
        transformed_scan_1,
        transformed_scan_2,
        np.array(initial_theta_tx_ty).reshape((2, 3)),
    )

    result = least_squares(
        objective_function,
        initial_params,
        args=(
            [transformed_scan_1, transformed_scan_2],
            dx,
            dy,
            m_shape,
            x_min,
            y_min,
        ),
        verbose=2,
        max_nfev=max_nfev,
        # callback=callback,
    )

    # Extract the optimized parameters
    optimized_theta_tx_ty = result.x[:6].reshape((2, 3))
    optimized_map = result.x[6:].reshape(m_shape)

    print_guess(
        optimized_theta_tx_ty,
        [
            theta_scanner_1,
            x_scanner_1,
            y_scanner_1,
            theta_scanner_2,
            x_scanner_2,
            y_scanner_2,
        ],
    )
    plot_transformed_scans(
        transformed_scan_1, transformed_scan_2, optimized_theta_tx_ty
    )

    plot_map(
        optimized_map,
        x_min,
        x_max,
        y_min,
        y_max,
        transformed_scan_1,
        transformed_scan_2,
        optimized_theta_tx_ty,
    )
