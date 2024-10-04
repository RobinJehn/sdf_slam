import numpy as np


### Map helpers ###
def bilinear_interpolation(
    point: tuple[float, float],
    M: np.ndarray,
    dx: float,
    dy: float,
    x_offset: float,
    y_offset: float,
) -> float:
    """
    Perform bilinear interpolation for a given point on a 2D grid.

    Parameters:
    point: The (x, y) coordinates of the point to interpolate.
    M: The 2D grid of values.
    dx: The grid spacing in the x-direction.
    dy: The grid spacing in the y-direction.
    x_offset: The x-coordinate of the grid origin.
    y_offset: The y-coordinate of the grid origin.

    Returns:
        The interpolated value at the given point.
    """

    x, y = point
    grid_x, grid_y = M.shape

    # Convert point coordinates to indices
    x_idx = (x - x_offset) / dx
    y_idx = (y - y_offset) / dy

    # Find the integer coordinates surrounding the point
    w = int(np.floor(x_idx))
    h = int(np.floor(y_idx))

    # Compute interpolation weights
    a = x_idx - w
    b = y_idx - h

    # Ensure indices are within bounds
    w = np.clip(w, 0, grid_x - 2)
    h = np.clip(h, 0, grid_y - 2)
    a = np.clip(a, 0, 1)
    b = np.clip(b, 0, 1)

    # Extract the 4 neighboring grid points
    f00 = M[w, h]
    f10 = M[w + 1, h]
    f01 = M[w, h + 1]
    f11 = M[w + 1, h + 1]

    # Bilinear interpolation formula
    interpolated_value = (
        (1 - a) * (1 - b) * f00 + a * (1 - b) * f10 + (1 - a) * b * f01 + a * b * f11
    )
    return interpolated_value


### Transformation helpers ###
def create_transformation_matrix(
    rotation_angle: float, tx: float, ty: float
) -> np.ndarray:
    """
    Create a 3x3 homogeneous transformation matrix for 2D rotation and translation.

    Parameters:
    - rotation_angle: angle of rotation in radians
    - tx: translation along the x-axis
    - ty: translation along the y-axis

    Returns:
    - 3x3 transformation matrix
    """
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)

    # Homogeneous transformation matrix
    T = np.array([[cos_theta, -sin_theta, tx], [sin_theta, cos_theta, ty], [0, 0, 1]])

    return T


def apply_transformation(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply the transformation matrix T to an array of 2D points.

    Parameters:
    - T: 3x3 transformation matrix
    - points: Nx2 array of 2D points

    Returns:
    - Nx2 array of transformed 2D points
    """
    # Convert points to homogeneous coordinates
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])

    # Apply the transformation
    transformed_points = np.dot(T, homogeneous_points.T).T

    # Return the transformed points in 2D (ignore the homogeneous coordinate)
    return transformed_points[:, :2]


def transform_to_scanner_frame(
    points: np.ndarray, theta_scanner: float, x_scanner: float, y_scanner: float
) -> np.ndarray:
    """
    Transform a set of 2D points into the scanner's frame.

    Args:
        points: Nx2 array of 2D points
        theta_scanner: angle of the scanner in radians
        x_scanner: x-coordinate of the scanner
        y_scanner: y-coordinate of the scanner

    Returns:
        transformed_points: Nx2 array of transformed 2D points into the scanner's frame
    """
    # Create the transformation matrix and then inverse it
    T = create_transformation_matrix(theta_scanner, x_scanner, y_scanner)
    inv_T = np.linalg.inv(T)
    transformed_points = apply_transformation(inv_T, points)
    return transformed_points


def transform_from_scanner_frame(
    points: np.ndarray, theta_scanner: float, x_scanner: float, y_scanner: float
) -> np.ndarray:
    """
    Transform a set of 2D points from the scanner's frame back to the global frame.

    Args:
        points: Nx2 array of 2D points in scanner's frame
        theta_scanner: angle of the scanner in radians
        x_scanner: x-coordinate of the scanner
        y_scanner: y-coordinate of the scanner

    Returns:
        transformed_points: Nx2 array of transformed 2D points in the global frame
    """
    # Create the transformation matrix to rotate by theta and translate by x_scanner, y_scanner
    T = create_transformation_matrix(theta_scanner, x_scanner, y_scanner)
    transformed_points = apply_transformation(T, points)
    return transformed_points


def scan_point_residuals(
    params: np.ndarray,
    points: list[np.ndarray],
    dx: float,
    dy: float,
    M_shape: tuple[int],
    x_offset: float,
    y_offset: float,
) -> np.ndarray:
    """Compute the residuals for each point in the scan data.

    Args:
        params: array of transformation parameters and map values
        points: list of 2D scan data points
        dx: grid spacing in the x-direction
        dy: grid spacing in the y-direction
        M_shape: shape of the map grid
        x_offset: x-coordinate of the grid origin
        y_offset: y-coordinate of the grid origin

    Returns:
        Array of residuals for each point in the scan data
    """
    # Extract the transformation parameters (theta, tx, ty for each scan)
    num_scans = len(points)
    scan_params = params[: num_scans * 3].reshape((num_scans, 3))
    # Extract map values from parameters (flattened map)
    M = params[num_scans * 3 :].reshape(M_shape)

    residuals = []

    for i, scan in enumerate(points):
        theta, tx, ty = scan_params[i]

        scan_global = transform_from_scanner_frame(scan, theta, tx, ty)

        # Compute residuals using bilinear interpolation on the map
        for point in scan_global:
            interpolated_value = bilinear_interpolation(
                point, M, dx, dy, x_offset, y_offset
            )
            residual = -interpolated_value  # Observed value is zero
            residuals.append(residual)

    residuals = np.array(residuals)

    return residuals


def scan_line_residuals(
    params: np.ndarray,
    points: list[np.ndarray],
    dx: float,
    dy: float,
    M_shape: tuple[int],
    x_offset: float,
    y_offset: float,
):
    """Compute the residuals for each point in the scan data.

    Args:
        params: array of transformation parameters and map values
        points: list of 2D scan data points
        dx: grid spacing in the x-direction
        dy: grid spacing in the y-direction
        M_shape: shape of the map grid
        x_offset: x-coordinate of the grid origin
        y_offset: y-coordinate of the grid origin

    Returns:
        Array of residuals for each point in the scan data
    """
    # Extract the transformation parameters (theta, tx, ty for each scan)
    num_scans = len(points)
    scan_params = params[: num_scans * 3].reshape((num_scans, 3))
    # Extract map values from parameters (flattened map)
    M = params[num_scans * 3 :].reshape(M_shape)

    residuals = []

    step_size = 0.2
    number_of_points = 30
    for i, scan in enumerate(points):
        scan_line_points = []
        distances = []
        theta, tx, ty = scan_params[i]

        for point in scan:
            distance_to_origin = np.linalg.norm(point)
            vector_to_origin = -point / distance_to_origin * step_size
            for j in range(
                1, min(number_of_points + 1, int(distance_to_origin / step_size))
            ):
                scan_line_points.append(point + vector_to_origin * j)
                distances.append(step_size * j)

        scan_line_points = np.array(scan_line_points)
        scan_global = transform_from_scanner_frame(scan_line_points, theta, tx, ty)

        # Compute residuals using bilinear interpolation on the map
        for i, point_global in enumerate(scan_global):
            interpolated_value = bilinear_interpolation(
                point_global, M, dx, dy, x_offset, y_offset
            )
            residual = distances[i] - interpolated_value

            # print(point_global, interpolated_value, distances[i], residual)
            residuals.append(residual)

    residuals = np.array(residuals)

    return residuals
