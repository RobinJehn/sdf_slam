import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import os
import yaml


# 2D transformation matrix (homogeneous coordinates)
def transformation_matrix(x: float, y: float, theta: float):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0, 0, 1],
        ]
    )


def compute_ground_truth(scan_filename: str, odom_filename: str):
    # Load scan data and odometry from your .mat files
    data_scans = scipy.io.loadmat(scan_filename)
    scans = []
    for scan in data_scans["Scan"][0]:
        points = scan[0][0][0]
        log_odds = scan[0][0][1]
        # Only use points with high log-odds (adjust threshold as needed)
        scan_points = [point for point, lo in zip(points, log_odds) if lo > 0.84]
        scans.append(scan_points)

    data_odom = scipy.io.loadmat(odom_filename)
    odom = data_odom["Odom"]
    # Prepend the initial pose [0, 0, 0]
    odom = np.vstack(([0, 0, 0], odom))

    # Transform each scan into the global frame using the odometry data
    global_scans = []
    gt_transforms = []
    cur_T = transformation_matrix(0, 0, 0)
    for i, scan in enumerate(scans):
        # Update transformation using the current odometry (assumes odom[i] = [dx, dy, dtheta])
        cur_T = cur_T @ transformation_matrix(odom[i][0], odom[i][1], odom[i][2])
        transformed_scan = []
        for point in scan:
            ph = np.array([point[0], point[1], 1])
            tp = cur_T @ ph
            transformed_scan.append([tp[0], tp[1]])
        global_scans.append(np.array(transformed_scan))

        # Transforms
        if i > 0:
            tx, ty = cur_T[0, 2], cur_T[1, 2]
            theta = np.arctan2(cur_T[1, 0], cur_T[0, 0])
            gt_transforms.append([tx, ty, theta])

    gt_transforms = np.array(gt_transforms)

    # Combine all scans into one global set
    all_points = np.concatenate(global_scans, axis=0)
    tree_gt = KDTree(all_points)
    return tree_gt, gt_transforms


def get_map_params(folder: str):
    # assumes the yaml file is called params.yml
    yaml_file = os.path.join(folder, "params.yml")
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    return data["num_points"], data["min_coords"], data["max_coords"]


def create_grid(num_points, min_coords, max_coords):
    nx, ny = num_points
    x_coords = np.linspace(min_coords[0], max_coords[0], nx)
    y_coords = np.linspace(min_coords[1], max_coords[1], ny)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing="ij")
    grid_points = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    return grid_points


def load_last_parameters(folder: str):
    files = os.listdir(folder)
    param_files = [f for f in files if f.startswith("params_") and f.endswith(".txt")]
    max_num = -1
    max_file = None
    for f in param_files:
        num = int(f.split("_")[1].split(".")[0])
        if num > max_num:
            max_num = num
            max_file = f
    if max_file:
        est_flat = np.loadtxt(os.path.join(folder, max_file))
    else:
        raise FileNotFoundError("No parameter files found in the specified folder.")
    return est_flat


def load_parameters(file: str):
    return np.loadtxt(file)


def plot_map_error_dist(errors: list[float]):
    plt.hist(errors, bins=50, alpha=0.7)
    plt.title("Map Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.show()


def plot_path_diff(gt_trans, est_trans):
    gt_positions = gt_trans[:, :2]
    est_positions = est_trans[:, :2]
    # Plot the ground truth trajectory vs. estimated transformations
    plt.figure(figsize=(8, 6))
    plt.plot(gt_positions[:, 0], gt_positions[:, 1], "bo-", label="Ground Truth")
    plt.plot(est_positions[:, 0], est_positions[:, 1], "ro-", label="Estimated")
    plt.title("Trajectory Comparison")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


def extract_estimated_transfoms(est_flat, num_map_points):
    # Evaluate the transformation (frame) estimates
    num_transform_params = 3
    est_trans_flat = est_flat[num_map_points:]
    num_est_trans = len(est_trans_flat) // num_transform_params
    est_trans = est_trans_flat.reshape((num_est_trans, num_transform_params))
    return est_trans


def compute_transform_errors(est_trans, gt_trans):
    # Compute translation error (Euclidean distance) and rotation error (absolute angle difference)
    trans_errors = np.linalg.norm(est_trans[:, :2] - gt_trans[:, :2], axis=1)
    rot_errors = np.abs(est_trans[:, 2] - gt_trans[:, 2])
    # Wrap rotation error to [0, pi]
    rot_errors = (rot_errors + np.pi) % (2 * np.pi) - np.pi
    rot_errors = np.abs(rot_errors)

    return trans_errors, rot_errors


def plot_ground_truth_grid(grid_points, true_distances):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        grid_points[:, 0], grid_points[:, 1], c=true_distances, cmap="viridis", s=5
    )
    plt.title("True Distances at Each Grid Point")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Distance")
    plt.show()


def score_gt(exp_folder: str, tree_gt, gt_trans):
    num_points, min_coords, max_coords = get_map_params(exp_folder)
    grid_points = create_grid(num_points, min_coords, max_coords)
    true_distances, _ = tree_gt.query(grid_points)
    est_flat = load_last_parameters(exp_folder)
    num_map_points = num_points[0] * num_points[1]
    est_map = est_flat[:num_map_points]
    map_error = np.abs(est_map - true_distances)
    est_trans = extract_estimated_transfoms(est_flat, num_map_points)
    trans_errors, rot_errors = compute_transform_errors(est_trans, gt_trans)

    return (
        map_error,
        trans_errors,
        rot_errors,
    )


def score(exp_folder: str, scan_file: str, odom_file: str):
    num_points, min_coords, max_coords = get_map_params(exp_folder)
    grid_points = create_grid(num_points, min_coords, max_coords)
    tree_gt, gt_trans = compute_ground_truth(scan_file, odom_file)
    true_distances, _ = tree_gt.query(grid_points)
    est_flat = load_last_parameters(exp_folder)
    num_map_points = num_points[0] * num_points[1]
    est_map = est_flat[:num_map_points]
    map_error = np.abs(est_map - true_distances)
    est_trans = extract_estimated_transfoms(est_flat, num_map_points)
    trans_errors, rot_errors = compute_transform_errors(est_trans, gt_trans)

    return np.mean(map_error) + 15 * np.mean(trans_errors) + 300 * np.mean(rot_errors)
