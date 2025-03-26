import yaml
import subprocess
import itertools
import copy
import os
from utils import score_gt, compute_ground_truth
import numpy as np


def modify_yaml(yaml_file, change_dict):
    # Load the YAML file
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    # Update the YAML data with change_dict (merging dictionaries)
    for key, value in change_dict.items():
        if isinstance(value, dict):
            data[key] = {**data.get(key, {}), **value}
        else:
            data[key] = value
    # Write the modified YAML back to file
    with open(yaml_file, "w") as f:
        yaml.safe_dump(data, f)


def run_executable(executable_path):
    # Build and execute the command
    cmd = [executable_path]
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Standard Output:\n", result.stdout)
    print("Standard Error:\n", result.stderr)
    return result.returncode, result.stdout, result.stderr


if __name__ == "__main__":
    yaml_file = "../config/sdf.yml"
    executable_path = "../build/sdf_slam"

    # Define grid search parameters (each parameter is a tuple: (min, max, step))
    grid_search = {
        "objective_args": {
            "scan_point_factor": (1, 10, 3),
            "scan_line_factor": (1, 10, 3),
            "smoothness_factor": (1, 10, 3),
            "odometry_factor": (1, 10, 3),
        },
    }

    # Save the original YAML content to restore it later
    with open(yaml_file, "r") as f:
        original_yaml = f.read()

    # Generate parameter values for each section
    section_param_values = {}
    for section, params in grid_search.items():
        param_values = {}
        for param, (start, stop, step) in params.items():
            # Generate values from start to stop (inclusive)
            values = list(range(start, stop + 1, step))
            param_values[param] = values
        section_param_values[section] = param_values

    # Generate a list of dictionaries for all parameter combinations for each section
    section_combinations = {}
    for section, params in section_param_values.items():
        keys = list(params.keys())
        combos = list(itertools.product(*(params[key] for key in keys)))
        section_combinations[section] = [dict(zip(keys, combo)) for combo in combos]

    # Get the overall combinations across sections (Cartesian product)
    overall_combos = list(
        itertools.product(*(section_combinations[sec] for sec in section_combinations))
    )
    print(f"Total number of configurations to test: {len(overall_combos)}")

    tree_gt, gt_transforms = compute_ground_truth(
        "real_data/liang/Scan_Simu76.mat", "real_data/liang/Odom_Simu76.mat"
    )
    experiments_folder = "."
    results = []
    # Iterate over each overall configuration
    for combo in overall_combos:
        # combo is a tuple of dictionaries, one per section
        change_dict = {}
        for sec, values in zip(section_combinations.keys(), combo):
            change_dict[sec] = values
        print("Testing configuration:", change_dict)
        modify_yaml(yaml_file, change_dict)
        retcode, stdout, stderr = run_executable(executable_path)
        print("Return Code:", retcode)

        # Find the experiment folder with the highest number
        highest_number = -1
        highest_folder = None
        for folder in os.listdir(experiments_folder):
            if folder.startswith("experiment_"):
                try:
                    number = int(folder.split("_")[1])
                    if number > highest_number:
                        highest_number = number
                        highest_folder = folder
                except ValueError:
                    continue

        map_error, trans_errors, rot_errors = score_gt(
            highest_folder, tree_gt, gt_transforms
        )
        score = (
            np.mean(map_error) + 15 * np.mean(trans_errors) + 300 * np.mean(rot_errors)
        )
        print(
            f"Map Error: {np.mean(map_error)}, Translation Errors: {np.mean(trans_errors)}, Rotation Errors: {np.mean(rot_errors)}, Score: {score}"
        )

        results.append(
            {
                "config": copy.deepcopy(change_dict),
                "return_code": retcode,
                "stdout": stdout,
                "stderr": stderr,
                "score": score,
                "map_error": np.mean(map_error),
                "rot_error": np.mean(rot_errors),
                "trans_error": np.mean(trans_errors),
            }
        )

    # Optionally, save the grid search results to a file
    with open("grid_search_results.yaml", "w") as f:
        yaml.safe_dump(results, f)

    # Restore the original YAML file
    with open(yaml_file, "w") as f:
        f.write(original_yaml)
