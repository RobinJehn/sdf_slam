#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <SuiteSparseQR.hpp>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

#include "config/utils.hpp"
#include "map/map.hpp"
#include "map/utils.hpp"
#include "optimization/objective.hpp"
#include "optimization/optimizer.hpp"
#include "optimization/utils.hpp"
#include "scan/generate.hpp"
#include "scan/utils.hpp"
#include "state/state.hpp"
#include "visualize/utils.hpp"

// Converts Eigen::SparseMatrix to CHOLMOD sparse matrix
void eigenToCholmod(const Eigen::SparseMatrix<double> &eigenMatrix, cholmod_sparse *&cholmodMatrix,
                    cholmod_common &c) {
  cholmodMatrix = cholmod_allocate_sparse(eigenMatrix.rows(), eigenMatrix.cols(),
                                          eigenMatrix.nonZeros(), true, true, -1, CHOLMOD_REAL, &c);
  if (!cholmodMatrix) {
    throw std::runtime_error("Failed to allocate CHOLMOD sparse matrix.");
  }

  auto *x = static_cast<double *>(cholmodMatrix->x);
  auto *i = static_cast<int *>(cholmodMatrix->i);
  auto *p = static_cast<int *>(cholmodMatrix->p);

  int k = 0;
  for (int col = 0; col < eigenMatrix.outerSize(); ++col) {
    p[col] = k;
    for (Eigen::SparseMatrix<double>::InnerIterator it(eigenMatrix, col); it; ++it) {
      i[k] = it.row();
      x[k++] = it.value();
    }
  }
  p[eigenMatrix.cols()] = k;
}

// Runs the optimization process
void runOptimization(ObjectiveFunctor<2> &functor, Eigen::VectorXd &params, cholmod_common &c,
                     const std::vector<pcl::PointCloud<pcl::PointXY>> &scans,
                     const MapArgs<2> &map_args,
                     const Eigen::Transform<double, 2, Eigen::Affine> &initial_frame,
                     const OptimizationArgs &opt_args) {
  Eigen::SparseMatrix<double> jacobian_sparse;
  Eigen::VectorXd residuals(functor.values());
  double lambda = opt_args.initial_lambda;

  functor(params, residuals);
  double error = residuals.squaredNorm();
  std::cout << "Initial Error: " << error << std::endl;

  for (int iter = 0; iter < opt_args.max_iters; ++iter) {
    // Compute the Jacobian matrix
    functor.sparse_df(params, jacobian_sparse);
    if (jacobian_sparse.nonZeros() == 0) {
      throw std::runtime_error("Jacobian has no non-zero entries.");
    }

    // RHS of the linear system
    Eigen::VectorXd jt_residuals = -jacobian_sparse.transpose() * residuals;
    cholmod_dense *rhs =
        cholmod_allocate_dense(jt_residuals.size(), 1, jt_residuals.size(), CHOLMOD_REAL, &c);
    std::memcpy(rhs->x, jt_residuals.data(), jt_residuals.size() * sizeof(double));

    // LHS of the linear system
    Eigen::SparseMatrix<double> jt_j = jacobian_sparse.transpose() * jacobian_sparse;
    Eigen::SparseMatrix<double> identity(jt_j.rows(), jt_j.cols());
    identity.setIdentity();
    Eigen::SparseMatrix<double> lhs_eigen = jt_j + lambda * identity;

    // Cholesky factorization
    cholmod_sparse *lhs;
    eigenToCholmod(lhs_eigen, lhs, c);
    cholmod_factor *lhs_factorised = cholmod_analyze(lhs, &c);
    cholmod_factorize(lhs, lhs_factorised, &c);

    // Solve the linear system
    cholmod_dense *cholmod_result = cholmod_solve(CHOLMOD_A, lhs_factorised, rhs, &c);
    if (!cholmod_result) {
      cholmod_free_dense(&rhs, &c);
      cholmod_free_sparse(&lhs, &c);
      throw std::runtime_error("cholmod_solve failed.");
    }

    Eigen::VectorXd delta(jt_residuals.size());
    std::memcpy(delta.data(), cholmod_result->x, jt_residuals.size() * sizeof(double));
    params += delta;

    // Update lambda based on error improvement
    functor(params, residuals);
    double new_error = residuals.squaredNorm();
    lambda = new_error < error ? lambda / opt_args.lambda_factor : lambda * opt_args.lambda_factor;
    error = new_error;

    // Logging
    const double update_norm = delta.norm();
    if (opt_args.std_out) {
      std::cout << "Iteration: " << iter << " Error: " << error << " Update Norm: " << update_norm
                << std::endl;
    }
    if (opt_args.visualize) {
      visualizeMap(params, scans, map_args, initial_frame);
    }

    // Free memory
    cholmod_free_sparse(&lhs, &c);
    cholmod_free_dense(&rhs, &c);
    cholmod_free_dense(&cholmod_result, &c);

    // Early stopping
    if (update_norm < opt_args.tolerance) break;
  }
}

int main() {
  try {
    // Setup and read configuration
    Args<2> args = setup_from_yaml<2>("../config/sdf.yml");
    Scans scans = read_scans(args.general_args.data_path);
    const std::vector<pcl::PointCloud<pcl::PointXY>> point_clouds = scans.scans;

    // // ---- New Block: Convert 2D scan (PointXY) to 3D (PointXYZ) and visualize normals ----
    // {
    //   pcl::PointCloud<pcl::PointXY>::Ptr point_cloud_ptr(
    //       new pcl::PointCloud<pcl::PointXY>(point_clouds[0]));
    //   pcl::PointCloud<pcl::Normal>::Ptr norm_cloud = compute_normals<2>(point_cloud_ptr, 0.1);

    //   // Convert the 2D cloud to a 3D cloud (with z = 0)
    //   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud3d(new pcl::PointCloud<pcl::PointXYZ>);
    //   for (const auto &pt : point_cloud_ptr->points) {
    //     pcl::PointXYZ pt3d;
    //     pt3d.x = pt.x;
    //     pt3d.y = pt.y;
    //     pt3d.z = 0;  // 2D data: z is set to zero
    //     cloud3d->points.push_back(pt3d);
    //   }
    //   cloud3d->width = static_cast<uint32_t>(cloud3d->points.size());
    //   cloud3d->height = 1;
    //   cloud3d->is_dense = true;

    //   // Set up the visualizer.
    //   pcl::visualization::PCLVisualizer viewer("2D Normal Visualization");
    //   viewer.setBackgroundColor(0.0, 0.0, 0.0);
    //   viewer.addPointCloud<pcl::PointXYZ>(cloud3d, "scan");
    //   // Visualize normals with a step of 1 and a scale factor of 10.0 (adjust as needed)
    //   viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud3d, norm_cloud, 1, 1.0,
    //                                                           "normals");
    //   viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4,
    //                                           "scan");

    //   // Main loop for the visualizer.
    //   while (!viewer.wasStopped()) {
    //     viewer.spinOnce(100);
    //     std::this_thread::sleep_for(std::chrono::milliseconds(100));
    //   }
    // }

    // Initialize the map and set up the optimization parameters
    Map<2> map = init_map(args.map_args, args.general_args.from_ground_truth, args.general_args.initial_value);
    Eigen::VectorXd params = flatten<2>(State<2>(map, scans.frames));

    // Visualize the initial map
    visualizeMap(params, point_clouds, args.map_args, scans.frames[0]);

    // Set up the objective functor for optimization
    const int num_points = point_clouds[0].size() * point_clouds.size();
    const int num_residuals =
        num_points * (args.objective_args.scanline_points + 1) + map.total_points();
    ObjectiveFunctor<2> functor(params.size(), num_residuals, args.map_args, point_clouds,
                                args.objective_args, scans.frames[0]);

    // Run the optimization process
    cholmod_common c;
    cholmod_start(&c);
    runOptimization(functor, params, c, point_clouds, args.map_args, scans.frames[0],
                    args.optimization_args);
    cholmod_finish(&c);

    // Visualize the optimized map
    visualizeMap(params, point_clouds, args.map_args, scans.frames[0]);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
