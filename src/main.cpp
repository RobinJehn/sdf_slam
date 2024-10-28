#include "map/map.hpp"
#include "optimization/objective.hpp"
#include "optimization/optimizer.hpp"
#include "optimization/utils.hpp"
#include "scan/generate.hpp"
#include "state/state.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <SuiteSparseQR.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>

// Converts Eigen::SparseMatrix to CHOLMOD sparse matrix
void eigenToCholmod(const Eigen::SparseMatrix<double> &eigenMatrix,
                    cholmod_sparse *&cholmodMatrix, cholmod_common &c) {
  cholmodMatrix = cholmod_allocate_sparse(
      eigenMatrix.rows(), eigenMatrix.cols(), eigenMatrix.nonZeros(), true,
      true, -1, CHOLMOD_REAL, &c);
  if (!cholmodMatrix) {
    std::cerr << "Error: Failed to allocate cholmod sparse matrix."
              << std::endl;
    return;
  }

  double *x = static_cast<double *>(cholmodMatrix->x);
  int *i = static_cast<int *>(cholmodMatrix->i);
  int *p = static_cast<int *>(cholmodMatrix->p);

  int k = 0;
  for (int col = 0; col < eigenMatrix.outerSize(); ++col) {
    p[col] = k;
    for (Eigen::SparseMatrix<double>::InnerIterator it(eigenMatrix, col); it;
         ++it) {
      i[k] = it.row();
      x[k++] = it.value();
    }
  }
  p[eigenMatrix.cols()] = k;
}

// Setup initial scan positions and angles
void setupScanPositions(double &theta1, double &theta2, Eigen::Vector2d &pos1,
                        Eigen::Vector2d &pos2) {
  theta1 = 9 * M_PI / 16;
  pos1 = {3.5, -5};
  theta2 = 8 * M_PI / 16;
  pos2 = {4, -5};
}

// Sets up and displays a PCL 3D point cloud visualization
void display3DPointClouds(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud1,
                          const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud2) {
  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud1_color(
      cloud1, 255, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ>(cloud1, cloud1_color, "cloud1");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud1");

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud2_color(
      cloud2, 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ>(cloud2, cloud2_color, "cloud2");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud2");

  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();

  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

// Creates a color-mapped OpenCV image from the optimized map
cv::Mat createColorMappedImage(const Eigen::MatrixXd &optimized_map,
                               int output_width, int output_height,
                               double x_min, double x_max, double y_min,
                               double y_max) {
  cv::Mat map_image(output_height, output_width, CV_8UC1);
  double min_value = optimized_map.minCoeff();
  double max_value = optimized_map.maxCoeff();
  int map_points_x = optimized_map.rows();
  int map_points_y = optimized_map.cols();

  // Fill map_image by sampling and scaling optimized_map data
  for (int i = 0; i < output_height; ++i) {
    for (int j = 0; j < output_width; ++j) {
      int map_i = static_cast<int>(i * map_points_x /
                                   static_cast<double>(output_height));
      int map_j = static_cast<int>(j * map_points_y /
                                   static_cast<double>(output_width));
      double value = optimized_map(map_i, map_j);
      map_image.at<uchar>(i, j) = static_cast<uchar>(255 * (value - min_value) /
                                                     (max_value - min_value));
    }
  }

  // Apply color mapping
  cv::Mat color_map_image;
  cv::applyColorMap(map_image, color_map_image, cv::COLORMAP_JET);
  return color_map_image;
}

// Overlays points on the given color-mapped image
void overlayPoints(cv::Mat &image, const std::vector<Eigen::Vector2d> &points,
                   double x_min, double y_min, double scale_x, double scale_y) {
  for (const auto &point : points) {
    int x = static_cast<int>((point.x() - x_min) * scale_x);
    int y = static_cast<int>((point.y() - y_min) * scale_y);
    if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
      cv::circle(image, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
    }
  }
}

// Displays the map in OpenCV
void displayMapWithPoints(const Eigen::MatrixXd &optimized_map,
                          const std::vector<Eigen::Vector2d> &points,
                          int output_width, int output_height, double x_min,
                          double x_max, double y_min, double y_max) {
  double scale_x = static_cast<double>(output_width) / (x_max - x_min);
  double scale_y = static_cast<double>(output_height) / (y_max - y_min);

  cv::Mat color_map_image = createColorMappedImage(
      optimized_map, output_width, output_height, x_min, x_max, y_min, y_max);
  overlayPoints(color_map_image, points, x_min, y_min, scale_x, scale_y);

  cv::imshow("Optimized Map with Points", color_map_image);
  cv::waitKey(0);
}

// Applies transformations to points based on optimized parameters and prepares
// visualization
void visualizeOptimizedMapAndPointClouds(
    const Eigen::VectorXd &params,
    const std::vector<pcl::PointCloud<pcl::PointXY>> &scans, int map_points_x,
    int map_points_y, double x_min, double x_max, double y_min, double y_max,
    int output_width = 500, int output_height = 500) {
  State<2> optimized_state = unflatten<2>(params, {map_points_x, map_points_y},
                                          {x_min, y_min}, {x_max, y_max});
  Eigen::Transform<double, 2, Eigen::Affine> optimized_frame_1 =
      optimized_state.transformations_[0];
  Eigen::Transform<double, 2, Eigen::Affine> optimized_frame_2 =
      optimized_state.transformations_[1];

  std::vector<Eigen::Vector2d> points;
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_scan_1(
      new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_scan_2(
      new pcl::PointCloud<pcl::PointXYZ>());

  for (const auto &point : scans[0]) {
    Eigen::Vector2d p(point.x, point.y);
    Eigen::Vector2d transformed_p = optimized_frame_1 * p;
    transformed_scan_1->push_back(
        pcl::PointXYZ(static_cast<float>(transformed_p.x()),
                      static_cast<float>(transformed_p.y()), 0));
    points.push_back(transformed_p);
  }

  for (const auto &point : scans[1]) {
    Eigen::Vector2d p(point.x, point.y);
    Eigen::Vector2d transformed_p = optimized_frame_2 * p;
    transformed_scan_2->push_back(
        pcl::PointXYZ(static_cast<float>(transformed_p.x()),
                      static_cast<float>(transformed_p.y()), 0));
    points.push_back(transformed_p);
  }

  // Generate and display the map image
  Eigen::MatrixXd optimized_map(map_points_x, map_points_y);
  int index = 0;
  for (int x = 0; x < map_points_x; ++x) {
    for (int y = 0; y < map_points_y; ++y) {
      optimized_map(y, x) = std::max(-3.0, std::min(3.0, params(index++)));
    }
  }
  displayMapWithPoints(optimized_map, points, output_width, output_height,
                       x_min, x_max, y_min, y_max);

  // Display the transformed point clouds
  // display3DPointClouds(transformed_scan_1, transformed_scan_2);
}

// Runs the optimization process
void runOptimization(ObjectiveFunctor<2> &functor, Eigen::VectorXd &params,
                     cholmod_common &c,
                     const std::vector<pcl::PointCloud<pcl::PointXY>> &scans,
                     int map_points_x, int map_points_y, bool visualize) {
  Eigen::SparseMatrix<double> jacobian_sparse;
  Eigen::VectorXd residuals(functor.values());
  double lambda = 1;
  int max_iters = 100;
  double tolerance = 1e-10;

  for (int iter = 0; iter < max_iters; ++iter) {
    functor.sparse_df(params, jacobian_sparse);
    functor(params, residuals);

    std::cout << "Iteration: " << iter << " Error: " << residuals.squaredNorm()
              << std::endl;

    if (jacobian_sparse.nonZeros() == 0) {
      std::cerr << "Error: Jacobian has no non-zero entries." << std::endl;
      return;
    }

    Eigen::SparseMatrix<double> jt_j =
        jacobian_sparse.transpose() * jacobian_sparse;

    Eigen::SparseMatrix<double> identity(jt_j.rows(), jt_j.cols());
    identity.setIdentity();
    Eigen::SparseMatrix<double> lm_matrix = jt_j + lambda * identity;
    Eigen::VectorXd jt_residuals = -jacobian_sparse.transpose() * residuals;

    cholmod_sparse *cholmod_lm_matrix;
    eigenToCholmod(lm_matrix, cholmod_lm_matrix, c);

    cholmod_dense *rhs = cholmod_allocate_dense(
        jt_residuals.size(), 1, jt_residuals.size(), CHOLMOD_REAL, &c);
    std::memcpy(rhs->x, jt_residuals.data(),
                jt_residuals.size() * sizeof(double));

    cholmod_factor *L = cholmod_analyze(cholmod_lm_matrix, &c);
    cholmod_factorize(cholmod_lm_matrix, L, &c);
    cholmod_dense *cholmod_result = cholmod_solve(CHOLMOD_A, L, rhs, &c);

    if (!cholmod_result) {
      std::cerr << "Error: cholmod_solve failed." << std::endl;
      cholmod_free_dense(&rhs, &c);
      cholmod_free_sparse(&cholmod_lm_matrix, &c);
      return;
    }

    Eigen::VectorXd delta(jt_residuals.size());
    std::memcpy(delta.data(), cholmod_result->x,
                jt_residuals.size() * sizeof(double));
    params += delta;

    double prev_error = residuals.squaredNorm();
    functor(params, residuals);
    double new_error = residuals.squaredNorm();

    if (delta.norm() < tolerance)
      break;
    lambda = new_error < prev_error ? lambda * 0.3 : lambda * 3;

    cholmod_free_dense(&rhs, &c);
    cholmod_free_dense(&cholmod_result, &c);
    cholmod_free_sparse(&cholmod_lm_matrix, &c);

    if (visualize) {
      visualizeOptimizedMapAndPointClouds(
          params, scans, map_points_x, map_points_y, -3, 2 * M_PI + 3, -8, 4);
    }
  }
}

int main() {
  double theta1, theta2;
  Eigen::Vector2d pos1, pos2;
  setupScanPositions(theta1, theta2, pos1, pos2);

  auto scans = create_scans(pos1, theta1, pos2, theta2);
  std::vector<pcl::PointCloud<pcl::PointXY>> point_clouds = {*scans.first,
                                                             *scans.second};

  const int map_points_x = 200;
  const int map_points_y = 200;

  // Initialize the map and set up the optimization parameters
  Map<2> map =
      init_map(-3, 2 * M_PI + 3, -8, 4, map_points_x, map_points_y, true);
  ObjectiveFunctor<2> functor(6 + map_points_x * map_points_y,
                              (scans.first->size() + scans.second->size()) * 21,
                              {map_points_x, map_points_y}, {-3, -8},
                              {2 * M_PI + 3, 4}, point_clouds, 20, true, 0.1);

  Eigen::Transform<double, 2, Eigen::Affine> initial_frame =
      Eigen::Translation<double, 2>(pos1) * Eigen::Rotation2D<double>(theta1);
  Eigen::Transform<double, 2, Eigen::Affine> initial_frame_2 =
      Eigen::Translation<double, 2>(pos2) * Eigen::Rotation2D<double>(theta2);
  Eigen::VectorXd params =
      flatten<2>(State<2>(map, {initial_frame, initial_frame_2}));

  cholmod_common c;
  cholmod_start(&c);
  runOptimization(functor, params, c, point_clouds, map_points_x, map_points_y,
                  false);
  cholmod_finish(&c);

  // Visualize the optimized map and transformed point clouds
  visualizeOptimizedMapAndPointClouds(params, point_clouds, map_points_x,
                                      map_points_y, -3, 2 * M_PI + 3, -8, 4);

  return 0;
}
