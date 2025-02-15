#include "utils.hpp"

#include <thread>

#include "optimization/utils.hpp"

template <int Dim>
void plotPointsWithValuesPCL(
    const std::vector<std::pair<Eigen::Matrix<double, Dim, 1>, double>> &points_with_values) {
  // Create a PCL point cloud of type pcl::PointXYZRGB
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

  // Find min and max values for color scaling
  double min_value = std::numeric_limits<double>::max();
  double max_value = std::numeric_limits<double>::lowest();
  for (const auto &point_value : points_with_values) {
    min_value = std::min(min_value, point_value.second);
    max_value = std::max(max_value, point_value.second);
  }

  // Scale and add each point to the point cloud
  for (const auto &point_value : points_with_values) {
    const Eigen::Matrix<double, Dim, 1> &point = point_value.first;
    double value = point_value.second;

    // Scale the value to a range between 0 and 255 for color intensity
    int intensity = static_cast<int>(255 * (value - min_value) / (max_value - min_value));

    // Create a PCL point with RGB color based on the intensity
    pcl::PointXYZRGB pcl_point;
    pcl_point.x = point(0);
    pcl_point.y = point(1);
    pcl_point.z = (Dim == 3) ? point(2) : 0;  // Set z=0 for 2D points

    // Set the color (using a blue shade here, adjust RGB channels as needed)
    pcl_point.r = intensity;
    pcl_point.g = 0;
    pcl_point.b = 255 - intensity;

    // Add point to the cloud
    cloud->push_back(pcl_point);
  }

  // Set up the PCL visualizer
  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");

  // Set point size and add a coordinate system
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,
                                           "cloud");
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();

  // Start the viewer loop
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

cv::Mat map_to_image(const Eigen::MatrixXd &map, int output_width, int output_height) {
  cv::Mat map_image(output_height, output_width, CV_8UC1);
  const double min_value = map.minCoeff();
  const double max_value = map.maxCoeff();
  const int map_points_x = map.cols();
  const int map_points_y = map.rows();

  // Fill map_image by sampling and scaling map data
  for (int idx_x = 0; idx_x < output_width; ++idx_x) {
    for (int idx_y = 0; idx_y < output_height; ++idx_y) {
      const int map_idx_x =
          static_cast<int>(idx_x * map_points_x / static_cast<double>(output_width));

      // In the image the origin is in the top left, while in the map it is in the bottom left
      const int map_idx_y = static_cast<int>((output_height - idx_y - 1) * map_points_y /
                                             static_cast<double>(output_height));
      // (row, col)
      const double value = map(map_idx_y, map_idx_x);
      // .at(row, col)
      map_image.at<uchar>(idx_y, idx_x) =
          static_cast<uchar>(255 * (value - min_value) / (max_value - min_value));
    }
  }

  // Apply color mapping
  cv::Mat color_map_image;
  cv::applyColorMap(map_image, color_map_image, cv::COLORMAP_JET);
  return color_map_image;
}

void overlayPoints(cv::Mat &image, const std::vector<Eigen::Vector2d> &points,
                   const Eigen::Vector2d &min_coords, const Eigen::Vector2d &max_coords,
                   const Eigen::Vector2d &scale) {
  for (const auto &point : points) {
    const Eigen::Vector2d map_point = (point - min_coords).cwiseProduct(scale);
    const int x = static_cast<int>(map_point.x());
    const int y = image.rows - static_cast<int>(map_point.y());
    if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
      cv::circle(image, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
    }
  }
}

struct CallbackData {
  Eigen::MatrixXd map;
  const int output_width;
  const int output_height;
};

// Mouse callback function
void onMouse(int event, int x, int y, int flags, void *userdata) {
  CallbackData *data = reinterpret_cast<CallbackData *>(userdata);

  if (event == cv::EVENT_LBUTTONDOWN) {  // || event == cv::EVENT_MOUSEMOVE
    Eigen::MatrixXd map = data->map;

    const int map_index_x =
        static_cast<int>(map.cols() * x / static_cast<double>(data->output_width));

    // In the image the origin is in the top left, while in the map it is in the bottom left
    const int map_index_y =
        map.rows() - static_cast<int>(map.rows() * y / static_cast<double>(data->output_height)) -
        1;

    // Print the pixel and map coordinates to the console.
    std::cout << "Pixel (" << x << ", " << y << "), Map (" << map_index_x << ", " << map_index_y
              << "): " << map(map_index_y, map_index_x) << std::endl;
  }
}

void displayMapWithPoints(const Eigen::MatrixXd &map, const std::vector<Eigen::Vector2d> &points,
                          const Eigen::Vector2d &min_coords, const Eigen::Vector2d &max_coords,
                          const int output_width, const int output_height) {
  const Eigen::Vector2d image_size = Eigen::Vector2d(output_width, output_height);
  const Eigen::Vector2d map_size = max_coords - min_coords;
  const Eigen::Vector2d scale = image_size.cwiseQuotient(map_size);

  cv::Mat color_map_image = map_to_image(map, output_width, output_height);
  overlayPoints(color_map_image, points, min_coords, max_coords, scale);

  CallbackData data = {map, output_width, output_height};
  cv::setMouseCallback("Map with Points", onMouse, &data);
  cv::imshow("Map with Points", color_map_image);
  cv::waitKey(0);
}

template <int Dim>
std::vector<Eigen::Matrix<double, Dim, 1>> scan_to_global(
    const std::vector<Eigen::Transform<double, Dim, Eigen::Affine>> &transformations,
    const std::vector<
        pcl::PointCloud<typename std::conditional<Dim == 2, pcl::PointXY, pcl::PointXYZ>::type>>
        &scans) {
  using Point = Eigen::Matrix<double, Dim, 1>;

  std::vector<Point> global_points;
  for (int i = 0; i < scans.size(); i++) {
    for (const auto &scanner_point : scans[i]) {
      Point scanner_p;
      scanner_p[0] = scanner_point.x;
      scanner_p[1] = scanner_point.y;
      if constexpr (Dim == 3) {
        scanner_p[2] = scanner_point.z;
      }
      Point global_p = transformations[i] * scanner_p;
      global_points.push_back(global_p);
    }
  }

  return global_points;
}

void visualizeMap(const Eigen::VectorXd &params,
                  const std::vector<pcl::PointCloud<pcl::PointXY>> &scans,
                  const MapArgs<2> &map_args,
                  const Eigen::Transform<double, 2, Eigen::Affine> &initial_frame,
                  const int output_width, const int output_height) {
  State<2> state = unflatten<2>(params, initial_frame, map_args);

  std::vector<Eigen::Vector2d> global_points = scan_to_global<2>(state.transformations_, scans);

  // Generate and display the map image
  Eigen::MatrixXd map(map_args.num_points[0], map_args.num_points[1]);
  for (int x = 0; x < map_args.num_points[0]; ++x) {
    for (int y = 0; y < map_args.num_points[1]; ++y) {
      // (row, column) row is y, column is x
      map(y, x) = std::max(-100.0, std::min(100.0, state.map_.get_value_at({x, y})));
    }
  }
  displayMapWithPoints(map, global_points, map_args.min_coords, map_args.max_coords, output_width,
                       output_height);
}

// Explicit template instantiation for 2D points
template void plotPointsWithValuesPCL<2>(
    const std::vector<std::pair<Eigen::Matrix<double, 2, 1>, double>> &);

// Explicit template instantiation for 3D points
template void plotPointsWithValuesPCL<3>(
    const std::vector<std::pair<Eigen::Matrix<double, 3, 1>, double>> &);

// Explicit template instantiation for scan_to_global with 2D points
template std::vector<Eigen::Matrix<double, 2, 1>> scan_to_global<2>(
    const std::vector<Eigen::Transform<double, 2, Eigen::Affine>> &transformations,
    const std::vector<pcl::PointCloud<pcl::PointXY>> &scans);

// Explicit template instantiation for scan_to_global with 3D points
template std::vector<Eigen::Matrix<double, 3, 1>> scan_to_global<3>(
    const std::vector<Eigen::Transform<double, 3, Eigen::Affine>> &transformations,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>> &scans);
