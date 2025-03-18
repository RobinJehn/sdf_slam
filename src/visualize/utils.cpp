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

cv::Mat map_to_image(const Eigen::MatrixXd &map, int output_width, int output_height,
                     const bool clamp_colour_map, double min_value, double max_value) {
  cv::Mat map_image(output_height, output_width, CV_8UC1);
  const int map_points_x = map.cols();
  const int map_points_y = map.rows();

  // Determine min and max values for color scaling
  if (!clamp_colour_map) {
    min_value = map.minCoeff();
    max_value = map.maxCoeff();
  }

  // Ensure that the min and max values are different
  if (max_value == min_value) {
    max_value = min_value + 1;
  }

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
      const double value_clamped = std::clamp(value, min_value, max_value);
      // .at(row, col)
      map_image.at<uchar>(idx_y, idx_x) =
          static_cast<uchar>(255 * (value_clamped - min_value) / (max_value - min_value));
    }
  }

  // Apply color mapping
  cv::Mat color_map_image;
  cv::applyColorMap(map_image, color_map_image, cv::COLORMAP_JET);
  return color_map_image;
}

void overlay_points(cv::Mat &image, const std::vector<Eigen::Vector2d> &points,
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

void overlay_surface_normals(cv::Mat &image, const Map<2> &map,
                             const pcl::search::KdTree<pcl::PointXY>::Ptr &tree_global,
                             const pcl::PointCloud<pcl::Normal>::Ptr &normals_global,
                             const Eigen::Vector2d &scale) {
  for (int i = 0; i < map.get_num_points(0); i++) {
    for (int j = 0; j < map.get_num_points(1); j++) {
      const typename Map<2>::index_t index = {i, j};
      typename Map<2>::Vector grid_pt = map.get_location(index);
      // Move the normal to the middle of the square
      grid_pt += 0.5 * Eigen::Vector2d(map.get_d(0), map.get_d(1));

      pcl::PointXY grid_pt_pcl;
      grid_pt_pcl.x = grid_pt.x() + 0.5 * map.get_d(0);
      grid_pt_pcl.y = grid_pt.y() + 0.5 * map.get_d(1);

      std::vector<int> nn_indices;
      std::vector<float> nn_dists;
      tree_global->nearestKSearch(grid_pt_pcl, 1, nn_indices, nn_dists);
      if (nn_indices.empty()) {
        continue;
      }
      const pcl::Normal n = normals_global->points[nn_indices[0]];

      const Eigen::Vector2d map_point = (grid_pt - map.get_min_coords()).cwiseProduct(scale);
      const int x = static_cast<int>(map_point.x());
      const int y = image.rows - static_cast<int>(map_point.y());

      const Eigen::Vector2d normal_end = grid_pt + 0.1 * Eigen::Vector2d(n.normal_x, n.normal_y);
      const Eigen::Vector2d map_normal_end =
          (normal_end - map.get_min_coords()).cwiseProduct(scale);
      const int x_end = static_cast<int>(map_normal_end.x());
      const int y_end = image.rows - static_cast<int>(map_normal_end.y());
      cv::line(image, cv::Point(x, y), cv::Point(x_end, y_end), cv::Scalar(0, 255, 0), 1);
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

void display_map(const Eigen::MatrixXd &map,  //
                 const Map<2> &map_map,       //
                 const std::vector<Eigen::Vector2d> &points,
                 const Eigen::Vector2d &min_coords,  //
                 const Eigen::Vector2d &max_coords,
                 const pcl::search::KdTree<pcl::PointXY>::Ptr &tree_global,
                 const pcl::PointCloud<pcl::Normal>::Ptr &normals_global,  //
                 const VisualizationArgs &vis_args) {
  const Eigen::Vector2d image_size = Eigen::Vector2d(vis_args.output_width, vis_args.output_height);
  const Eigen::Vector2d map_size = max_coords - min_coords;
  const Eigen::Vector2d scale = image_size.cwiseQuotient(map_size);

  cv::Mat color_map_image =
      map_to_image(map, vis_args.output_width, vis_args.output_height, vis_args.clamp_colour_map,
                   vis_args.min_value, vis_args.max_value);

  if (vis_args.show_points) {
    overlay_points(color_map_image, points, min_coords, max_coords, scale);
  }
  if (vis_args.show_normals) {
    overlay_surface_normals(color_map_image, map_map, tree_global, normals_global, scale);
  }

  CallbackData data = {map, vis_args.output_width, vis_args.output_height};
  cv::setMouseCallback("Map with Points", onMouse, &data);
  cv::imshow("Map with Points", color_map_image);
  cv::waitKey(0);
}

pcl::PointCloud<pcl::PointXY>::Ptr scans_to_global_pcl_2d(
    const std::vector<Eigen::Transform<double, 2, Eigen::Affine>> &transformations,
    const std::vector<pcl::PointCloud<pcl::PointXY>> &scans) {
  std::vector<pcl::PointCloud<pcl::PointXY>::Ptr> point_clouds_ptrs;
  for (const auto &cloud : scans) {
    pcl::PointCloud<pcl::PointXY>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXY>(cloud));
    point_clouds_ptrs.push_back(cloud_ptr);
  }
  const std::vector<pcl::PointCloud<pcl::PointXY>::Ptr> point_clouds_global =
      local_to_global(transformations, point_clouds_ptrs);
  const pcl::PointCloud<pcl::PointXY>::Ptr cloud_global = combine_scans<2>(point_clouds_global);

  return cloud_global;
}

template <int Dim>
std::vector<Eigen::Matrix<double, Dim, 1>> scans_to_global_eigen(
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

void visualize_map(const Eigen::VectorXd &params,
                   const std::vector<pcl::PointCloud<pcl::PointXY>> &scans,
                   const MapArgs<2> &map_args,
                   const Eigen::Transform<double, 2, Eigen::Affine> &initial_frame,
                   const VisualizationArgs &vis_args) {
  State<2> state = unflatten<2>(params, initial_frame, map_args);

  std::vector<Eigen::Vector2d> global_points =
      scans_to_global_eigen<2>(state.transformations_, scans);
  const pcl::PointCloud<pcl::PointXY>::Ptr cloud_global =
      scans_to_global_pcl_2d(state.transformations_, scans);
  pcl::search::KdTree<pcl::PointXY>::Ptr tree_global(new pcl::search::KdTree<pcl::PointXY>);
  tree_global->setInputCloud(cloud_global);

  // Global normals
  std::vector<pcl::PointCloud<pcl::PointXY>::Ptr> scans_ptr = cloud_to_cloud_ptr(scans);
  const pcl::PointCloud<pcl::Normal>::Ptr normals_global =
      compute_normals_global_2d(scans_ptr, state.transformations_);

  // Generate and display the map image
  Eigen::MatrixXd map(map_args.num_points[0], map_args.num_points[1]);
  for (int x = 0; x < map_args.num_points[0]; ++x) {
    for (int y = 0; y < map_args.num_points[1]; ++y) {
      // (row, column) row is y, column is x
      map(y, x) = state.map_.get_value_at({x, y});
    }
  }
  display_map(map, state.map_, global_points, map_args.min_coords, map_args.max_coords, tree_global,
              normals_global, vis_args);
}

// Explicit template instantiation for 2D points
template void plotPointsWithValuesPCL<2>(
    const std::vector<std::pair<Eigen::Matrix<double, 2, 1>, double>> &);

// Explicit template instantiation for 3D points
template void plotPointsWithValuesPCL<3>(
    const std::vector<std::pair<Eigen::Matrix<double, 3, 1>, double>> &);

// Explicit template instantiation for scans_to_global_eigen with 2D points
template std::vector<Eigen::Matrix<double, 2, 1>> scans_to_global_eigen<2>(
    const std::vector<Eigen::Transform<double, 2, Eigen::Affine>> &transformations,
    const std::vector<pcl::PointCloud<pcl::PointXY>> &scans);

// Explicit template instantiation for scans_to_global_eigen with 3D points
template std::vector<Eigen::Matrix<double, 3, 1>> scans_to_global_eigen<3>(
    const std::vector<Eigen::Transform<double, 3, Eigen::Affine>> &transformations,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>> &scans);
