#include "utils.hpp"
#include <thread>

// Function to create a PCL point cloud with color based on associated values
template <int Dim>
void plotPointsWithValuesPCL(
    const std::vector<std::pair<Eigen::Matrix<double, Dim, 1>, double>>
        &points_with_values) {
  // Create a PCL point cloud of type pcl::PointXYZRGB
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>());

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
    int intensity =
        static_cast<int>(255 * (value - min_value) / (max_value - min_value));

    // Create a PCL point with RGB color based on the intensity
    pcl::PointXYZRGB pcl_point;
    pcl_point.x = point(0);
    pcl_point.y = point(1);
    pcl_point.z = (Dim == 3) ? point(2) : 0; // Set z=0 for 2D points

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
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();

  // Start the viewer loop
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

// Explicit template instantiation for 2D points
template void plotPointsWithValuesPCL<2>(
    const std::vector<std::pair<Eigen::Matrix<double, 2, 1>, double>> &);

// Explicit template instantiation for 3D points
template void plotPointsWithValuesPCL<3>(
    const std::vector<std::pair<Eigen::Matrix<double, 3, 1>, double>> &);