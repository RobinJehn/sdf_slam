#include "scene.hpp"

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "shape/circle.hpp"
#include "shape/rectangle.hpp"
#include "shape/sinusoid.hpp"

void Scene::add_shape(const std::shared_ptr<Shape> &shape) { shapes_.push_back(shape); }

bool Scene::intersect_ray(const Eigen::Vector2d &origin, double angle,
                          Eigen::Vector2d &intersection) const {
  bool hit = false;
  double min_t = std::numeric_limits<double>::max();
  Eigen::Vector2d best_intersection;
  for (const auto &shape : shapes_) {
    Eigen::Vector2d pt;
    double t;
    if (shape->intersect_ray(origin, angle, pt, t)) {
      if (t > 1e-6 && t < min_t) {
        min_t = t;
        best_intersection = pt;
        hit = true;
      }
    }
  }
  if (hit) {
    intersection = best_intersection;
  }
  return hit;
}

const std::vector<std::shared_ptr<Shape>> &Scene::get_shapes() const { return shapes_; }

std::string Scene::to_string() const {
  std::ostringstream oss;
  oss << "scene " << shapes_.size() << "\n";
  for (const auto &shape : shapes_) {
    oss << shape->to_string() << "\n";
  }
  return oss.str();
}

Scene Scene::from_string(const std::string &str) {
  Scene scene;
  std::istringstream iss(str);
  std::string header;
  int numShapes = 0;
  iss >> header >> numShapes;  // Expect header "scene"
  std::string dummy;
  std::getline(iss, dummy);  // consume rest of line

  for (int i = 0; i < numShapes; ++i) {
    std::string line;
    std::getline(iss, line);
    if (line.empty()) continue;
    std::istringstream lineStream(line);
    std::string type;
    lineStream >> type;
    std::shared_ptr<Shape> shape;
    if (type == "Circle") {
      shape = Circle::from_string(line);
    } else if (type == "Sinusoid") {
      shape = Sinusoid::from_string(line);
    } else if (type == "Rectangle") {
      shape = Rectangle::from_string(line);
    }
    if (shape) {
      scene.add_shape(shape);
    }
  }
  return scene;
}

Scene Scene::from_yaml(const YAML::Node &node) {
  Scene scene;
  for (const auto &shape_node : node) {
    std::string type = shape_node["type"].as<std::string>();
    std::shared_ptr<Shape> shape;
    if (type == "circle") {
      shape = Circle::from_yaml(shape_node);
    } else if (type == "sinusoid") {
      shape = Sinusoid::from_yaml(shape_node);
    } else if (type == "rectangle") {
      shape = Rectangle::from_yaml(shape_node);
    }
    if (shape) {
      scene.add_shape(shape);
    }
  }
  return scene;
}

Scene Scene::from_file(const std::filesystem::path &path) {
  std::ifstream in_file(path.string());
  if (in_file.is_open()) {
    std::string file_content;
    std::string line;
    while (std::getline(in_file, line)) {
      file_content += line + "\n";
    }
    return Scene::from_string(file_content);
  } else {
    std::cerr << "Unable to open file to read scene information." << std::endl;
    return Scene();
  }
}
