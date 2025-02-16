#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <Eigen/SparseCore>  // Triplet
#include <vector>

#include "map/map.hpp"

/**
 * @brief Fills the smoothness derivative of a 2D map using the upwind method.
 *
 * This function calculates the derivative of the smoothness term with respect to the map
 * using an upwind differencing scheme.
 *
 * @param map The 2D map for which the smoothness derivative is to be calculated.
 * @param smoothness_factor A factor that scales the smoothness term. Higher values enforce
 *        stronger smoothness constraints.
 */
void fill_dSmoothness_dMap_2d_upwind(const Map<2> &map, const double smoothness_factor,
                                     pcl::search::KdTree<pcl::PointXY>::Ptr &tree_global,
                                     const pcl::PointCloud<pcl::Normal>::Ptr &normals_global,
                                     std::vector<Eigen::Triplet<double>> &triplet_list,
                                     const int residual_index_offset);

/**
 * @brief Computes the smoothness derivative of a 2D map and fills the triplet list with the
 * results.
 *
 * This function calculates the derivative of the smoothness term with respect to the map in a 2D
 * setting. It uses a KdTree for nearest neighbor search and the provided normals to compute the
 * derivative.
 *
 * @param map The 2D map for which the smoothness derivative is computed.
 * @param smoothness_factor A factor that influences the smoothness calculation.
 * @param tree_global A pointer to a KdTree used for nearest neighbor search.
 * @param normals_global A pointer to a point cloud containing the normals of the global map.
 * @param triplet_list A reference to a list of Eigen triplets where the results will be stored.
 * @param residual_index_offset An offset for the residual index to correctly place the results in
 * the triplet list.
 */
void fill_dSmoothness_dMap_2d_forward(const Map<2> &map, const double smoothness_factor,
                                      pcl::search::KdTree<pcl::PointXY>::Ptr &tree_global,
                                      const pcl::PointCloud<pcl::Normal>::Ptr &normals_global,
                                      std::vector<Eigen::Triplet<double>> &triplet_list,
                                      const int residual_index_offset);

/**
 * @brief Fills the triplet list with the derivatives of the smoothness term with respect to the map
 * in 2D.
 *
 * This function computes the derivatives of the smoothness term for a given 2D map and fills the
 * provided triplet list with these derivatives. The smoothness term is influenced by the smoothness
 * factor, and the function uses a KD-tree for nearest neighbor searches and a point cloud of
 * normals for the computations.
 *
 * @param map The 2D map for which the derivatives are computed.
 * @param smoothness_factor The factor that influences the smoothness term.
 * @param tree_global A pointer to a KD-tree used for nearest neighbor searches.
 * @param normals_global A pointer to a point cloud containing the normals of the global map.
 * @param triplet_list A reference to a vector of Eigen triplets where the computed derivatives will
 * be stored.
 * @param residual_index_offset The offset for the residual index.
 * @param type The type of derivative to be computed.
 */
void fill_dSmoothness_dMap_2d(const Map<2> &map, const double smoothness_factor,
                              pcl::search::KdTree<pcl::PointXY>::Ptr &tree_global,
                              const pcl::PointCloud<pcl::Normal>::Ptr &normals_global,
                              std::vector<Eigen::Triplet<double>> &triplet_list,
                              const int residual_index_offset, const DerivativeType type);
