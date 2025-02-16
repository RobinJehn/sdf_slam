#pragma once

#include <array>

#include "map.hpp"

/**
 * @brief Computes the upwind difference for a 2D map at a given index.
 *
 * This function calculates the upwind difference for a 2D map, which is a numerical
 * method used to approximate derivatives. It is particularly useful in solving
 * hyperbolic partial differential equations where the direction of the wind or
 * flow is important.
 *
 * @param map The 2D map for which the upwind difference is to be computed.
 * @param index The index within the map at which the upwind difference is to be calculated.
 * @return A std::array containing the upwind differences in the x and y directions.
 */
std::array<double, 2> upwind_difference_2d(const Map<2> &map, const Map<2>::index_t &index);

/**
 * @brief Computes the central difference approximation of the gradient at a given index in a 2D
 * map.
 *
 * This function calculates the gradient of the map at the specified index using the central
 * difference method. The central difference method approximates the derivative by averaging the
 * differences between neighboring points.
 *
 * @param map The 2D map from which the gradient is to be computed.
 * @param index The index at which the gradient is to be computed.
 * @return A std::array containing the gradient components in the x and y directions.
 */
std::array<double, 2> central_difference_2d(const Map<2> &map, const Map<2>::index_t &index);

/**
 * @brief Computes the forward difference for a 2D map at a given index.
 *
 * @param map The 2D map for which the forward difference is to be computed.
 * @param index The index within the map at which the forward difference is to be calculated.
 * @return A std::array containing the forward differences in the x and y directions.
 */
std::array<double, 2> forward_difference_2d(const Map<2> &map, const Map<2>::index_t &index);

/**
 * @brief Computes the 2D derivatives of a given map.
 *
 * This function calculates the derivatives of a 2D map based on the specified derivative type.
 *
 * @param map The input map for which the derivatives are to be computed.
 * @param type The type of derivative to compute (e.g., gradient, Laplacian).
 * @return A std::array containing two Map<2> objects representing the computed derivatives.
 */
std::array<Map<2>, 2> df_2d(const Map<2> &map, const DerivativeType &type);
