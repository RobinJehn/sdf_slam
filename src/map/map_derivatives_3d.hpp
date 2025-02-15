#pragma once

#include <array>

#include "map.hpp"

/**
 * @brief Computes the upwind difference in a 3D map at a given index.
 *
 * This function calculates the upwind difference for a 3D map at the specified index.
 * The upwind difference is a numerical differentiation technique used to approximate
 * the gradient of a function.
 *
 * @param map The 3D map from which to compute the upwind difference.
 * @param index The index in the 3D map at which to compute the upwind difference.
 * @return A std::array containing the upwind differences in the x, y, and z directions.
 */
std::array<double, 3> upwind_difference_3d(const Map<3> &map, const Map<3>::index_t &index);

/**
 * @brief Computes the central difference in 3D for a given map at a specified index.
 *
 * This function calculates the central difference approximation of the gradient
 * at a specific index in a 3D map. The central difference method is used to
 * approximate the derivative by considering the values at neighboring points.
 *
 * @param map The 3D map from which to compute the central difference.
 * @param index The index at which to compute the central difference.
 * @return A std::array containing the central difference approximations in the
 *         x, y, and z directions.
 */
std::array<double, 3> central_difference_3d(const Map<3> &map, const Map<3>::index_t &index);

/**
 * @brief Computes the derivative of a 3D map using the specified derivative type.
 *
 * This function calculates the derivative of a 3D map using either the upwind
 * difference or central difference method, as specified by the DerivativeType.
 *
 * @param map The 3D map from which to compute the derivative.
 * @param type The type of derivative to compute (upwind or central difference).
 * @return A std::array of 3D maps containing the derivatives in the x, y, and z directions.
 */
std::array<Map<3>, 3> df_3d(const Map<3> &map, const DerivativeType &type);
