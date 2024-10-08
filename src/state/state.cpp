#include "state.hpp"
#include "map/map.hpp"

template <int Dim>
State<Dim>::State(const Map<Dim> &map,
                  const std::vector<Eigen::Transform<float, Dim, Eigen::Affine>>
                      &transformations)
    : map_(map), transformations_(transformations) {}

// Explicit template instantiation if needed
template struct State<2>;
template struct State<3>;