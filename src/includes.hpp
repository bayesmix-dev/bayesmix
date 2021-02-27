#ifndef BAYESMIX_INCLUDES_HPP_
#define BAYESMIX_INCLUDES_HPP_

#include "algorithms/load_algorithms.h"
#include "algorithms/neal2_algorithm.h"
#include "algorithms/neal8_algorithm.h"
#include "clustering/ClusterEstimator.hpp"
#include "clustering/lossfunction/BinderLoss.hpp"
#include "clustering/lossfunction/LossFunction.hpp"
#include "clustering/lossfunction/VariationInformation.hpp"
#include "clustering/uncertainty/CredibleBall.hpp"
#include "collectors/file_collector.h"
#include "collectors/memory_collector.h"
#include "hierarchies/load_hierarchies.h"
#include "hierarchies/nnig_hierarchy.h"
#include "hierarchies/nnw_hierarchy.h"
#include "mixings/dirichlet_mixing.h"
#include "mixings/load_mixings.h"
#include "mixings/pityor_mixing.h"
#include "runtime/factory.h"
#include "utils/cluster_utils.h"
#include "utils/io_utils.h"
#include "utils/proto_utils.h"

#endif  // BAYESMIX_INCLUDES_HPP_
