#include "slice_sampler.h"

#include "src/hierarchies/nnig_hierarchy.h"

void SliceSampler::print_startup_message() const {
  std::string msg = "Running SliceSampler algorithm with " +
                    bayesmix::HierarchyId_Name(unique_values[0]->get_id()) +
                    " hierarchies, " +
                    bayesmix::MixingId_Name(mixing->get_id()) + " mixing...";
  std::cout << msg << std::endl;
}

void SliceSampler::initialize() {
  ConditionalAlgorithm::initialize();
  slice_u = Eigen::VectorXd::Zero(data.rows());
  this->mixing =
      std::dynamic_pointer_cast<TruncatedSBMixing>(BaseAlgorithm::mixing);
  sample_slice();
}

void SliceSampler::step() {
  sample_unique_values();
  sample_weights();
  sample_slice();
  sample_allocations();
}

void SliceSampler::sample_slice() {
  auto &rng = bayesmix::Rng::Instance().get();
  Eigen::VectorXd weights = mixing->get_mixing_weights(false, false);
  for (int i = 0; i < data.rows(); i++) {
    slice_u(i) = stan::math::uniform_rng(0.0, weights(allocations[i]), rng);
    slice_u(i) = std::max(1e-16, slice_u(i));
  }
}

void SliceSampler::sample_unique_values() {
  for (auto &un : unique_values) {
    un->sample_full_cond(!update_hierarchy_params());
  }
}

void SliceSampler::sample_weights() {
  auto &rng = bayesmix::Rng::Instance().get();
  mixing->update_state(unique_values, allocations);
  int num_old_sticks = mixing->get_sticks().size();
  int max_alloc = *std::max_element(allocations.begin(), allocations.end());
  Eigen::VectorXd new_sticks = mixing->get_sticks();
  Eigen::VectorXd new_sticks_head = new_sticks.head(max_alloc + 1);
  mixing->set_sticks(new_sticks_head);

  // Now we compute how many more sticks we need and sample from
  // the prior
  double min_u = slice_u.minCoeff();
  double sum_w = mixing->get_mixing_weights(false, false).sum();
  int iter = 0;
  while (sum_w <= (1.0 - min_u)) {
    iter += 1;
    sum_w += mixing->keep_breaking(1);
  }

  int num_new_sticks = mixing->get_sticks().size();
  if (num_new_sticks >= num_old_sticks) {
    for (int h = num_old_sticks; h < num_new_sticks; h++) {
      std::shared_ptr<AbstractHierarchy> new_unique =
          unique_values[0]->clone();
      new_unique->sample_prior();
      unique_values.push_back(new_unique);
    }
  } else {
    for (int h = num_new_sticks; h < num_old_sticks; h++) {
      unique_values.pop_back();
    }
  }
}

void SliceSampler::sample_allocations() {
  auto &rng = bayesmix::Rng::Instance().get();
  unsigned int num_components = mixing->get_num_components();
  for (int i = 0; i < data.rows(); i++) {
    // Compute weights
    Eigen::VectorXd prior_weights = mixing->get_mixing_weights(false, false);
    std::vector<double> probas_;
    std::vector<int> inds;
    for (int j = 0; j < num_components; j++) {
      if (slice_u(i) < prior_weights(j)) {
        probas_.push_back(unique_values[j]->get_like_lpdf(
            data.row(i), hier_covariates.row(i)));
        inds.push_back(j);
      }
    }
    Eigen::VectorXd probas =
        Eigen::VectorXd::Map(probas_.data(), probas_.size());
    probas = stan::math::softmax(probas);
    // Draw a NEW value for datum allocation
    unsigned int c_new = bayesmix::categorical_rng(probas, rng, 0);
    c_new = inds[c_new];
    unsigned int c_old = allocations[i];
    if (c_new != c_old) {
      allocations[i] = c_new;
      // Remove datum from old cluster, add to new
      unique_values[c_old]->remove_datum(
          i, data.row(i), update_hierarchy_params(), hier_covariates.row(i));
      unique_values[c_new]->add_datum(
          i, data.row(i), update_hierarchy_params(), hier_covariates.row(i));
    }
  }
}
