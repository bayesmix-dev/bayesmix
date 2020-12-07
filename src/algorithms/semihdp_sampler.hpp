#ifndef SRC_ALGORITHMS_SEMI_HDP_SAMPLER_HPP
#define SRC_ALGORITHMS_SEMI_HDP_SAMPLER_HPP

#include <omp.h>
#include <proto/cpp/semihdp.pb.h>

#include <Eigen/Dense>
#include <numeric>
#include <src/collectors/memory_collector.hpp>
#include <src/hierarchies/base_hierarchy.hpp>
#include <src/utils/distributions.hpp>
#include <src/utils/rng.hpp>
#include <stan/math/prim.hpp>
#include <stdexcept>
#include <vector>

using bayesmix::MarginalState;
using bayesmix::SemiHdpState;

/*
 * This class implements the algorithm for posterior simulation under the
 * semi-hierarchical Dirichlet process in [1].
 * 
 * Extra goodies: we can now tune the pseudo-prior generation, by randomly
 * drawing the cardinalities from a multinomial distribution, and perturbing
 * the atoms of the mixing measure.
 *
 * [1]: "The semi-hierarchical Dirichlet Process and its application to
 * clustering homogeneous distributions", Beraha, Guglielmi and Quintana
 * arXiv: 2005.10287
 */

class SemiHdpSampler {
 protected:
  bayesmix::SemiHdpParams params;

  std::vector<Eigen::MatrixXd> data;  // one vector per group
  int ngroups;
  std::vector<int> n_by_group;

  std::shared_ptr<BaseHierarchy> G0_master_hierarchy;
  std::shared_ptr<BaseHierarchy> G00_master_hierarchy;

  // these should be just copies
  std::vector<std::vector<std::shared_ptr<BaseHierarchy>>> rest_tables;
  // idiosincratic tables
  std::vector<std::vector<std::shared_ptr<BaseHierarchy>>> private_tables;
  // shared tables
  std::vector<std::shared_ptr<BaseHierarchy>> shared_tables;
  // counts
  std::vector<std::vector<int>> n_by_table;

  // stuff for pseudoprior
  std::vector<std::vector<std::shared_ptr<BaseHierarchy>>> rest_tables_pseudo;
  std::vector<std::vector<int>> n_by_table_pseudo;

  // theta_{ij} = theta_{c_i, s_{ij}}
  std::vector<std::vector<int>> table_allocs;
  // which restaurant each grups enters from
  std::vector<int> rest_allocs;
  std::vector<bool> is_used_rest;
  Eigen::VectorXd omega;
  std::vector<std::vector<int>> table_to_shared, table_to_private;

  // number of theta_stars equal to tau_h across all groups
  std::vector<int> m;

  double semihdp_weight = 0.5;
  double totalmass_rest, totalmass_hdp;

  std::vector<MemoryCollector<bayesmix::MarginalState>> pseudoprior_collectors;
  int pseudo_iter;

  bool adapt = false;

 public:
  SemiHdpSampler() {}
  ~SemiHdpSampler() {}

  SemiHdpSampler(const std::vector<Eigen::MatrixXd> &data,
                 std::shared_ptr<BaseHierarchy> hier,
                 bayesmix::SemiHdpParams params);

  void initialize();

  void step() {
    update_unique_vals();
    update_semihdp_weight();
    if (!adapt) {
      sample_pseudo_prior();
      update_rest_allocs();
    }
    update_omega();
    update_to_shared();
    update_table_allocs();
    relabel();
  }

  void run(int adapt_iter, int burnin, int iter, int thin,
           BaseCollector<bayesmix::SemiHdpState> *collector,
           const std::vector<MemoryCollector<bayesmix::MarginalState>>
               &pseudoprior_collectors,
           bool display_progress=false, int log_every=1) {
    this->pseudoprior_collectors = pseudoprior_collectors;
    std::cout << "Run, number of pseudoprior_collectors: "
              << this->pseudoprior_collectors.size() << std::endl;
    pseudo_iter = pseudoprior_collectors[0].get_size();

    initialize();
    update_unique_vals();

    if (adapt_iter > 0) {
      adapt = true;
      for (int i = 0; i < adapt_iter; i++) {
        step();
        if (display_progress & (i + 1) % log_every == 0) {
          std::cout << "Adapt iter: " << i << " / " << adapt_iter << std::endl;
        }
      }
      adapt = false;
    }

    print_debug_string();
    sample_pseudo_prior();

    std::cout << "Beginning" << std::endl;
    for (int i = 0; i < burnin; i++) {
      step();
      if (display_progress & (i + 1) % log_every == 0) {
        std::cout << "Burn-in iter: " << i + 1 << " / " << burnin << std::endl;
      }
    }

    for (int i = 0; i < iter; i++) {
      step();
      if (iter % thin == 0) collector->collect(get_state_as_proto());
      if (display_progress && (i + 1) % log_every == 0) {
        std::cout << "Running iter: " << i + 1 << " / " << iter << std::endl;
      }
    }
  }

  void update_unique_vals();
  void update_table_allocs();
  void update_to_shared();
  void update_rest_allocs();
  void update_semihdp_weight();
  void update_omega();

  void relabel();
  void sample_pseudo_prior();
  void perturb(bayesmix::MarginalState::ClusterState *out);

  double lpdf_for_group(int i, int r);
  void reassign_group(int i, int new_r, int old_r);

  Eigen::VectorXd _compute_mixture_distance(int i);
  void _count_m();
  void _count_n_by_theta_star();

  bayesmix::SemiHdpState get_state_as_proto();
  std::vector<std::vector<int>> get_table_allocs() const { return table_allocs; }
  std::vector<std::vector<int>> get_to_shared() const {
    return table_to_shared;
  }
  std::vector<std::vector<int>> get_to_private() const { return table_to_private; }
  std::vector<int> get_m() const { return m; }
  std::vector<int> get_rest_allocs() const { return rest_allocs; }

  void set_table_allocs(const std::vector<std::vector<int>> &s_) { table_allocs = s_; }
  void set_to_shared(const std::vector<std::vector<int>> &t_) { table_to_shared = t_; }
  void set_to_private(const std::vector<std::vector<int>> &v_) { table_to_private = v_; }
  void set_m(std::vector<int> &m_) { m = m_; }
  void set_rest_allocs(const std::vector<int> &c_) { rest_allocs = c_; }

  void set_rest_tables_debug(std::vector<int> sizes) {
    rest_tables.resize(sizes.size());
    for (int i = 0; i < sizes.size(); i++) {
      rest_tables[i].resize(sizes[i]);
    }
  }

  void set_private_tables_debug(std::vector<int> sizes) {
    private_tables.resize(sizes.size());
    for (int i = 0; i < sizes.size(); i++) {
      private_tables[i].resize(sizes[i]);
    }
  }

  void set_shared_tables_debug(int size) { shared_tables.resize(size); }

  std::shared_ptr<BaseHierarchy> get_table(int r, int l) const {
    return rest_tables[r][l];
  }
  std::shared_ptr<BaseHierarchy> get_shared_table(int h) { return shared_tables[h]; }
  std::shared_ptr<BaseHierarchy> get_private_table(int r, int l) {
    return private_tables[r][l];
  }

  void print_debug_string();

  void check();
};

#endif
