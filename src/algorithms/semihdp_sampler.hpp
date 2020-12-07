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
  std::vector<std::vector<std::shared_ptr<BaseHierarchy>>> theta_star;
  // idiosincratic tables
  std::vector<std::vector<std::shared_ptr<BaseHierarchy>>> theta_tilde;
  // shared tables
  std::vector<std::shared_ptr<BaseHierarchy>> taus;
  // counts
  std::vector<std::vector<int>> n_by_theta_star;

  // stuff for pseudoprior
  std::vector<std::vector<std::shared_ptr<BaseHierarchy>>> theta_star_pseudo;
  std::vector<std::vector<int>> n_by_theta_star_pseudo;

  // theta_{ij} = theta_{c_i, s_{ij}}
  std::vector<std::vector<int>> s;
  // which restaurant each grups enters from
  std::vector<int> c;
  std::vector<bool> is_used_c;
  Eigen::VectorXd omega;
  std::vector<std::vector<int>> t, v;

  // number of theta_stars equal to tau_h across all groups
  std::vector<int> m;

  double w = 0.5;
  double alpha, gamma, a_w, b_w;

  std::vector<MemoryCollector<bayesmix::MarginalState>> pseudoprior_collectors;
  int pseudo_iter;

  bool adapt = false;
  std::string c_update;

 public:
  SemiHdpSampler() {}
  ~SemiHdpSampler() {}

  SemiHdpSampler(const std::vector<Eigen::MatrixXd> &data,
                 std::shared_ptr<BaseHierarchy> hier,
                 bayesmix::SemiHdpParams params);

  void initialize();

  void step() {
    update_unique_vals();
    update_w();
    if (!adapt) {
      sample_pseudo_prior();
      update_c();
    }
    update_omega();
    update_t();
    update_s();
    relabel();
  }

  void run(int adapt_iter, int burnin, int iter, int thin,
           BaseCollector<bayesmix::SemiHdpState> *collector,
           const std::vector<MemoryCollector<bayesmix::MarginalState>>
               &pseudoprior_collectors) {
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
        if ((i + 1) % 100 == 0) {
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
      if ((i + 1) % 100 == 0) {
        std::cout << "Burn-in iter: " << i + 1 << " / " << burnin << std::endl;
      }
    }

    for (int i = 0; i < iter; i++) {
      step();
      if (iter % thin == 0) collector->collect(get_state_as_proto());
      if ((i + 1) % 100 == 0) {
        std::cout << "Running iter: " << i + 1 << " / " << iter << std::endl;
      }
    }
  }

  void update_unique_vals();
  void update_s();
  void update_t();
  void update_c();
  void update_w();
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
  std::vector<std::vector<int>> get_s() const { return s; }
  std::vector<std::vector<int>> get_t() const { return t; }
  std::vector<std::vector<int>> get_v() const { return v; }
  std::vector<int> get_m() const { return m; }
  std::vector<int> get_c() const { return c; }

  void set_s(const std::vector<std::vector<int>> &s_) { s = s_; }
  void set_t(const std::vector<std::vector<int>> &t_) { t = t_; }
  void set_v(const std::vector<std::vector<int>> &v_) { v = v_; }
  void set_m(std::vector<int> &m_) { m = m_; }
  void set_c(const std::vector<int> &c_) { c = c_; }

  void set_theta_star_debug(std::vector<int> sizes) {
    theta_star.resize(sizes.size());
    for (int i = 0; i < sizes.size(); i++) {
      theta_star[i].resize(sizes[i]);
    }
  }

  void set_theta_tilde_debug(std::vector<int> sizes) {
    theta_tilde.resize(sizes.size());
    for (int i = 0; i < sizes.size(); i++) {
      theta_tilde[i].resize(sizes[i]);
    }
  }

  void set_tau_debug(int size) { taus.resize(size); }

  std::shared_ptr<BaseHierarchy> get_theta_star(int r, int l) const {
    return theta_star[r][l];
  }
  std::shared_ptr<BaseHierarchy> get_tau(int h) { return taus[h]; }
  std::shared_ptr<BaseHierarchy> get_theta_tilde(int r, int l) {
    return theta_tilde[r][l];
  }

  void print_debug_string();

  void check();
};

#endif
