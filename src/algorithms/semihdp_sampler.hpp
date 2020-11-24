#ifndef SRC_ALGORITHMS_SEMI_HDP_SAMPLER_HPP
#define SRC_ALGORITHMS_SEMI_HDP_SAMPLER_HPP

#include <proto/cpp/semihdp.pb.h>

#include <Eigen/Dense>
#include <numeric>
#include <src/collectors/memory_collector.hpp>
#include <src/hierarchies/nnig_hierarchy.hpp>
#include <src/utils/distributions.hpp>
#include <src/utils/rng.hpp>
#include <stan/math/prim.hpp>
#include <stdexcept>
#include <vector>
#include <omp.h>

using bayesmix::MarginalState;
using bayesmix::SemiHdpState;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class SemiHdpSampler {
 protected:
  std::vector<MatrixXd> data;  // one vector per group
  int ngroups;
  std::vector<int> n_by_group;

  NNIGHierarchy master_hierarchy;

  // these should be just copies
  std::vector<std::vector<NNIGHierarchy>> theta_star;
  // idiosincratic tables
  std::vector<std::vector<NNIGHierarchy>> theta_tilde;
  // shared tables
  std::vector<NNIGHierarchy> taus;
  // counts
  std::vector<std::vector<int>> n_by_theta_star;

  // stuff for pseudoprior
  std::vector<std::vector<NNIGHierarchy>> theta_star_pseudo;
  std::vector<std::vector<int>> n_by_theta_star_pseudo;

  // theta_{ij} = theta_{c_i, s_{ij}}
  std::vector<std::vector<int>> s;
  // which restaurant each grups enters from
  std::vector<int> c;
  std::vector<bool> is_used_c;
  VectorXd omega;
  //
  std::vector<std::vector<int>> t;
  std::vector<std::vector<int>> v;
  //   std::vector<std::vector<int>> h;

  // number of theta_stars equal to tau_h across all groups
  std::vector<int> m;

  double w = 0.5;

  double alpha = 1;
  double gamma = 1;

  double a_w = 2;
  double b_w = 2;

  std::vector<MemoryCollector<MarginalState>> pseudoprior_collectors;
  int pseudo_iter;

  bool adapt = false;
  std::string c_update;

 public:
  SemiHdpSampler() {}
  ~SemiHdpSampler() {}

  SemiHdpSampler(const std::vector<MatrixXd> &data, std::string c_update="full");

  void initialize();

  void step() {
    update_unique_vals();
    // check();

    update_w();
    // check();

    if (!adapt) {
      sample_pseudo_prior();
      update_c();
      // check();
    }

    update_omega();

    update_s();
    relabel();
    // check();
  }

  void run(int adapt_iter, int burnin, int iter, int thin,
           BaseCollector<bayesmix::SemiHdpState> *collector,
           const std::vector<MemoryCollector<MarginalState>>
               &pseudoprior_collectors) {
    this->pseudoprior_collectors = pseudoprior_collectors;
    std::cout << "Run, number of pseudoprior_collectors: "
              << this->pseudoprior_collectors.size() << std::endl;
    pseudo_iter = pseudoprior_collectors[0].get_size();

    initialize();
    update_unique_vals();

    // for (int i=i; i < ngroups; i++) {
    //   reassign_group(i, 0, i);
    // }

    // update_c();

    if (adapt_iter > 0) {
      adapt = true;
      for (int i = 0; i < adapt_iter; i++) {
        step();
        if ((i+1) % 100 == 0) {
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
  // void update_c_metropolis();
  void update_w();
  void update_omega();

  void relabel();
  void sample_pseudo_prior();
  void perturb(MarginalState::ClusterVal *out);

  double semihdp_marg_lpdf(const VectorXd& datum);

  double lpdf_for_group(int i, int r);

  void reassign_group(int i, int new_r, int old_r);

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

  NNIGHierarchy get_theta_star(int r, int l) const { return theta_star[r][l]; }

  NNIGHierarchy get_tau(int h) { return taus[h]; }

  NNIGHierarchy get_theta_tilde(int r, int l) { return theta_tilde[r][l]; }

  void print_debug_string();

  void check();
};

#endif