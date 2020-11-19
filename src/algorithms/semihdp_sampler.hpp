#ifndef SRC_ALGORITHMS_SEMI_HDP_SAMPLER_HPP
#define SRC_ALGORITHMS_SEMI_HDP_SAMPLER_HPP

#include <proto/cpp/semihdp.pb.h>

#include <Eigen/Dense>
#include <src/collectors/memory_collector.hpp>
#include <src/hierarchies/nnig_hierarchy.hpp>
#include <src/utils/distributions.hpp>
#include <src/utils/rng.hpp>
#include <stan/math/prim.hpp>
#include <vector>

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

  MemoryCollector<bayesmix::SemiHdpState> pseudoprior_collector;
  int pseudo_iter;

 public:
  SemiHdpSampler() {}
  ~SemiHdpSampler() {}

  SemiHdpSampler(const std::vector<MatrixXd> &data);

  void initialize();

  void step() {
    update_unique_vals();
    update_s();
    update_w();
    update_c();
    relabel();
  }

  void pseudo_step() {
    update_unique_vals();
    update_s();
    relabel();
    update_w();
  }

  void run(int pseudo_burn, int pseudo_iter, int burnin, int iter, int thin,
           BaseCollector<bayesmix::SemiHdpState> *collector) {
    this->pseudo_iter = pseudo_iter;
    for (int i=0; i < pseudo_burn; i++) pseudo_step();

    for (int i = 0; i < pseudo_iter; i++) {
      pseudo_step();
      collect_pseudo();
    }

    std::cout << "Finished Pseudo Chain" << std::endl;
    print_debug_string();

    for (int i=0; i < burnin; i++) step();

    for (int i = 0; i < iter; i++) {
      step();
      if (iter % thin == 0)
        collector->collect(get_state_as_proto());
    }
  }

  void update_unique_vals();

  void update_s();
  void update_t();
  void update_c();
  void update_w();
  void relabel();
  void collect_pseudo();
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
};

#endif