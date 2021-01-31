#include "distributions.hpp"

#include <Eigen/Dense>
#include <random>
#include <stan/math/prim.hpp>

#include "src/utils/proto_utils.hpp"

int bayesmix::categorical_rng(const Eigen::VectorXd &probas,
                              std::mt19937_64 &rng, int start /*= 0*/) {
  return stan::math::categorical_rng(probas, rng) + (start - 1);
}

double bayesmix::multi_normal_prec_lpdf(const Eigen::VectorXd &datum,
                                        const Eigen::VectorXd &mean,
                                        const Eigen::MatrixXd &prec_chol,
                                        double prec_logdet) {
  using stan::math::NEG_LOG_SQRT_TWO_PI;
  double base = prec_logdet + NEG_LOG_SQRT_TWO_PI * datum.size();
  double exp = (prec_chol * (datum - mean)).squaredNorm();
  return 0.5 * (base - exp);
}

double bayesmix::gaussian_mixture_dist(
    Eigen::VectorXd means1, Eigen::VectorXd vars1, Eigen::VectorXd weights1,
    Eigen::VectorXd means2, Eigen::VectorXd vars2, Eigen::VectorXd weights2) {
  double mix1 = 0.0;
#pragma omp parallel for collapse(2) reduction(+ : mix1)
  for (int i = 0; i < means1.size(); i++) {
    for (int j = 0; j < means1.size(); j++) {
      mix1 += weights1(i) * weights1(j) *
              std::exp(stan::math::normal_lpdf(means1(i), means1(j),
                                               vars1(i) + vars1(j)));
    }
  }

  double mix2 = 0.0;
#pragma omp parallel for collapse(2) reduction(+ : mix2)
  for (int i = 0; i < means2.size(); i++) {
    for (int j = 0; j < means2.size(); j++) {
      mix2 += weights2(i) * weights2(j) *
              std::exp(stan::math::normal_lpdf(means2(i), means2(j),
                                               vars2(i) + vars2(j)));
    }
  }

  double inter = 0.0;
#pragma omp parallel for collapse(2) reduction(+ : inter)
  for (int i = 0; i < means1.size(); i++) {
    for (int j = 0; j < means2.size(); j++) {
      inter += weights1(i) * weights2(j) *
               std::exp(stan::math::normal_lpdf(means1(i), means2(j),
                                                vars1(i) + vars2(j)));
    }
  }

  return mix1 + mix2 - 2 * inter;
}

double bayesmix::gaussian_mixture_dist(std::vector<Eigen::VectorXd> means1,
                                       std::vector<Eigen::MatrixXd> precs1,
                                       Eigen::VectorXd weights1,
                                       std::vector<Eigen::VectorXd> means2,
                                       std::vector<Eigen::MatrixXd> precs2,
                                       Eigen::VectorXd weights2) {
  std::vector<Eigen::MatrixXd> vars1;
  std::vector<Eigen::MatrixXd> vars2;

  for (const auto &p : precs1) vars1.push_back(stan::math::inverse_spd(p));

  for (const auto &p : precs2) vars2.push_back(stan::math::inverse_spd(p));

  double mix1 = 0.0;
  for (int i = 0; i < means1.size(); i++) {
    for (int j = 0; j < means1.size(); j++) {
      Eigen::MatrixXd var_ij = vars1[i] + vars1[j];
      mix1 += weights1(i) * weights1(j) *
              std::exp(
                  stan::math::multi_normal_lpdf(means1[i], means1[j], var_ij));
    }
  }

  double mix2 = 0.0;
  for (int i = 0; i < means2.size(); i++) {
    for (int j = 0; j < means2.size(); j++) {
      Eigen::MatrixXd var_ij = vars2[i] + vars2[j];
      mix2 += weights2(i) * weights2(j) *
              std::exp(
                  stan::math::multi_normal_lpdf(means2[i], means2[j], var_ij));
    }
  }

  double inter = 0.0;
  for (int i = 0; i < means1.size(); i++) {
    for (int j = 0; j < means2.size(); j++) {
      Eigen::MatrixXd var_ij = vars1[i] + vars2[j];
      inter += weights1(i) * weights2(j) *
               std::exp(stan::math::multi_normal_lpdf(means1[i], means2[j],
                                                      var_ij));
    }
  }

  return mix1 + mix2 - 2 * inter;
}

double bayesmix::gaussian_mixture_dist(
    std::vector<bayesmix::MarginalState::ClusterState> clus1,
    Eigen::VectorXd weights1,
    std::vector<bayesmix::MarginalState::ClusterState> clus2,
    Eigen::VectorXd weights2) {
  double out;

  if (clus1[0].has_uni_ls_state()) {
    Eigen::VectorXd means1(clus1.size());
    Eigen::VectorXd vars1(clus1.size());
    Eigen::VectorXd means2(clus2.size());
    Eigen::VectorXd vars2(clus2.size());
    for (int i = 0; i < clus1.size(); i++) {
      means1(i) = clus1[i].uni_ls_state().mean();
      vars1(i) = clus1[i].uni_ls_state().var();
    }

    for (int i = 0; i < clus2.size(); i++) {
      means2(i) = clus2[i].uni_ls_state().mean();
      vars2(i) = clus2[i].uni_ls_state().var();
    }

    out = gaussian_mixture_dist(means1, vars1, weights1, means2, vars2,
                                weights2);
  } else if (clus1[0].has_multi_ls_state()) {
    std::vector<Eigen::VectorXd> means1, means2;
    std::vector<Eigen::MatrixXd> precs1, precs2;

    for (const auto &c : clus1) {
      means1.push_back(bayesmix::to_eigen(c.multi_ls_state().mean()));
      precs1.push_back(bayesmix::to_eigen(c.multi_ls_state().prec()));
    }

    for (const auto &c : clus2) {
      means2.push_back(bayesmix::to_eigen(c.multi_ls_state().mean()));
      precs2.push_back(bayesmix::to_eigen(c.multi_ls_state().prec()));
    }

    out = gaussian_mixture_dist(means1, precs1, weights1, means2, precs2,
                                weights2);
  } else {
    throw std::invalid_argument("Parameter type not recognized");
  }

  return out;
}
