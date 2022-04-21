#include "distributions.h"

#include <Eigen/Dense>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <random>
#include <stan/math/prim.hpp>

#include "src/utils/proto_utils.h"

int bayesmix::categorical_rng(const Eigen::VectorXd &probas, std::mt19937 &rng,
                              const int start /*= 0*/) {
  return stan::math::categorical_rng(probas, rng) + (start - 1);
}

double bayesmix::multi_normal_prec_lpdf(const Eigen::VectorXd &datum,
                                        const Eigen::VectorXd &mean,
                                        const Eigen::MatrixXd &prec_chol,
                                        const double prec_logdet) {
  using stan::math::NEG_LOG_SQRT_TWO_PI;
  double base = prec_logdet + NEG_LOG_SQRT_TWO_PI * datum.size();
  double exp = (prec_chol * (datum - mean)).squaredNorm();
  return 0.5 * (base - exp);
}

Eigen::VectorXd bayesmix::multi_normal_prec_lpdf_grid(
    const Eigen::MatrixXd &data, const Eigen::VectorXd &mean,
    const Eigen::MatrixXd &prec_chol, const double prec_logdet) {
  using stan::math::NEG_LOG_SQRT_TWO_PI;
  Eigen::VectorXd exp =
      ((data.rowwise() - mean.transpose()) * prec_chol.transpose())
          .rowwise()
          .squaredNorm();
  Eigen::VectorXd base = Eigen::ArrayXd::Ones(data.rows()) * prec_logdet +
                         NEG_LOG_SQRT_TWO_PI * data.cols();
  return (base - exp) * 0.5;
}

Eigen::VectorXd bayesmix::multi_normal_diag_rng(
    const Eigen::VectorXd &mean,
    const Eigen::DiagonalMatrix<double, Eigen::Dynamic> &cov_diag,
    std::mt19937 &rng) {
  size_t N = mean.size();
  Eigen::VectorXd output(N);
  for (size_t i = 0; i < N; i++) {
    output[i] = stan::math::normal_rng(0, 1, rng);
  }
  return output.cwiseProduct(cov_diag.diagonal().cwiseSqrt()) + mean;
}

Eigen::VectorXd bayesmix::multi_normal_prec_chol_rng(
    const Eigen::VectorXd &mean, const Eigen::LLT<Eigen::MatrixXd> &prec_chol,
    std::mt19937 &rng) {
  size_t N = mean.size();
  Eigen::VectorXd output(N);
  boost::variate_generator<std::mt19937 &, boost::normal_distribution<>>
      std_normal_rng(rng, boost::normal_distribution<>(0, 1));

  Eigen::VectorXd z(N);
  for (int i = 0; i < N; i++) {
    z(i) = std_normal_rng();
  }

  output = mean + prec_chol.matrixU().solve(z);

  return output;
}

double bayesmix::multi_normal_lpdf_woodbury_chol(
    const Eigen::RowVectorXd &datum, const Eigen::VectorXd &mean,
    const Eigen::DiagonalMatrix<double, Eigen::Dynamic> &sigma_diag_inverse,
    const Eigen::MatrixXd &wood_factor, const double &cov_logdet) {
  using stan::math::NEG_LOG_SQRT_TWO_PI;
  double exp =
      -0.5 * ((datum.transpose() - mean)
                  .dot(sigma_diag_inverse * (datum.transpose() - mean)) -
              (wood_factor * (datum.transpose() - mean)).squaredNorm());

  double base = -0.5 * cov_logdet + NEG_LOG_SQRT_TWO_PI * mean.size();
  return base + exp;
}

double bayesmix::multi_normal_lpdf_woodbury(const Eigen::VectorXd &datum,
                                            const Eigen::VectorXd &mean,
                                            const Eigen::VectorXd &sigma_diag,
                                            const Eigen::MatrixXd &lambda) {
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> sigma_diag_inverse =
      sigma_diag.cwiseInverse().asDiagonal();
  auto [wood_chol, cov_logdet] =
      compute_wood_chol_and_logdet(sigma_diag_inverse, lambda);

  return multi_normal_lpdf_woodbury_chol(datum, mean, sigma_diag_inverse,
                                         wood_chol, cov_logdet);
}

std::pair<Eigen::MatrixXd, double> bayesmix::compute_wood_chol_and_logdet(
    const Eigen::DiagonalMatrix<double, Eigen::Dynamic> &sigma_diag_inverse,
    const Eigen::MatrixXd &lambda) {
  int q = lambda.cols();
  Eigen::MatrixXd temp_chol =
      (lambda.transpose() * sigma_diag_inverse * lambda +
       Eigen::MatrixXd::Identity(q, q))
          .llt()
          .matrixL()
          .solve(Eigen::MatrixXd::Identity(q, q));

  double cov_logdet =
      -2 * Eigen::MatrixXd(temp_chol).diagonal().array().log().sum() -
      sigma_diag_inverse.diagonal().array().log().sum();

  Eigen::MatrixXd wood_chol =
      temp_chol * lambda.transpose() * sigma_diag_inverse;

  return std::make_pair(wood_chol, cov_logdet);
}

double bayesmix::multi_student_t_invscale_lpdf(
    const Eigen::VectorXd &datum, const double df, const Eigen::VectorXd &mean,
    const Eigen::MatrixXd &invscale_chol, const double scale_logdet) {
  int dim = datum.size();
  double exp =
      0.5 * (df + dim) *
      std::log(1 + (invscale_chol * (datum - mean)).squaredNorm() / df);
  double base = stan::math::lgamma((df + dim) * 0.5) -
                stan::math::lgamma(df * 0.5) - (0.5 * dim) * std::log(df) -
                (0.5 * dim) * stan::math::LOG_PI + 0.5 * scale_logdet;
  return base - exp;
}

Eigen::VectorXd bayesmix::multi_student_t_invscale_lpdf_grid(
    const Eigen::MatrixXd &data, const double df, const Eigen::VectorXd &mean,
    const Eigen::MatrixXd &invscale_chol, const double scale_logdet) {
  int dim = data.cols();
  int n = data.rows();
  double base_coeff = stan::math::lgamma((df + dim) * 0.5) -
                      stan::math::lgamma(df * 0.5) -
                      (0.5 * dim) * std::log(df) -
                      (0.5 * dim) * stan::math::LOG_PI + 0.5 * scale_logdet;
  Eigen::VectorXd base = Eigen::VectorXd::Ones(n) * base_coeff;
  Eigen::VectorXd quadforms =
      ((data.rowwise() - mean.transpose()) * invscale_chol.transpose())
          .rowwise()
          .squaredNorm();
  Eigen::VectorXd exp =
      (quadforms.array() / df + 1.0).log() * 0.5 * (df + dim);
  return base - exp;
}

double bayesmix::gaussian_mixture_dist(const Eigen::VectorXd &means1,
                                       const Eigen::VectorXd &vars1,
                                       const Eigen::VectorXd &weights1,
                                       const Eigen::VectorXd &means2,
                                       const Eigen::VectorXd &vars2,
                                       const Eigen::VectorXd &weights2) {
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

double bayesmix::gaussian_mixture_dist(
    const std::vector<Eigen::VectorXd> &means1,
    const std::vector<Eigen::MatrixXd> &precs1,
    const Eigen::VectorXd &weights1,
    const std::vector<Eigen::VectorXd> &means2,
    const std::vector<Eigen::MatrixXd> &precs2,
    const Eigen::VectorXd &weights2) {
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
    const std::vector<bayesmix::AlgorithmState::ClusterState> &clus1,
    const Eigen::VectorXd &weights1,
    const std::vector<bayesmix::AlgorithmState::ClusterState> &clus2,
    const Eigen::VectorXd &weights2) {
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
