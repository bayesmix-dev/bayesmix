#include "nnw_updater.h"

#include "algorithm_state.pb.h"
#include "src/hierarchies/likelihoods/states/includes.h"
#include "src/hierarchies/priors/hyperparams.h"
#include "src/utils/proto_utils.h"

AbstractUpdater::ProtoHypersPtr NNWUpdater::compute_posterior_hypers(
    AbstractLikelihood& like, AbstractPriorModel& prior) {
  // Likelihood and Prior downcast
  auto& likecast = downcast_likelihood(like);
  auto& priorcast = downcast_prior(prior);

  // Getting required quantities from likelihood and prior
  int card = likecast.get_card();
  Eigen::VectorXd data_sum = likecast.get_data_sum();
  Eigen::MatrixXd data_sum_squares = likecast.get_data_sum_squares();
  auto hypers = priorcast.get_hypers();

  // No update possible
  if (card == 0) {
    return prior.get_hypers_proto();
  }

  // Compute posterior hyperparameters
  Eigen::VectorXd mean;
  double var_scaling, deg_free;
  Eigen::MatrixXd scale, scale_inv, scale_chol;
  var_scaling = hypers.var_scaling + card;
  deg_free = hypers.deg_free + card;
  Eigen::VectorXd mubar = data_sum.array() / card;  // sample mean
  mean = (hypers.var_scaling * hypers.mean + card * mubar) /
         (hypers.var_scaling + card);

  // Compute tau_n
  Eigen::MatrixXd tau_temp =
      data_sum_squares - card * mubar * mubar.transpose();
  tau_temp += (card * hypers.var_scaling / (card + hypers.var_scaling)) *
              (mubar - hypers.mean) * (mubar - hypers.mean).transpose();
  scale_inv = tau_temp + hypers.scale_inv;
  int dim = mean.size();
  scale = scale_inv.llt().solve(Eigen::MatrixXd::Identity(dim, dim));
  scale_chol = Eigen::LLT<Eigen::MatrixXd>(scale).matrixU();

  // Proto conversion
  bayesmix::to_proto(mean, post_hypers.mutable_nnw_state()->mutable_mean());
  post_hypers.mutable_nnw_state()->set_var_scaling(var_scaling);
  post_hypers.mutable_nnw_state()->set_deg_free(deg_free);
  bayesmix::to_proto(scale, post_hypers.mutable_nnw_state()->mutable_scale());
  bayesmix::to_proto(scale_chol,
                     post_hypers.mutable_nnw_state()->mutable_scale_chol());
  return std::make_shared<ProtoHypers>(post_hypers);
}
