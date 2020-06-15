#ifndef HIERARCHYNNW_IMP_HPP
#define HIERARCHYNNW_IMP_HPP

#include "HierarchyNNW.hpp"


template<class Hypers>
void HierarchyNNW<Hypers>::check_state_validity(){
        // Check if tau is a square matrix
        unsigned int dim = state[0].size();
        assert(dim == state[1].rows());
        assert(dim == state[1].cols());
        // Check if tau is symmetric positive semi definite
        assert( state[1].isApprox(state[1].transpose()) );
        assert( tau_chol_factor.info() != Eigen::NumericalIssue );
}


//! \param tau Value to set to state[1]
template<class Hypers>
void HierarchyNNW<Hypers>::set_tau_and_utilities(const Eigen::MatrixXd &tau){
    if(state.size() == 1){ // e.g. if the hierarchy is being initialized
        state.push_back(tau);
    }
    else {
        state[1] = tau;
    }

    // Update tau utilities
    tau_chol_factor = Eigen::LLT<Eigen::MatrixXd>(tau);
    tau_chol_factor_eval = tau_chol_factor.matrixL();
    Eigen::VectorXd diag = tau_chol_factor_eval.diagonal();
    tau_log_det =  2*log(diag.array()).sum();
}


//! \param data                  Matrix of row-vectorial data points
//! \param mu0, lambda, tau0, nu Original values for hyperparameters
//! \return                      Vector of updated values for hyperparameters
template<class Hypers>
std::vector<Eigen::MatrixXd> HierarchyNNW<Hypers>::normal_wishart_update(
    const Eigen::MatrixXd &data, const EigenRowVec &mu0, const double lambda,
    const Eigen::MatrixXd &tau0, const double nu){
    // Initialize relevant objects
    unsigned int n = data.rows();
    Eigen::MatrixXd lambda_post(1,1), nu_post(1,1);

    // Compute updated hyperparameters
    EigenRowVec mubar = data.colwise().mean(); // sample mean
    lambda_post(0,0) = lambda + n;
    nu_post(0,0) = nu + 0.5 * n;
    EigenRowVec mu_post = (lambda*mu0 + n*mubar) * (1/(lambda+n));
    // Compute tau_post
    Eigen::MatrixXd tau_temp = Eigen::MatrixXd::Zero(data.cols(), data.cols());
    for(size_t i = 0; i < n; i++){
        EigenRowVec datum = data.row(i);
        tau_temp += (datum-mubar).transpose()*(datum-mubar); // column * row
    }
    tau_temp += (n*lambda/(n+lambda)) * (mubar-mu0).transpose()*(mubar-mu0);
    tau_temp = 0.5*tau_temp + tau0.inverse();
    Eigen::MatrixXd tau_post = tau_temp.inverse();

    return std::vector<Eigen::MatrixXd>{mu_post,lambda_post,tau_post,nu_post};
}


//! \param data Matrix of row-vectorial data points
//! \return     Likehood vector evaluated in data
template<class Hypers>
Eigen::VectorXd HierarchyNNW<Hypers>::like(const Eigen::MatrixXd &data){
    // Initialize relevant objects
    unsigned int n = data.rows();
    Eigen::VectorXd result(n);
    EigenRowVec mu(state[0]);

    for(size_t i = 0; i < n; i++){
        // Compute likelihood for each data point
        EigenRowVec datum = data.row(i);
        result(i) = std::pow(2.0*M_PI, -data.cols()/2.0) *
            std::exp( 0.5 * (tau_log_det - ((
            tau_chol_factor_eval.transpose()*(datum-mu).transpose()
            ).squaredNorm()) ) );
        // Unoptimized likelihood by Stan:
        // result(i) = std::exp(stan::math::multi_normal_prec_lpdf(
        //     datum, mu, state[1]));
    }
    return result;
}


//! \param data Matrix of row-vectorial data points
//! \return     Marginal distribution vector evaluated in data
template<class Hypers>
Eigen::VectorXd HierarchyNNW<Hypers>::eval_marg(const Eigen::MatrixXd &data){
    // Initialize relevant objects
    unsigned int n = data.rows();
    Eigen::VectorXd result(n);
    unsigned int dim = data.cols();

    // Get values of hyperparameters
    EigenRowVec mu0 = hypers->get_mu0();
    double lambda = hypers->get_lambda();
    Eigen::MatrixXd tau0 = hypers->get_tau0();
    double nu = hypers->get_nu();

    // Compute dof and scale of marginal distribution
    double nu_n = 2*nu - dim + 1;
    Eigen::MatrixXd sigma_n = tau0.inverse() * ( nu-0.5*(dim-1) ) * lambda/
        (lambda+1);

    for(size_t i = 0; i < n; i++){
        // Compute marginal for each data point
        EigenRowVec datum = data.row(i);
        result(i) = exp( stan::math::multi_student_t_lpdf(datum, nu_n, mu0,
            sigma_n) );
    }
    return result;
}


template<class Hypers>
void HierarchyNNW<Hypers>::draw(){
    // Get values of hyperparameters
    EigenRowVec mu0 = hypers->get_mu0();
    double lambda = hypers->get_lambda();
    Eigen::MatrixXd tau0 = hypers->get_tau0();
    double nu = hypers->get_nu();

    // Generate new state values from their prior centering distribution
    Eigen::MatrixXd tau_new = stan::math::wishart_rng( nu, tau0, this->rng );
    Eigen::MatrixXd sigma = tau_new.inverse();
    EigenRowVec mu_new = stan::math::multi_normal_rng( mu0, sigma*(1/lambda),
        this->rng );

    // Update state
    state[0] = mu_new;
    set_tau_and_utilities(tau_new);
}


//! \param data Matrix of row-vectorial data points
template<class Hypers>
void HierarchyNNW<Hypers>::sample_given_data(const Eigen::MatrixXd &data){
    // Get values of hyperparameters
    EigenRowVec mu0 = hypers->get_mu0();
    double lambda = hypers->get_lambda();
    Eigen::MatrixXd tau0 = hypers->get_tau0();
    double nu = hypers->get_nu();

    // Update values
    std::vector<Eigen::MatrixXd> temp = normal_wishart_update(data, mu0, lambda,
        tau0, nu);
    EigenRowVec mu_post = temp[0];
    double lambda_post = temp[1](0,0);
    Eigen::MatrixXd tau_post = temp[2];
    double nu_post = temp[3](0,0);

    // Generate new state values from their prior centering distribution
    Eigen::MatrixXd tau_new = stan::math::wishart_rng(nu_post, tau_post,
        this->rng);
    Eigen::MatrixXd sigma = tau_new.inverse();
    EigenRowVec mu_new = stan::math::multi_normal_rng(mu_post,
        sigma*(1/lambda_post), this->rng);

    // Update state
    state[0] = mu_new;
    set_tau_and_utilities(tau_new);
}


#endif // HIERARCHYNNW_IMP_HPP
