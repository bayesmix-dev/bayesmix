#ifndef HIERARCHYNNW_HPP
#define HIERARCHYNNW_HPP

#include "HierarchyBase.hpp"


//! Normal Normal-Wishart hierarchy for multivariate data.

//! This class represents a hierarchy, i.e. a cluster, whose multivariate data
//! are distributed according to a multinomial normal likelihood, the parameters
//! of which have a Normal-Wishart centering distribution. That is:
//!           phi = (mu,tau)   (state);
//! f(x_i|mu,tau) = N(mu,tau)  (data likelihood);
//!      (mu,tau) ~ G          (unique values distribution);
//!             G ~ MM         (mixture model);
//!            G0 = N-W        (centering distribution).
//! state[0] = mu is called location, and state[1] = tau is called precision.
//! The state's hyperparameters, contained in the Hypers object, are (mu0,
//! lambda, tau0, nu), which are respectively vector, scalar, matrix, and
//! scalar. Note that this hierarchy is conjugate, thus the marginal and the
//! posterior distribution are available in closed form and Neal's algorithm 2
//! may be used with it.

//! \param Hypers Name of the hyperparameters class

template<class Hypers>
class HierarchyNNW : public HierarchyBase<Hypers> {
protected:
    using HierarchyBase<Hypers>::state;
    using HierarchyBase<Hypers>::hypers;
    using EigenRowVec = Eigen::Matrix<double, 1, Eigen::Dynamic>;

    // UTILITIES FOR LIKELIHOOD COMPUTATION
    //! Lower factor object of the Cholesky decomposition of tau
    Eigen::LLT<Eigen::MatrixXd> tau_chol_factor;
    //! Matrix-form evaluation of tau_chol_factor
    Eigen::MatrixXd tau_chol_factor_eval;
    //! Determinant of tau in logarithmic scale
    double tau_log_det;

    // AUXILIARY TOOLS
    //! Raises error if the state values are not valid w.r.t. their own domain
    void check_state_validity() override;
    //! Special setter tau and its utilities
    void set_tau_and_utilities(const Eigen::MatrixXd &tau);

    //! Returns updated values of the prior hyperparameters via their posterior
    std::vector<Eigen::MatrixXd> normal_wishart_update(
        const Eigen::MatrixXd &data, const EigenRowVec &mu0,
        const double lambda, const Eigen::MatrixXd &tau0, const double nu);

public:
    //! Returns true if the hierarchy models multivariate data (here, true)
    bool is_multivariate() const override {return true;}

    // DESTRUCTOR AND CONSTRUCTORS
    ~HierarchyNNW() = default;
    HierarchyNNW(std::shared_ptr<Hypers> hypers_) {
        hypers = hypers_;
        unsigned int dim = hypers->get_mu0().size();
        state.push_back( hypers->get_mu0() );
        set_tau_and_utilities( hypers->get_lambda() *
            Eigen::MatrixXd::Identity(dim, dim) );
    }

    // EVALUATION FUNCTIONS
    //! Evaluates the likelihood of data in the given points
    Eigen::VectorXd like(const Eigen::MatrixXd &data) override;
    //! Evaluates the marginal distribution of data in the given points
    Eigen::VectorXd eval_marg(const Eigen::MatrixXd &data) override;

    // SAMPLING FUNCTIONS
    //! Generates new values for state from the centering prior distribution
    void draw() override;
    //! Generates new values for state from the centering posterior distribution
    void sample_given_data(const Eigen::MatrixXd &data) override;

    // GETTERS AND SETTERS
    //! \param state_ State value to set
    //! \param check  If true, a state validity check occurs after assignment
    void set_state(const std::vector<Eigen::MatrixXd> &state_,
        bool check = true) override {
        state[0] = state_[0];
        set_tau_and_utilities(state_[1]);
        if(check){
            check_state_validity();
        }
    }
};

#include "HierarchyNNW.imp.hpp"

#endif // HIERARCHYNNW_HPP
