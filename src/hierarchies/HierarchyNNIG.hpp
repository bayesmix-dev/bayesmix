#ifndef HIERARCHYNNIG_HPP
#define HIERARCHYNNIG_HPP

#include "HierarchyBase.hpp"


//! Normal Normal-InverseGamma hierarchy for univariate data.

//! This class represents a hierarchy, i.e. a cluster, whose univariate data are
//! distributed according to a normal likelihood, the parameters of which have a
//! Normal-InverseGamma centering distribution. That is:
//!           phi = (mu,sig)     (state);
//! f(x_i|mu,sig) = N(mu,sig^2)  (data likelihood);
//!    (mu,sig^2) ~ G            (unique values distribution);
//!             G ~ MM           (mixture model);
//!            G0 = N-IG         (centering distribution).
//! state[0] = mu is called location, and state[1] = sig is called scale. The
//! state's hyperparameters, contained in the Hypers object, are (mu_0, lambda,
//! alpha, beta), all scalar values. Note that this hierarchy is conjugate, thus
//! the marginal and the posterior distribution are available in closed form and
//! Neal's algorithm 2 may be used with it.

//! \param Hypers Name of the hyperparameters class

template<class Hypers>
class HierarchyNNIG : public HierarchyBase<Hypers> {
protected:
    using HierarchyBase<Hypers>::state;
    using HierarchyBase<Hypers>::hypers;

    // AUXILIARY TOOLS
    //! Raises error if the state values are not valid w.r.t. their own domain
    void check_state_validity() override;

    //! Returns updated values of the prior hyperparameters via their posterior
    std::vector<double> normal_gamma_update(const Eigen::VectorXd &data,
        const double mu0, const double alpha0, const double beta0,
        const double lambda);

public:
    //! Returns true if the hierarchy models multivariate data (here, false)
    bool is_multivariate() const override {return false;}

    // DESTRUCTOR AND CONSTRUCTORS
    ~HierarchyNNIG() = default;
    HierarchyNNIG(std::shared_ptr<Hypers> hypers_) {
        hypers = hypers_;
        state = std::vector<Eigen::MatrixXd>(2,Eigen::MatrixXd(1,1));
        state[0](0,0) = hypers->get_mu0();
        state[1](0,0) = sqrt(hypers->get_beta0()/(hypers->get_alpha0()-1));
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
};


#include "HierarchyNNIG.imp.hpp"

#endif // HIERARCHYNNIG_HPP
