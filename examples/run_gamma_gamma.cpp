#include <iostream>

#include "src/includes.h"
#include "gamma_gamma_hier.hpp"


Eigen::MatrixXd simulate_data(int ndata) {
    Eigen::MatrixXd data(ndata, 1);
    auto& rng = bayesmix::Rng::Instance().get();
    for (int i=0; i < ndata; i++) {
        if (stan::math::uniform_rng(0, 1, rng) < 0.5) {
            data(i, 0) = stan::math::gamma_rng(1, 5, rng);
        } else {
            data(i, 0) = stan::math::gamma_rng(1, 0.5, rng);
        }
    }
    return data;
}

int main() {
    auto hier = std::make_shared<GammaGammaHierarchy>();
    hier->set_hypers(1.0, 2.0, 2.0);
    
    bayesmix::DPPrior mix_prior;
    double totalmass = 1.0;
    mix_prior.mutable_fixed_value()->set_totalmass(totalmass);
    auto mixing = MixingFactory::Instance().create_object("DP");
    mixing->get_mutable_prior()->CopyFrom(mix_prior);
    mixing->set_num_components(5);

    auto algo = AlgorithmFactory::Instance().create_object("Neal8");
    MemoryCollector* coll = new MemoryCollector();

    Eigen::MatrixXd data = simulate_data(200);
    algo->set_mixing(mixing);
    algo->set_data(data);
    algo->set_hierarchy(hier);

    bayesmix::AlgorithmParams params;
    params.set_algo_id("Neal8");
    params.set_rng_seed(0);
    params.set_burnin(1000);
    params.set_iterations(2000);
    params.set_init_num_clusters(10);
    params.set_neal8_n_aux(3);

    algo->read_params_from_proto(params);
    algo->run(coll);

    std::cout << "Done" << std::endl;

    delete coll;
}