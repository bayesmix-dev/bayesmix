#include "split_and_merge_algorithm.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include "src/utils/distributions.h"

void SplitAndMergeAlgorithm::read_params_from_proto(
  const bayesmix::AlgorithmParams &params){
  BaseAlgorithm::read_params_from_proto(params);
  T = params.splitmerge_n_restr_gs_updates();
  K = params.splitmerge_n_mh_updates();
  M = params.splitmerge_n_full_gs_updates();
} 

void SplitAndMergeAlgorithm::initialize(){
  MarginalAlgorithm::initialize();

  if(mixing->get_id()!=bayesmix::MixingId::DP){
    throw std::invalid_argument(
      "Invalid mixing supplied to Split and Merge, only DP mixing supported");
  }
}

void SplitAndMergeAlgorithm::print_startup_message() const {
  std::string msg = "Running Split and Merge algorithm with " +
                    bayesmix::HierarchyId_Name(unique_values[0]->get_id()) +
                    " hierarchies, " +
                    bayesmix::MixingId_Name(mixing->get_id()) + " mixing...";
  std::cout << msg << std::endl;
}

void SplitAndMergeAlgorithm::sample_allocations(){
  auto &rng = bayesmix::Rng::Instance().get();

  // MH updates
  for(unsigned int k=0; k<K; ++k){
    unsigned int n_data = data.rows();

    // Sample i and j from the datapoints
    Eigen::VectorXd probas = Eigen::VectorXd::Constant(n_data, 1.0/n_data);
    unsigned int i = bayesmix::categorical_rng(probas, rng, 0);
    probas = Eigen::VectorXd::Constant(n_data-1, 1.0/(n_data-1));
    unsigned int j = bayesmix::categorical_rng(probas, rng, 0);
    if(j>=i){
      j++;
    }

    compute_S(i,j);
    compute_C_launch(i,j);

    // Restricted GS steps
    for(unsigned int t=0; t<T; ++t){
      restricted_GS(i,j);
    }

    split_or_merge(i,j);
  }

  // Full GS updates
  for(unsigned int k=0; k<M; ++k){
    full_GS();
  }
  
}

void SplitAndMergeAlgorithm::sample_unique_values(){
  for (auto &un : unique_values) {
    un->sample_full_cond(!update_hierarchy_params());
  }
}

Eigen::VectorXd SplitAndMergeAlgorithm::lpdf_marginal_component(
    std::shared_ptr<AbstractHierarchy> hier, const Eigen::MatrixXd &grid,
    const Eigen::RowVectorXd &covariate) const {
  return hier->prior_pred_lpdf_grid(grid, covariate);
}

void SplitAndMergeAlgorithm::compute_S(const unsigned int i, const unsigned int j) {
    unsigned int lengthS=0;
    for(int k=0; k<allocations.size(); k++){
        if(allocations[k]==allocations[i]||allocations[k]==allocations[j])
            lengthS++;
    }
    lengthS=lengthS-2;
    S={}; //Necessary, since it needs to be zero at each iter, not only at the beginning
    S.resize(lengthS,0);
    unsigned int index=0;
    for (int k = 0; k < allocations.size(); ++k) {
        if ((allocations[k]==allocations[i]||allocations[k]==allocations[j])&& (k!=i) && (k!=j)) {
            if (index>=lengthS)
                std::cerr<< "Index out of bounds. index="<<index
                    <<",lengthS="<<lengthS<< ", i="<<i<<", j="<<j<<std::endl;
            S[index]=k;
            index++;
        }
    }
}

void SplitAndMergeAlgorithm::compute_C_launch(const unsigned int i,
  const unsigned int j){
  allocations_cl.clear();
  allocations_cl.resize(S.size());

  cl.clear();
  cl.push_back(unique_values[0]->clone());
  cl.push_back(unique_values[0]->clone());
  
  auto &rng = bayesmix::Rng::Instance().get(); 
  Eigen::VectorXd probas(2);
  probas(0)=0.5;
  probas(1)=0.5;
  for (size_t k = 0; k < S.size(); ++k) {
    if (bayesmix::categorical_rng(probas, rng, 0)){
      allocations_cl[k]=1;
      cl[1]->add_datum(S[k], data.row(S[k]), false);
    }else{
      allocations_cl[k]=0;
      cl[0]->add_datum(S[k], data.row(S[k]), false);
    }
  }
  /* We update the posterior parameters only at the last insertion to ease
   * computations.
   */
  cl[0]->add_datum(i, data.row(i), true);
  cl[1]->add_datum(j, data.row(j), true);
}

// in cl (vettore di due celle) ci saranno anche i a j
void SplitAndMergeAlgorithm::split_or_merge(const unsigned int i, const unsigned int j){
  if(allocations[i]==allocations[j]){  
    std::vector<unsigned int> clSplit (allocations.size()); 
    std::shared_ptr<AbstractHierarchy> data_i(unique_values[0]->clone());
    std::shared_ptr<AbstractHierarchy> data_j(unique_values[0]->clone());  
    std::shared_ptr<AbstractHierarchy> data_J(unique_values[0]->clone());
    double p_i=0; 
    double p_j=0;
    double p_J=0;
    unsigned int LabI = unique_values.size();
    
    clSplit[i]=LabI;
    clSplit[j]=allocations[j];
    double q=std::exp(restricted_GS(i,j,true));
    unsigned int z=0;
    std::set<int> set_i=cl[0]->get_data_idx();
    std::set<int> set_j=cl[1]->get_data_idx();
    std::set<int>::const_iterator iter_i=set_i.cbegin();
    std::set<int>::const_iterator iter_j=set_j.cbegin();
    for(unsigned int k=0; k < clSplit.size(); k++){
      if(((z<S.size())and(k==S[z]))or((k==i)or(k==j))){ //since S doesn't contain i and j but we put them in data_i or data_j
        if(*iter_i==k){
          if (!data_i->get_card()){ //first iteration
            p_i+=data_i->prior_pred_lpdf(data.row(k));
          }else{
            p_i+=data_i->conditional_pred_lpdf(data.row(k));
          }
          data_i->add_datum(k, data.row(k), update_hierarchy_params());    
          iter_i++;
          clSplit[k]=LabI;
        }else{
          if (!data_j->get_card()){ //first iteration
            p_j+=data_j->prior_pred_lpdf(data.row(k));
          }else{
            p_j+=data_j->conditional_pred_lpdf(data.row(k));
          }
          data_j->add_datum(k, data.row(k), update_hierarchy_params());
          iter_j++;
          clSplit[k]=allocations[j];
        }
        if((k!=i)and(k!=j)) z++;
      }else{
        clSplit[k]=allocations[k];                   
      }
      if(allocations[k]==allocations[j]){
        if(!data_J->get_card()){
          p_J+=data_J->prior_pred_lpdf(data.row(k));
        }else{
          p_J+=data_J->conditional_pred_lpdf(data.row(k));
        }
        data_J->add_datum(k, data.row(k), update_hierarchy_params());
      }                             
    }
    const double p1=1/q;
    const double alpha=mixing->get_mass_new_cluster(allocations.size(), false,
                                       true, LabI+1); //LabI should be number of clusters - 1, alternatively unique_values.size()
     // data_i->get_card()-1 = n. di dati con LabelI senza i, data_j->get_card()-1= n. di dati con label j senza j (+1 finale perchè usiamo la funzione gamma)
    const double p2=tgamma(data_j->get_card()-1-1+1)*tgamma(data_i->get_card()-1-1+1)/(S.size()+2-1)*alpha;
    const double p3=std::exp(p_i+p_j-p_J); 
    const double AcRa=std::min(1.0,p1*p2*p3); //acceptance ratio 
    if(accepted_proposal(AcRa)){
      unique_values.push_back(unique_values[0]->clone());
      for(unsigned int k=0; k<clSplit.size(); k++){
        
        if(allocations[k]!=clSplit[k]){ //it should happen only when we pass data from labj to labi
          if(unique_values[allocations[k]]->get_card()<=1){ //just to be sure, if all data end up in labi
            remove_singleton(allocations[k]);
            unique_values[clSplit[k]-1]->add_datum(k, data.row(k), update_hierarchy_params()); //new label is always shifted back since it's the new one
            allocations[k]=clSplit[k]-1;
            break;  //no more data cluster to update
          }
          
          else{
            unique_values[allocations[k]]->remove_datum(k, data.row(k), update_hierarchy_params());
            unique_values[clSplit[k]]->add_datum(k, data.row(k), update_hierarchy_params());
            allocations[k]=clSplit[k];
          }
          
        }
      }
    }
  }else{
    std::vector<unsigned int> clMerge (allocations.size()); 
    std::shared_ptr<AbstractHierarchy> data_i(unique_values[0]->clone());
    std::shared_ptr<AbstractHierarchy> data_j(unique_values[0]->clone());  
    std::shared_ptr<AbstractHierarchy> data_J(unique_values[0]->clone());
    double p_i=0; 
    double p_j=0;
    double p_J=0;
    unsigned int LabI = allocations[i];
     
    clMerge[i]=allocations[j];
    clMerge[j]=allocations[j];
    std::set<int> set_i=cl[0]->get_data_idx();
    std::set<int> set_j=cl[1]->get_data_idx();
    std::set<int>::const_iterator iter_i=set_i.cbegin();
    std::set<int>::const_iterator iter_j=set_j.cbegin();
    unsigned int z=0;
    for(unsigned int k=0; k < clMerge.size(); k++){
      if(((z<S.size())and(k==S[z]))or((k==i)or(k==j))){
        if(*iter_i==k){
          if (!data_i->get_card()){ //first iteration
            p_i+=data_i->prior_pred_lpdf(data.row(k));
          }else{
            p_i+=data_i->conditional_pred_lpdf(data.row(k));
          }
          data_i->add_datum(k, data.row(k), update_hierarchy_params());
          iter_i++;
        }else{
          if (!data_j->get_card()){ //first iteration
            p_j+=data_j->prior_pred_lpdf(data.row(k)); 
          }
          else{
            p_j+=data_j->conditional_pred_lpdf(data.row(k));
          }
          data_j->add_datum(k, data.row(k), update_hierarchy_params());
          iter_j++;
        }
        clMerge[k]=allocations[j];
        if((k!=i)and(k!=j))z++;                                 
      }else{
        clMerge[k]=allocations[k];              
      }
      if(allocations[k]==allocations[j]){
        if(!data_J->get_card()){
          p_J+=data_J->prior_pred_lpdf(data.row(k));
        }else{
          p_J+=data_J->conditional_pred_lpdf(data.row(k));
        }
        data_J->add_datum(k, data.row(k), update_hierarchy_params());
      }                                    
    }
    double q=1; 
    //Fake Gibbs Sampling in order to compute the probability q
    for(unsigned int k=0; k<S.size(); k++){
      if(allocations_cl[k]){ //cluster j
        cl[1]->remove_datum(S[k], data.row(S[k]), update_hierarchy_params());
      }else{
        cl[0]->remove_datum(S[k], data.row(S[k]), update_hierarchy_params());
      }
      if(cl[0]->get_card()>=1 and cl[1]->get_card()>=1){
        double p_i=cl[0]->conditional_pred_lpdf(data.row(S[k]));
        double p_j=cl[1]->conditional_pred_lpdf(data.row(S[k]));
      }else{
        if(cl[1]->get_card()==0){
          double p_j=cl[1]->prior_pred_lpdf(data.row(S[k]));
          double p_i=cl[0]->conditional_pred_lpdf(data.row(S[k]));
        }else{
           double p_i=cl[0]->prior_pred_lpdf(data.row(S[k]));
           double p_j=cl[1]->conditional_pred_lpdf(data.row(S[k]));
        }  
      }
   
      double p=(p_i)/(p_i + p_j);
      if(allocations[S[k]]==allocations[i]){
        allocations_cl[k]=0;
        cl[0]->add_datum(S[k], data.row(S[k]), update_hierarchy_params());
        q=q*p;
      }else{
        allocations_cl[k]=1;
        cl[1]->add_datum(S[k], data.row(S[k]), update_hierarchy_params());
        q=q*(1-p);
      }
    }
        
    const double p1=q;
    const double alpha=mixing->get_mass_new_cluster(allocations.size(), false,
                                       true,unique_values.size());
     // data_i->get_card()-1 = n. di dati con LabelI senza i, data_j->get_card()-1= n. di dati con label j senza j
    const double p2=tgamma(data_i->get_card()-1-1+1)*tgamma(data_j->get_card()-1-1+1)/(S.size()+2-1)*alpha;
    const double p3=std::exp(-p_i-p_j+p_J); 
    const double AcRa=std::min(1.0,p1*p2*p3); //acceptance ratio 
    if(accepted_proposal(AcRa)){
      for(unsigned int k=0; k<allocations.size();k++){
        if(allocations[k]==LabI){
          
          if(unique_values[LabI]->get_card()<=1){
            remove_singleton(LabI);
            unique_values[allocations[j]]->add_datum(k,data.row(k),update_hierarchy_params()); // allocations is already updated, also for j
            allocations[k]=allocations[j];
            break;
          }
          else unique_values[LabI]->remove_datum(k,data.row(k),update_hierarchy_params());  //REVIEW: we don't have to update params since we are only interested
                                                                                                     //in deleting cluster LabI right?
          unique_values[allocations[j]]->add_datum(k,data.row(k),update_hierarchy_params()); // here we can update only at the last iteration maybe?
          allocations[k]=allocations[j];
        }
      }
    }
  } //close for  
}

bool SplitAndMergeAlgorithm::accepted_proposal(const double acRa) const{
  // std::default_random_engine generator; //old generator
  
  std::uniform_real_distribution<> UnifDis(0.0, 1.0);
  return (UnifDis(bayesmix::Rng::Instance().get())<=acRa);
}

double SplitAndMergeAlgorithm::restricted_GS(const unsigned int i, 
  const unsigned int j, bool return_log_res_prod/*=false*/){
  auto &rng = bayesmix::Rng::Instance().get();

  double log_res_prod = 0;

  for(size_t k=0; k<S.size(); ++k){
    cl[allocations_cl[k]]->remove_datum(S[k], data.row(S[k]), 
      update_hierarchy_params());

    Eigen::VectorXd logprobas(2);
    logprobas(0) = mixing->get_mass_existing_cluster(S.size()+2-1, true, true,
      cl[0]);
    logprobas(0) += cl[0]->conditional_pred_lpdf(data.row(S[k]));
    logprobas(1) = mixing->get_mass_existing_cluster(S.size()+2-1, true, true,
      cl[1]);
    logprobas(1) += cl[1]->conditional_pred_lpdf(data.row(S[k]));

    Eigen::VectorXd probas = stan::math::softmax(logprobas);
    unsigned int c_new = bayesmix::categorical_rng(probas, rng, 0);

    if(return_log_res_prod){
      log_res_prod+=log(probas(c_new));
    }

    allocations_cl[k]=c_new;
    cl[allocations_cl[k]]->add_datum(S[k], data.row(S[k]), 
      update_hierarchy_params());
  }

  return log_res_prod;                                                       
}

void SplitAndMergeAlgorithm::full_GS(){
  unsigned int n_data = data.rows();
  auto &rng = bayesmix::Rng::Instance().get();
  for(size_t i=0; i<n_data; ++i){
    bool singleton = (unique_values[allocations[i]]->get_card()<=1);
    unsigned int c_old = allocations[i];
    if(singleton){
      remove_singleton(c_old);
    }else{
      unique_values[c_old]->remove_datum(
        i, data.row(i), update_hierarchy_params());
    }
    unsigned int n_clust = unique_values.size();

    Eigen::VectorXd logprobas(n_clust+1);
    for(size_t j=0; j<n_clust; ++j){
      logprobas(j) = mixing->get_mass_existing_cluster(
        n_data-1, true, true, unique_values[j]);
      logprobas(j) += unique_values[j]->conditional_pred_lpdf(
        data.row(i));
    }
    logprobas(n_clust) = mixing->get_mass_new_cluster(
      n_data-1, true, true, n_clust);
    logprobas(n_clust) += unique_values[0]->prior_pred_lpdf(
      data.row(i));

    unsigned int c_new = 
      bayesmix::categorical_rng(stan::math::softmax(logprobas), rng, 0);

    if(c_new==n_clust){
      std::shared_ptr<AbstractHierarchy> new_unique =
        unique_values[0]->clone();
      new_unique->add_datum(i, data.row(i), update_hierarchy_params());
      new_unique->sample_full_cond(!update_hierarchy_params());
      unique_values.push_back(new_unique);
      allocations[i] = unique_values.size() - 1;
    }else{
      allocations[i] = c_new;
      unique_values[c_new]->add_datum(
        i, data.row(i), update_hierarchy_params());
    }
  }
}
