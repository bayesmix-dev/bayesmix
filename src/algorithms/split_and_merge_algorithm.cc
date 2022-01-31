#include "split_and_merge_algorithm.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

void SplitAndMergeAlgorithm::read_params_from_proto(
  const bayesmix::AlgorithmParams &params){
  BaseAlgorithm::read_params_from_proto(params);
  T = params.splitmerge_n_restr_gs_updates();
  K = params.splitmerge_n_mh_updates();
  M = params.splitmerge_n_full_gs_updates();
} 

void SplitAndMergeAlgorithm::print_startup_message() const {
  std::string msg = "Running Split and Merge algorithm with " +
                    bayesmix::HierarchyId_Name(unique_values[0]->get_id()) +
                    " hierarchies, " +
                    bayesmix::MixingId_Name(mixing->get_id()) + " mixing...";
  std::cout << msg << std::endl;
}



void SplitAndMergeAlgorithm::compute_S(const unsigned int i, const unsigned int j) {
    unsigned int lengthS;
    for(int k; k<allocations.size(); k++){
        if(allocations[k]==allocations[i]||allocations[k]==allocations[j])
            lengthS++;
    }
    lengthS=lengthS-2;
    S={}; //Necessary, since it needs to be zero at each iter, not only at the beginning
    S=S.resize(lengthS,0);
    unsigned int index=0;
    for (int k = 0; k < allocations.size(); ++k) {
        if ((allocations[k]==allocations[i]||allocations[k]==allocations[j])&& (k!=i) && (k!=j)) {
            if (index>=lengthS)
                std::cerr<< "Index out of bounds. index="<<index
                    <<",lengthS="<<lengthS<< " allocations="<<allocations<<", i="<<i<<", j="<<j<<std::endl;
            S[index]=k;
            index++;
        }
    }
}

std::vector<unsigned int> SplitAndMergeAlgorithm::compute_C_launch(const unsigned int i, const unsigned int j){
    if (allocations[i]==allocations[j]) {
        LabI = *max_element(allocations.begin(), allocations.end()) + 1;
    }else{
        LabI=allocations[i];
    }
    std::vector<unsigned int>cl(S.size(),LabI);
    std::default_random_engine generator;
    std::bernoulli_distribution bdistr(0.5);
    for (int k = 0; k < S.size(); ++k) {
        if (bdistr(generator))
            cl[k]=allocations[j];
    }
    return cl;
}

void SplitAndMergeAlgorithm::split_or_merge(std::vector<unsigned int>& cl, const unsigned int i, const unsigned int j){
    if(allocations[i]==allocations[j]) { 
      LabI=*(std::max_element(allocations.begin(), allocations.end()));
      std::vector<unsigned int> clSplit (allocations.size()); #we could initialize the vector to LAbI
      Eigen::MatrixXd data_i();  
      Eigen::MatrixXd data_j();  
      Eigen::MatrixXd data_J();
      double p_i=0; 
      double p_j=0;
      double p_J=0;
      
      clSplit[i]=LabI;
      clSplit[j]=allocations[j];
      unsigned int CountLabI=0;
      unsigned int CountLabJ=0;
      double q=1.0;
      restricted_GS(cl,i,j,q);
      unsigned int z=0;
      unsigned int I=i;
      for(unsigned int i=0; i < clSplit.size(); i++){
          if((z<S.size())and(i==S[z])){
            if(cl[z]==LabI){
              CountLabI++;
              if (!data_i.rows()){ #first iteration
                p_i+=prior_pred_lpdf(data.row(S[z])) 
                                 }
              else{
                p_i+=conditional_pred_lpdf(data.row(S[z]), data_i) 
                  }
              data_i.conservativeResize(data_i.rows()+1,NoChange);
              data_i.row(data_i.rows()-1)=data.row(i);              
                          }
            else{
              CountLabJ++;
              if (!data_j.rows()){ #first iteration
                p_j+=prior_pred_lpdf(data.row(S[z])) 
                                 }
              else{
                p_j+=conditional_pred_lpdf(data.row(S[z]), data_j)
                  }
              data_j.conservativeResize(data_j.rows()+1,NoChange);
              data_j.row(data_j.rows()-1)=data.row(i)
                 }
            
            clSplit[i]=cl[z];
            z++;
                                      }
          else{
            if(i!=I and i!=j){
            clSPlit[i]=allocations[i];
                             }
              }
          if(allocations[i]==clSplit[j]){
            if(!data_J.rows()){
              p_J+=prior_pred_lpdf(data.row(i))
                              }
            else{
              p_J+=conditional_pred_lpdf(data.row(i), data_J)
                }
            data_J.conservativeResize(data_J.rows()+1,NoChange);
            data_J.row(data_J.rows()-1)=data.row(i)
                                        }
                                            
                                              }
      const double p1=1/q;
      const double p2=factorial(CountLabI-1)*factorial(CountLabJ-1)/(S.size()+2-1)*hierarchy.alpha; //alpha da fissare
      const double p3=std::exp(p_i+p_j-p_J); 
      const double AcRa=min(1,p1*p2*p3) //acceptance ratio 
      if(accepted_proposal(AcRa)) allocations=clSplit;
      }
  else{
    std::vector<unsigned int> clMerge (allocations.size()); 
    Eigen::MatrixXd data_i();  
    Eigen::MatrixXd data_j(); 
    Eigen::MatrixXd data_J();
    double p_i=0; 
    double p_j=0;
    double p_J=0;
     
    clMerge[i]=allocations[j];
    clMerge[j]=allocations[j];
    unsigned int CountLabI=0;
    unsigned int CountLabJ=0;
    unsigned int z=0;
    unsigned int I=i;
     for(unsigned int i=0; i < clMerge.size(); i++){
          if((z<S.size())and(i==S[z])){
            if(cl[z]==allocations[I]){
              CountLabI++;
              if (!data_i.rows()){ #first iteration
                p_i+=prior_pred_lpdf(data.row(S[z])) 
                                 }
              else{
                p_i+=conditional_pred_lpdf(data.row(S[z]), data_i) 
                  }
              data_i.conservativeResize(data_i.rows()+1,NoChange);
              data_i.row(data_i.rows()-1)=data.row(i);              
                          }
            else{
              CountLabJ++;
              if (!data_j.rows()){ #first iteration
                p_j+=prior_pred_lpdf(data.row(S[z])) 
                                 }
              else{
                p_j+=conditional_pred_lpdf(data.row(S[z]), data_j) 
                  }
              data_j.conservativeResize(data_j.rows()+1,NoChange);
              data_j.row(data_j.rows()-1)=data.row(i)
                 }
            
            clMerge[i]=allocations[j];
            z++;
                                      }      
          else{
            if(i!=I and i!=j){
            clMerge[i]=allocations[i];
                             }
              }
          if(allocations[i]==clMerge[j]){
            if(!data_J.rows()){
              p_J+=prior_pred_lpdf(data.row(i))
                              }
            else{
              p_J+=conditional_pred_lpdf(data.row(i), data_J)
                }
            data_J.conservativeResize(data_J.rows()+1,NoChange);
            data_J.row(data_J.rows()-1)=data.row(i)
                                        }
                                            
                                              }
      double q=1; 
      //Fake Gibbs Sampling in order to compute the probability q
      std::vector<unsigned int> cl_copy(cl);
      for(unsigned int k=0; k<S.size(); k++){
        double p_i=ComputeRestrGSProbabilities(cl_copy, i, j, k, cluster='i');
        double p_j=ComputeRestrGSProbabilities(cl_copy, i, j, k, cluster='j');
        double p=(p_i)/(p_i + p_j);
        cl_copy[k]=allocations[S[k]];
        if(cl_copy[k]==allocations[i]) q=q*p;
        else q=q*(1-p);
                                             }
        
      const double p1=q;
      const double p2=factorial(CountLabI-1)*factorial(CountLabJ-1)/(S.size()+2-1)*hierarchy.alpha; //fissare alpha
      const double p3=std::exp(-p_i-p_j+p_J); 
      const double AcRa=min(1,p1*p2*p3) #acceptance ratio 
      if(accepted_proposal(AcRa)) allocations=clMerge;
      }
    
}

bool SplitAndMergeAlgorithm::accepted_proposal(const double acRa) const{
    std::default_random_engine generator;
    std::uniform_real_distribution UnifDis(0.0, 1.0);
    return (UnifDis(generator)<=acRa);
                                                                        }

// standard Gibbs Sampling
void SplitAndMergeAlgorithm::restricted_GS(std::vector<unsigned int>& cl, const unsigned int i, 
                   const unsigned int j) const{ 
  for(unsigned int i=0; i<S.size(); i++){
    LabI=*(std::max_element(allocations.begin(), allocations.end())); #bisogna mettere LabI come _private
    p_i = ComputeRestrGSProbabilities(cl, i, j, z, 'i');
    p_j = ComputeRestrGSProbabilities(cl, i, j, z, 'j');
    p   = p_i/(p_i+p_j);  
    cl[i]= (accepted_proposal(p)) ? LabI : cl[i];
                                         }                                                        
                                                           }

// Modified Gibbs Sampling
void SplitAndMergeAlgorithm::restricted_GS(std::vector<unsigned int>& cl, const unsigned int i, 
                   const unsigned int j, double &res_prod)const{ 
  for(unsigned int i=0; i<S.size(); i++){
    LabI=*(std::max_element(allocations.begin(), allocations.end())); #bisogna mettere LabI come _private
    p_i = ComputeRestrGSProbabilities(cl, i, j, z, 'i');
    p_j = ComputeRestrGSProbabilities(cl, i, j, z, 'j');
    p   = p_i/(p_i+p_j);  
    cl[i]= (accepted_proposal(p)) ? LabI : allocations[j];
    if(cl[i]==LabI) res_prod=res_prod*p;
    else res_prod=res_prod*(1-p);
                                         }                                                        
                                                           }

double SplitAndMergeAlgorithm::ComputeRestrGSProbabilities(std::vector<unsigned int>& cl,
                    const unsigned int i, const unsigned int j, const unsigned int z,const char cluster='i') const{
    if(cluster!='i' and cluster!='j'){
      std::cerr<<"Unexpected value for the parameter cluster ";
      return 0.0;
                                      }
    else{
      unsigned int label=0;
      if(cluster=='i')label=LabI;
      else label=allocations[j];
      std::vector<unsigned int> v;
      v.reserve(cl.size());
      for(unsigned int k=0; k<cl.size(); k++){
        if(cl[k]==label && S[k]!=S[z]){
          v.push_back(S[k]);
                                      }
                                              }
      if(cluster=='i') v.push_back(i);
      else v.push_back(j);
      Eigen::MatrixXd ExtractedData;
      ExtractedData=data(v,Eigen::all);
      if(ExtractedData.rows()==0){
        std::cerr<<"No data points in one of the two clusters considered for restricted Gibbs sampling."+
                    "This is impossible, indeed there should always be at "+"least i or j in the datapoints.+
                    "least i or j in the datapoints."<<std::endl;
        return 0.0;
                                    }
      else return v.size()*std::exp(conditional_pred_lpdf(data(S[z],Eigen::all),ExtractedData));
         }
   
 }
