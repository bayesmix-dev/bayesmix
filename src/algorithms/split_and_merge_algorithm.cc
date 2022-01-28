#include "split_and_merge_algorithm.h"
#include <iostream>
#include <algorithm>
#include <random>

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
