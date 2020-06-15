#include "MemoryCollector.hpp"

State MemoryCollector::next_state() {
  if (curr_iter == size - 1) {
    curr_iter = -1;
    return chain[size - 1];
  } else {
    return chain[curr_iter];
  }
}

// \param iter_state State in Protobuf-object form to write to the collector
void MemoryCollector::collect(State iter_state) {
  chain.push_back(iter_state);
  size++;
}
