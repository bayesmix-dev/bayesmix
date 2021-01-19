#include "memory_collector.hpp"

bool MemoryCollector::next_state(google::protobuf::Message* out) {
  curr_iter++;
  if (curr_iter == size - 1) {
    return false;
  }
  out->ParseFromString(chain[curr_iter]);
  return true;
}

void MemoryCollector::collect(const google::protobuf::Message& state) {
  std::string s;
  state.SerializeToString(&s);
  chain.push_back(s);
  size++;
}

void MemoryCollector::get_state(unsigned int i,
                                google::protobuf::Message* out) {
  out->ParseFromString(chain[i]);
}

void MemoryCollector::reset() { curr_iter = 0; }
