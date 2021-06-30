#include "memory_collector.h"

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

bool MemoryCollector::next_state(google::protobuf::Message* out) {
  if (curr_iter == size) {
    return false;
  }
  out->ParseFromString(chain[curr_iter]);
  curr_iter++;
  return true;
}
