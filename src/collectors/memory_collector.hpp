#ifndef BAYESMIX_COLLECTORS_MEMORY_COLLECTOR_HPP_
#define BAYESMIX_COLLECTORS_MEMORY_COLLECTOR_HPP_

#include "base_collector.hpp"

//! Class for a "virtual" collector which contains all objects of the chain

//! This is a type of collector which includes a deque containing all states of
//! the chain. Unlike the file collector, this is a purely "virtual" collector,
//! in the sense that it does not write chain states anywhere and all
//! information contained in it is destroyed when the main that created it is
//! terminated. It is therefore useful in situation in which writing to a file
//! is not needed, for instance in a main program that both runes and algorithm
//! and computes the estimates.

class MemoryCollector : public BaseCollector {
 protected:
  //! Deque that contains all states in Protobuf-object form
  std::deque<std::string> chain;

  //! Reads the next state, based on the curr_iter curson
  bool next_state(google::protobuf::Message* out) override {
    curr_iter++;
    if (curr_iter == size - 1) {
      return false;
    }
    out->ParseFromString(chain[curr_iter]);
  }

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~MemoryCollector() = default;
  MemoryCollector() = default;

  //! Initializes collector (here, it does nothing)
  void start() override { return; }
  //! Closes collector (here, it does nothing)
  void finish() override { return; }

  //! Writes the given state to the collector
  void collect(const google::protobuf::Message& state) override {
    std::string s;
    state.SerializeToString(&s);
    chain.push_back(s);
    size++;
  }

  // GETTERS AND SETTERS
  //! Returns i-th state in the collector
  void get_state(unsigned int i, google::protobuf::Message* out) override {
    out->ParseFromString(chain[i]);
  }

  // //! Returns the whole chain in form of a deque of States
  // std::deque<MsgType> get_chain() override { return chain; }

  template <typename MsgType>
  void write_to_file(std::string outfile) {
    // THIS is probabily a HACK: we de-serialize from the chain
    // and re-serialize to file. Still it's a reasonable work-around.

    int outfd = open(outfile.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0777);
    google::protobuf::io::FileOutputStream* fout =
        new google::protobuf::io::FileOutputStream(outfd);

    for (std::string& serialized_state : chain) {
      MsgType state;
      state.ParseFromString(serialized_state);
      bool success =
          google::protobuf::util::SerializeDelimitedToZeroCopyStream(state,
                                                                     fout);
    }
  }
};

#endif  // BAYESMIX_COLLECTORS_MEMORY_COLLECTOR_HPP_
