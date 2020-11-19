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

template <typename MsgType>
class MemoryCollector : public BaseCollector {
 protected:
  //! Deque that contains all states in Protobuf-object form
  std::deque<MsgType> chain;

  //! Reads the next state, based on the curr_iter curson
  MsgType next_state() override {
    if (curr_iter == size - 1) {
      curr_iter = -1;
      return chain[size - 1];
    } else {
      return chain[curr_iter];
    }
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
  void collect(MsgType iter_state) override {
    chain.push_back(iter_state);
    size++;
  }

  // GETTERS AND SETTERS
  //! Returns i-th state in the collector
  MsgType get_state(unsigned int i) override { return chain[i]; }
  //! Returns the whole chain in form of a deque of States
  std::deque<MsgType> get_chain() override { return chain; }
};

#endif  // BAYESMIX_COLLECTORS_MEMORY_COLLECTOR_HPP_
