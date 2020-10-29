#ifndef MEMORYCOLLECTOR_HPP
#define MEMORYCOLLECTOR_HPP

#include "BaseCollector.hpp"

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
  std::deque<bayesmix::MarginalState> chain;

  //! Reads the next state, based on the curr_iter curson
  bayesmix::MarginalState next_state() override;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~MemoryCollector() = default;
  MemoryCollector() = default;

  //! Initializes collector (here, it does nothing)
  void start() override { return; }
  //! Closes collector (here, it does nothing)
  void finish() override { return; }

  //! Writes the given state to the collector
  void collect(bayesmix::MarginalState iter_state) override;

  // GETTERS AND SETTERS
  //! Returns i-th state in the collector
  bayesmix::MarginalState get_state(unsigned int i) override {
    return chain[i];
  }
  //! Returns the whole chain in form of a deque of States
  std::deque<bayesmix::MarginalState> get_chain() override { return chain; }
};

#endif  // MEMORYCOLLECTOR_HPP
