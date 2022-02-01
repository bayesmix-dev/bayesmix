#ifndef BAYESMIX_COLLECTORS_BASE_COLLECTOR_H_
#define BAYESMIX_COLLECTORS_BASE_COLLECTOR_H_

#include <fcntl.h>
#include <google/protobuf/message.h>
#include <stdio.h>
#include <unistd.h>

#include <deque>
#include <fstream>
#include <string>
#include <vector>

#include "algorithm_state.pb.h"

//! Abstract base class for a collector that contains a chain in Protobuf form

//! This is an abstract base class for a structure called data collector, or
//! collector for short.
//! A collector is used to store a sequence of Google Protobuf's objects, also
//! known as messages, by serializing them. Data can be retrieved and
//! de-serialized into a Protobuf object, either in C++ or in other programming
//! languages.
//! In particular, within this library, collectors are used to save the states
//! of the Markov chain generated by a Gibbs sampling algorithm at each
//! iteration. This includes allocations and unique values vectors, as well as
//! other relevant or convenient values: the clusters' cardinality, the mixing
//! state, and the iteration number. The skeleton corresponding to this
//! `AlgorithmState` message is described in the proto/algorithm_state.proto
//! file.
//! A collector is needed since it allows communication of the stored
//! information among different scripts, which otherwise would be impossible.
//! Also, one may want to save the whole Markov chain in order to perform
//! subsequent, separate analysis on it, so having access to detailed
//! information about the MCMC run may prove extremely useful.
//! This class spawns two inherited classes: the `FileCollector`, which stores
//! states in the computer memory, and the `MemoryCollector`, which writes
//! states to a binary file. Please refer to their respective files for more
//! information about them.

class BaseCollector {
 public:
  BaseCollector() = default;
  virtual ~BaseCollector() = default;

  //! Initializes collector
  virtual void start_collecting() = 0;

  //! Closes collector
  virtual void finish_collecting() = 0;

  //! Reads the next state and deserializes it into the pointer `out`.
  bool get_next_state(google::protobuf::Message *out) {
    return next_state(out);
  }

  std::vector<std::shared_ptr<google::protobuf::Message>> get_chunk(
      int size, bool &keep, google::protobuf::Message *base_msg) {
    std::vector<std::shared_ptr<google::protobuf::Message>> out;
    for (int i = 0; i < size; i++) {
      std::shared_ptr<google::protobuf::Message> msg(base_msg->New());
      keep = get_next_state(msg.get());
      if (!keep) {
        break;
      }
      out.push_back(msg);
    }
    return out;
  }

  //! Writes the given state to the collector
  virtual void collect(const google::protobuf::Message &state) = 0;

  //! Resets the collector to the beginning of the chain
  virtual void reset() = 0;

  //! Returns the number of stored states
  unsigned int get_size() const { return size; }

 protected:
  //! Reads the state, based on the curr_iter cursor, and returns exit code
  virtual bool next_state(google::protobuf::Message *out) = 0;

  //! Current size of the chain
  unsigned int size = 0;

  //! Read cursor that indicates the current iteration
  unsigned int curr_iter = 0;
};

#endif  // BAYESMIX_COLLECTORS_BASE_COLLECTOR_H_
