#ifndef BAYESMIX_COLLECTORS_BASE_COLLECTOR_HPP_
#define BAYESMIX_COLLECTORS_BASE_COLLECTOR_HPP_

#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <stdio.h>
#include <unistd.h>

#include <deque>
#include <fstream>
#include <string>
#include <vector>

#include "../../proto/cpp/marginal_state.pb.h"

//! Abstract base class for a collector that contains a chain in Protobuf form

//! This is an abstract base class for a structure called data collector, or
//! collector for short. A collector is meant to store the state of a Markov
//! chain at all iterations, composed of the allocations and the unique values
//! vectors. These values are stored in classes built via the Google Protocol
//! Buffers library, also known as Protobuf. In particular, t the end of each
//! iteration of a BNP algorithm of this library, its state is saved to the
//! collector. This means that the collector will contain the states of the
//! whole Markov chain by the end of the running of the algorithm.

class BaseCollector {
 protected:
  //! Current size of the chain
  unsigned int size = 0;
  //! Read cursor that indicates the current iteration
  unsigned int curr_iter = -1;

  //! Reads the next state, based on the curr_iter curson
  virtual bool next_state(google::protobuf::Message *out) = 0;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  virtual ~BaseCollector() = default;
  BaseCollector() = default;

  //! Initializes collector
  virtual void start() = 0;
  //! Closes collector
  virtual void finish() = 0;

  //! Reads the next state and advances the cursor by 1
  bool get_next_state(google::protobuf::Message *out) {
    return next_state(out);
  }

  //! Writes the given state to the collector
  virtual void collect(const google::protobuf::Message& state) = 0;

  // GETTERS AND SETTERS
  //! Returns i-th state in the collector
  virtual void get_state(unsigned int i, google::protobuf::Message *out) = 0;
  
  // //! Returns the whole chain in form of a deque of States
  // virtual std::deque<MsgType> get_chain() = 0;

  unsigned int get_size() const { return size; }
};

#endif  // BAYESMIX_COLLECTORS_BASE_COLLECTOR_HPP_
