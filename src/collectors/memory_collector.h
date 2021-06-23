#ifndef BAYESMIX_COLLECTORS_MEMORY_COLLECTOR_H_
#define BAYESMIX_COLLECTORS_MEMORY_COLLECTOR_H_

#include <google/protobuf/message.h>
#include <google/protobuf/util/delimited_message_util.h>

#include "base_collector.h"

//! Class for a "virtual" collector which contains all objects of the chain

//! This is a type of collector which includes a deque containing all states of
//! the chain. Unlike the file collector, this is a purely "virtual" collector,
//! in the sense that it does not write chain states anywhere and all
//! information contained in it is destroyed when the main that created it is
//! terminated. It is therefore useful in situation in which writing to a file
//! is not needed, for instance in a main program that both runes and algorithm
//! and computes the estimates. In this cases, this collector is more efficient
//! than its file-writing counterpart.
//! For more information about collectors, please refer to the `BaseCollector`
//! base class.

class MemoryCollector : public BaseCollector {
 public:
  MemoryCollector() = default;
  ~MemoryCollector() = default;

  void start_collecting() override { return; }

  void finish_collecting() override { return; }

  void collect(const google::protobuf::Message& state) override;

  void reset() override;

  //! Writes the i-th state in the collector to the given message pointer
  void get_state(unsigned int i, google::protobuf::Message* out);

  //! Templatized utility for writing states directly to file
  template <typename MsgType>
  void write_to_file(const std::string& outfile) {
    // THIS is probabily a HACK: we de-serialize from the chain and
    //! re-serialize to file. Still, it's a reasonable work-around.
    int outfd = open(outfile.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0777);
    google::protobuf::io::FileOutputStream fout(outfd);

    for (std::string& serialized_state : chain) {
      //! Parse Protobuf object and get exit code
      MsgType state;
      state.ParseFromString(serialized_state);
      bool success =
          google::protobuf::util::SerializeDelimitedToZeroCopyStream(state,
                                                                     &fout);
      if (!success) {
        std::cout << "ERROR WHILE SERIALIZING TO FILE" << std::endl;
      }
    }

    fout.Close();
    close(outfd);
  }

  //! Templatized utility for reading states directly from a file
  template <typename MsgType>
  void read_from_file(const std::string& infile) {
    int infd = open(infile.c_str(), O_RDONLY);
    if (infd == -1) std::cout << "errno: " << strerror(errno) << std::endl;

    google::protobuf::io::FileInputStream fin(infd);
    bool keep = true;
    bool clean_eof = false;
    chain.resize(0);

    while (keep) {
      //! Parse Protobuf object and get exit code
      MsgType msg;
      keep = google::protobuf::util::ParseDelimitedFromZeroCopyStream(
          &msg, &fin, &clean_eof);

      if (keep) collect(msg);
    }
    fin.Close();
    close(infd);
  }

 protected:
  //! Reads the next state, based on the curr_iter curson
  bool next_state(google::protobuf::Message* out);

  //! Deque that contains all states in Protobuf-object form
  std::deque<std::string> chain;
};

#endif  // BAYESMIX_COLLECTORS_MEMORY_COLLECTOR_H_
