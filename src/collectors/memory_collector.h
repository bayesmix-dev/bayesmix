#ifndef BAYESMIX_COLLECTORS_MEMORY_COLLECTOR_H_
#define BAYESMIX_COLLECTORS_MEMORY_COLLECTOR_H_

#include <google/protobuf/message.h>
#include <google/protobuf/util/delimited_message_util.h>

#include "base_collector.h"

//! Class for a collector that writes its content into memory.

//! An instance of MemoryCollector saves a sequence of Protobuf objects
//! by storing byte-serialized objects into a deque of strings.
//! When reading, the objects are simply deserialized from the deque.

class MemoryCollector : public BaseCollector {
 public:
  MemoryCollector() = default;
  ~MemoryCollector() = default;

  void start_collecting() override { return; }

  void finish_collecting() override { return; }

  void collect(const google::protobuf::Message& state) override;

  void reset() override;

  //! Writes the i-th state in the collector to the given message pointer
  void get_state(const unsigned int i, google::protobuf::Message* out);

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
  bool next_state(google::protobuf::Message* const out) override;

  //! Deque that contains all states in Protobuf-object form
  std::deque<std::string> chain;
};

#endif  // BAYESMIX_COLLECTORS_MEMORY_COLLECTOR_H_
