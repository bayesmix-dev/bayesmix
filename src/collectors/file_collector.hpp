#ifndef BAYESMIX_COLLECTORS_FILE_COLLECTOR_HPP_
#define BAYESMIX_COLLECTORS_FILE_COLLECTOR_HPP_

#include "base_collector.hpp"

//! Class for a collector that writes its content to a file.

//! This is a type of collector that writes each state passed to it to a binary
//! file. Unlike the memory collector, the contents of this collector are
//! permanent, because every state collected by it remains ever after the
//! termination of the main that created it, in Protobuf form, in the
//! corresponding file. This approach is mandatory, for instance, if different
//! main programs are used both to run the algorithm and the estimates.
//! Therefore, a file collector has both a reading and a writing mode.

class FileCollector : public BaseCollector {
 protected:
  //! Unix file descriptor for reading mode
  int infd;
  //! Unix file descriptor for writing mode
  int outfd;
  //! Pointer to a reading file stream
  google::protobuf::io::FileInputStream *fin;
  //! Pointer to a writing file stream
  google::protobuf::io::FileOutputStream *fout;
  //! Name of file from which read/write
  std::string filename;
  //! Flag that indicates if the collector is open in read-mode
  bool is_open_read = false;
  //! Flag that indicates if the collector is open in write-mode
  bool is_open_write;

  //! Opens collector in reading mode
  void open_for_reading();
  //! Terminates reading mode for the collector
  void close_reading();
  //! Reads the next state, based on the curr_iter curson
  bayesmix::MarginalState next_state() override;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~FileCollector() {
    if (is_open_write) {
      fout->Close();
      close(outfd);
    }
    if (is_open_read) {
      fin->Close();
      close(infd);
    }
  }
  FileCollector(const std::string &filename_) : filename(filename_) {}
  //! Initializes collector
  void start() override;
  //! Closes collector
  void finish() override;

  //! Writes the given state to the collector
  void collect(bayesmix::MarginalState iter_state) override;
  //! Returns i-th state in the collector
  bayesmix::MarginalState get_state(unsigned int i) override;
  //! Returns the whole chain in form of a deque of States
  std::deque<bayesmix::MarginalState> get_chain() override;
};

#endif  // BAYESMIX_COLLECTORS_FILE_COLLECTOR_HPP_
