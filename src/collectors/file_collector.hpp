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

template <typename MsgType>
class FileCollector : public BaseCollector<MsgType> {
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
  MsgType next_state() override;

  using BaseCollector<MsgType>::size;
  using BaseCollector<MsgType>::curr_iter;
  using BaseCollector<MsgType>::get_next_state;

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
  void collect(MsgType iter_state) override;
  //! Returns i-th state in the collector
  MsgType get_state(unsigned int i) override;
  //! Returns the whole chain in form of a deque of States
  std::deque<MsgType> get_chain() override;
};

template <typename MsgType>
void FileCollector<MsgType>::open_for_reading() {
  infd = open(filename.c_str(), O_RDONLY);
  if (infd == -1) {
    std::cout << "Errno: " << strerror(errno) << std::endl;
  }
  fin = new google::protobuf::io::FileInputStream(infd);
  is_open_read = true;
}

template <typename MsgType>
void FileCollector<MsgType>::close_reading() {
  fin->Close();
  close(infd);
  is_open_read = false;
}

// \return Chain state in Protobuf-object form
template <typename MsgType>
MsgType FileCollector<MsgType>::next_state() {
  if (!is_open_read) {
    open_for_reading();
  }

  MsgType out;
  bool keep = google::protobuf::util::ParseDelimitedFromZeroCopyStream(
      &out, fin, nullptr);
  if (!keep) {
    std::out_of_range("Error: surpassed EOF in FileCollector");
  }
  if (curr_iter == size - 1) {
    curr_iter = -1;
    close_reading();
  }
  return out;
}

template <typename MsgType>
void FileCollector<MsgType>::start() {
  int outfd = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0777);
  fout = new google::protobuf::io::FileOutputStream(outfd);
  is_open_write = true;
}

template <typename MsgType>
void FileCollector<MsgType>::finish() {
  if (is_open_write) {
    fout->Close();
    close(outfd);
    is_open_write = false;
  }
}

// \param iter_state State in Protobuf-object form to write to the collector
template <typename MsgType>
void FileCollector<MsgType>::collect(MsgType iter_state) {
  bool success = google::protobuf::util::SerializeDelimitedToZeroCopyStream(
      iter_state, fout);
  size++;
  if (!success) {
    std::cout << "Writing in FileCollector failed" << std::endl;
  }
}

// \param i Position of the requested state in the chain
// \return  Chain state in Protobuf-object form
template <typename MsgType>
MsgType FileCollector<MsgType>::get_state(unsigned int i) {
  MsgType state;
  for (size_t j = 0; j < i + 1; j++) {
    state = get_next_state();
  }
  if (i < size - 1) {
    curr_iter = -1;
    close_reading();
  }
  return state;
}

// \return Chain in deque form
template <typename MsgType>
std::deque<MsgType> FileCollector<MsgType>::get_chain() {
  open_for_reading();
  bool keep = true;
  std::deque<bayesmix::MarginalState> out;
  while (keep) {
    bayesmix::MarginalState msg;
    keep = google::protobuf::util::ParseDelimitedFromZeroCopyStream(&msg, fin,
                                                                    nullptr);
    if (keep) {
      out.push_back(msg);
    }
  }
  close_reading();
  return out;
}

#endif  // BAYESMIX_COLLECTORS_FILE_COLLECTOR_HPP_
