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
  bool next_state(google::protobuf::Message *out) override;

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
  void collect(const google::protobuf::Message &state) override;

  //! Returns i-th state in the collector
  void get_state(unsigned int i, google::protobuf::Message *out) override;

  //! Returns the whole chain in form of a deque of States
  // std::deque<MsgType> get_chain() override;
};

void FileCollector::open_for_reading() {
  infd = open(filename.c_str(), O_RDONLY);
  if (infd == -1) {
    std::cout << "Errno: " << strerror(errno) << std::endl;
  }
  fin = new google::protobuf::io::FileInputStream(infd);
  is_open_read = true;
}

void FileCollector::close_reading() {
  fin->Close();
  close(infd);
  is_open_read = false;
}

// \return Chain state in Protobuf-object form
bool FileCollector::next_state(google::protobuf::Message *out) {
  if (!is_open_read) {
    open_for_reading();
  }

  bool keep = google::protobuf::util::ParseDelimitedFromZeroCopyStream(
      out, fin, nullptr);
  if (!keep) {
    curr_iter = -1;
    close_reading();
  }
  return keep;
}

void FileCollector::start() {
  int outfd = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0777);
  fout = new google::protobuf::io::FileOutputStream(outfd);
  is_open_write = true;
}

void FileCollector::finish() {
  if (is_open_write) {
    fout->Close();
    close(outfd);
    is_open_write = false;
  }
}

// \param iter_state State in Protobuf-object form to write to the collector
void FileCollector::collect(const google::protobuf::Message &state) {
  bool success =
      google::protobuf::util::SerializeDelimitedToZeroCopyStream(state, fout);
  size++;
  if (!success) {
    std::cout << "Writing in FileCollector failed" << std::endl;
  }
}

// \param i Position of the requested state in the chain
void FileCollector::get_state(unsigned int i, google::protobuf::Message *out) {
  for (size_t j = 0; j < i + 1; j++) {
    get_next_state(out);
  }
  if (i < size - 1) {
    curr_iter = -1;
    close_reading();
  }
}

// // \return Chain in deque form
// template <typename MsgType>
// std::deque<MsgType> FileCollector::get_chain() {
//   open_for_reading();
//   bool keep = true;
//   std::deque<MsgType> out;
//   while (keep) {
//     MsgType msg;
//     keep = google::protobuf::util::ParseDelimitedFromZeroCopyStream(&msg,
//     fin,
//                                                                     nullptr);
//     if (keep) {
//       out.push_back(msg);
//     }
//   }
//   close_reading();
//   return out;
// }

#endif  // BAYESMIX_COLLECTORS_FILE_COLLECTOR_HPP_
