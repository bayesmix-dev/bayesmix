#ifndef BAYESMIX_COLLECTORS_FILE_COLLECTOR_H_
#define BAYESMIX_COLLECTORS_FILE_COLLECTOR_H_

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>

#include <future>

#include "base_collector.h"

//! Class for a collector that writes (and reads) its content to a file.

//! An instance of FileCollector saves a sequence of Protobuf objects to a file
//! and is able to read them back, returning them one by one. When writing to
//! the file, the objects are simply serialized into bytes. When reading, for
//! efficiency's sake, we instead read a chunk of 'chunk_size' objects and
//! deserialized them into a buffer, asynchronously. When the buffer has been
//! read, we erase it and fill it again with the next chunk of objects.

class FileCollector : public BaseCollector {
 public:
  FileCollector(const std::string &filename_, const int chunk_size = 100)
      : filename(filename_), chunk_size(chunk_size) {}

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

  void start_collecting() override;

  void finish_collecting() override;

  void collect(const google::protobuf::Message &state) override;

  void reset() override;

 protected:
  //! Opens collector in reading mode
  void open_for_reading();

  //! Terminates reading mode for the collector
  void close_reading();

  bool next_state(google::protobuf::Message *const out) override;

  //! Populates the buffer with the next chunk of objects.
  void populate_buffer(google::protobuf::Message *const base_msg);

  //! Unix file descriptor for reading mode
  int infd;

  //! Unix file descriptor for writing mode
  int outfd;

  int chunk_size;

  int curr_buffer_pos;

  //! Buffer of std::future objects, one per message. std::future is needed
  //! to perform the reading asynchronously.
  std::vector<std::future<
      std::tuple<std::shared_ptr<google::protobuf::Message>, bool>>>
      msg_buffer;

  //! Reads one message from the file and returns a tuple containing (a shared
  //! ptr to) the message and a bool indicating whether the message was
  //! successfully read.
  std::tuple<std::shared_ptr<google::protobuf::Message>, bool> read_one(
      google::protobuf::Message *const base_msg);

  //! Pointer to a reading file stream
  google::protobuf::io::FileInputStream *fin;

  //! Pointer to a writing file stream
  google::protobuf::io::FileOutputStream *fout;

  //! Name of file from which read/write
  std::string filename;

  //! Flag that indicates if the collector is open in read-mode
  bool is_open_read = false;

  //! Flag that indicates if the collector is open in write-mode
  bool is_open_write = false;
};

#endif  // BAYESMIX_COLLECTORS_FILE_COLLECTOR_H_
