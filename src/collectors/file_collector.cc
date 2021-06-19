#include "file_collector.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/util/delimited_message_util.h>

void FileCollector::start_collecting() {
  int outfd = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0777);
  fout = new google::protobuf::io::FileOutputStream(outfd);
  is_open_write = true;
}

void FileCollector::finish_collecting() {
  if (is_open_write) {
    fout->Close();
    close(outfd);
    is_open_write = false;
  }
}

void FileCollector::collect(const google::protobuf::Message &state) {
  //! Parse Protobuf object and get exit code
  bool success =
      google::protobuf::util::SerializeDelimitedToZeroCopyStream(state, fout);
  size++;
  if (!success) {
    std::cout << "Writing in FileCollector failed" << std::endl;
  }
}

void FileCollector::reset() {
  curr_iter = 0;
  close_reading();
}

void FileCollector::open_for_reading() {
  infd = open(filename.c_str(), O_RDWR);
  if (infd == -1) {
    std::cout << "Errno: " << strerror(errno) << std::endl;
  }
  fin = new google::protobuf::io::FileInputStream(infd);
  is_open_read = true;
}

void FileCollector::close_reading() {
  if (is_open_read) {
    fin->Close();
    close(infd);
    is_open_read = false;
  }
}

bool FileCollector::next_state(google::protobuf::Message *out) {
  if (!is_open_read) {
    open_for_reading();
  }
  curr_iter++;
  //! Parse Protobuf object and get exit code
  bool keep = google::protobuf::util::ParseDelimitedFromZeroCopyStream(
      out, fin, nullptr);
  if (!keep) {
    curr_iter = 0;
    close_reading();
  }
  return keep;
}
