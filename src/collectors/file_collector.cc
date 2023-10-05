#include "file_collector.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/util/delimited_message_util.h>

void FileCollector::start_collecting() {
  int outfd =
      open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, 0666);
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
  infd = open(filename.c_str(), O_RDONLY | O_BINARY, 0666);
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

bool FileCollector::next_state(google::protobuf::Message *const out) {
  if (!is_open_read) {
    set_base_msg(out);
    open_for_reading();
    populate_buffer();
  }

  curr_iter++;
  curr_buffer_pos++;

  if (curr_buffer_pos == chunk_size) {
    populate_buffer();
  }

  std::tuple<std::shared_ptr<google::protobuf::Message>, bool> msg_tuple =
      msg_buffer[curr_buffer_pos].get();

  out->CopyFrom(*std::get<0>(msg_tuple));
  bool keep = std::get<1>(msg_tuple);

  if (!keep) {
    curr_iter = 0;
    curr_buffer_pos = 0;
    close_reading();
  }

  return keep;
}

void FileCollector::populate_buffer() {
  msg_buffer = std::vector<std::future<
      std::tuple<std::shared_ptr<google::protobuf::Message>, bool>>>();
  curr_buffer_pos = 0;
  for (int i = 0; i < chunk_size; i++) {
    std::future<std::tuple<std::shared_ptr<google::protobuf::Message>, bool>>
        future_msg =
            std::async(std::launch::deferred, &FileCollector::read_one, this);

    msg_buffer.push_back(std::move(future_msg));
  }
}

std::tuple<std::shared_ptr<google::protobuf::Message>, bool>
FileCollector::read_one() {
  std::shared_ptr<google::protobuf::Message> msg(base_msg->New());
  bool keep = google::protobuf::util::ParseDelimitedFromZeroCopyStream(
      msg.get(), fin, nullptr);
  return std::make_tuple(msg, keep);
}
