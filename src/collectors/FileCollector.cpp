#include "FileCollector.hpp"

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
bayesmix::MarginalState FileCollector::next_state() {
  if (!is_open_read) {
    open_for_reading();
  }

  bayesmix::MarginalState out;
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
void FileCollector::collect(bayesmix::MarginalState iter_state) {
  bool success = google::protobuf::util::SerializeDelimitedToZeroCopyStream(
      iter_state, fout);
  size++;
  if (!success) {
    std::cout << "Writing in FileCollector failed" << std::endl;
  }
}

// \param i Position of the requested state in the chain
// \return  Chain state in Protobuf-object form
bayesmix::MarginalState FileCollector::get_state(unsigned int i) {
  bayesmix::MarginalState state;
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
std::deque<bayesmix::MarginalState> FileCollector::get_chain() {
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
