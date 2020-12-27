void FileCollector::open_for_reading() {
  infd = open(filename.c_str(), O_RDWR);
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
  curr_iter++;
  std::cout << "FileCollector::next_state, curr_iter: " << curr_iter
            << std::endl;

  bool keep = google::protobuf::util::ParseDelimitedFromZeroCopyStream(
      out, fin, nullptr);
  std::cout << "keep: " << keep << std::endl;
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