#ifndef FACTORY_HPP
#define FACTORY_HPP
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

template <class AbstractProduct> // typename... Args>
class Factory {
 private:
  using Identifier = std::string;
  using Builder = std::function<std::shared_ptr<AbstractProduct>()>;
    // (Args...)

  std::map<Identifier, Builder> storage;

  // CONSTRUCTORS AND COPY OPERATORS
  Factory() = default;
  Factory(const Factory &f) = delete;
  Factory &operator=(const Factory &f) = delete;

 public:
  ~Factory() = default;

  static Factory &Instance() {
    static Factory factory;
    return factory;
  }

  std::shared_ptr<AbstractProduct> create_object(const Identifier &name) const{
    // Args... args
    auto f = storage.find(name);
    if (f == storage.end()) {
      std::string err = "Error: factory identifier \"" + name + "\" not found";
      throw std::invalid_argument(err);
    } else {
      return f->second();
      // return f->second(std::forward<Args>(args)...);
    }
  }

  void add_builder(const Identifier &name, const Builder &builder) {
    auto f = storage.insert(std::make_pair(name, builder));
    if (f.second == false) {
      std::cout << "Warning: new duplicate builder \"" << name
                << "\" was not added to factory" << std::endl;
    }
  }

  std::vector<Identifier> list_of_known_builders() const {
    std::vector<Identifier> tmp;
    tmp.reserve(storage.size());
    for (auto i = storage.begin(); i != storage.end(); i++) {
      tmp.push_back(i->first);
    }
    return tmp;
  }

  bool check_existence(const Identifier &algo) const {
    return !(storage.find(algo) == storage.end());
  }
};

#endif  // FACTORY_HPP
