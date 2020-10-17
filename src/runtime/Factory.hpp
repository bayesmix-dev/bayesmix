#ifndef FACTORY_HPP
#define FACTORY_HPP
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

//! Generic object factory for an abstract product.

//! An object factory allows to choose one of several derived objects from a
//! single abstract base class at runtime. This type of class is implemented as
//! a singleton and stores functions that build such objects, called the
//! builders, which can be called at need at runtime, based on identifiers of
//! the specific objects. The storage must first be filled with the appropriate
//! builders, which can be as simple as a function returning a smart pointer to
//! a new instance. This can be done in a main file or in an appropriate
//! function. This factory is templatized as a variadic template, that allows
//! passing any number of parameters of any type to the contructors of the
//! objects.

//! \param AbstractProduct Class name for the abstract base object
//! \param Args...         Collection of parameters for the objects
//! constructors

template <class AbstractProduct, typename... Args>
class Factory {
 private:
  using Identifier = std::string;
  using Builder = std::function<std::unique_ptr<AbstractProduct>(Args...)>;

  //! Storage for algorithm builders
  std::map<Identifier, Builder> storage;

  // CONSTRUCTORS AND COPY OOPERATORS
  Factory() = default;
  Factory(const Factory &f) = delete;
  Factory &operator=(const Factory &f) = delete;

 public:
  //! Public destructor
  ~Factory() = default;

  //! Creates the factory via Meyer's trick

  //! \return A reference to the factory object
  static Factory &Instance() {
    static Factory factory;
    return factory;
  }

  //! Creates a specific object based on an identifier

  //! \param name Identifier for the object
  //! \param args Collection of any number of parameters
  //! \return     An unique pointer to the created object
  std::unique_ptr<AbstractProduct> create_object(const Identifier &name,
                                                 Args... args) const {
    auto f = storage.find(name);
    if (f == storage.end()) {
      throw std::invalid_argument("Error: factory identifier not found");
    } else {
      return f->second(std::forward<Args>(args)...);
    }
  }

  //! Adds a builder function to the storage

  //! \param name    Identifier to associate the builder with
  //! \param bulider Builder function for a specific object type
  void add_builder(const Identifier &name, const Builder &builder) {
    auto f = storage.insert(std::make_pair(name, builder));
    if (f.second == false) {
      std::cout << "Warning: new duplicate builder was not added to factory"
                << std::endl;
    }
  }

  //! Returns a list of identifiers of all builders in the storage
  std::vector<Identifier> list_of_known_builders() const {
    std::vector<Identifier> tmp;
    tmp.reserve(storage.size());
    for (auto i = storage.begin(); i != storage.end(); i++) {
      tmp.push_back(i->first);
    }
    return tmp;
  }

  //! Checks whether the given algorithm is already in the storage

  //! \param algo Id for the algorithm to check
  bool check_existence(const Identifier &algo) const {
    return !(storage.find(algo) == storage.end());
  }
};

#endif  // FACTORY_HPP
