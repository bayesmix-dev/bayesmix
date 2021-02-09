#ifndef BAYESMIX_RUNTIME_FACTORY_H_
#define BAYESMIX_RUNTIME_FACTORY_H_

#include <google/protobuf/generated_enum_reflection.h>

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
//! function.
//! The Identifier template parameter must be a protobuf 'enum' type
//! since we use protobuf to pass between ids and strings

//! \param Identifier Protobuf enum type for the indentifier
//! \param AbstractProduct Class name for the abstract base object

template <typename Identifier, class AbstractProduct>
class Factory {
 private:
  using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

  //! Storage for algorithm builders
  std::map<Identifier, Builder> storage;

  // CONSTRUCTORS AND COPY OOPERATORS
  Factory() = default;
  Factory(const Factory &f) = delete;
  Factory &operator=(const Factory &f) = delete;

  static std::string id_to_name(const Identifier &id) {
    return google::protobuf::internal::NameOfEnum(
        google::protobuf::GetEnumDescriptor<Identifier>(), id);
  }

  static Identifier name_to_id(const std::string &name) {
    Identifier id;
    google::protobuf::internal::ParseNamedEnum<Identifier>(
        google::protobuf::GetEnumDescriptor<Identifier>(), name, &id);
    return id;
  }

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

  //! \param id Identifier for the object
  //! \return     An shared pointer to the created object
  std::shared_ptr<AbstractProduct> create_object(const Identifier &id) const {
    auto f = storage.find(id);
    if (f == storage.end()) {
      std::string err =
          "Factory identifier \"" + id_to_name(id) + "\" not found";
      throw std::invalid_argument(err);
    } else {
      return f->second();
      // return f->second(std::forward<Args>(args)...);
    }
  }

  //! \param name string identifier for the object
  //! \return     An shared pointer to the created object
  std::shared_ptr<AbstractProduct> create_object(
      const std::string &name) const {
    try {
      return create_object(name_to_id(name));
    } catch (const std::invalid_argument &e) {
      std::string err =
          "No identifier found for name: \"" + name + "\". \n" + e.what();
      throw std::invalid_argument(err);
    }
  }

  //! Adds a builder function to the storage

  //! \param id    Identifier to associate the builder with
  //! \param bulider Builder function for a specific object type
  void add_builder(const Identifier &id, const Builder &builder) {
    auto f = storage.insert(std::make_pair(id, builder));
    if (f.second == false) {
      std::cout << "Warning: new duplicate builder \"" << id
                << "\" was not added to factory" << std::endl;
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

  //! \param id Id for the object to check
  bool check_existence(const Identifier &id) const {
    return !(storage.find(id) == storage.end());
  }
};

#endif  // BAYESMIX_RUNTIME_FACTORY_H_
