#ifndef BAYESMIX_RUNTIME_FACTORY_H_
#define BAYESMIX_RUNTIME_FACTORY_H_

#include <google/protobuf/generated_enum_reflection.h>

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

//! Generic object factory for an abstract product.

//! An object factory allows to choose at runtime one of several derived
//! objects from a single abstract base class. A factory is implemented as a
//! singleton, and stores simple functions that can build such objects by
//! returning a smart pointer to a new instance of them. These functions are
//! also known as builders, and can be called at need at runtime. Each object
//! type has a unique identifier, which is provided by the user to the factory
//! in order to select the builder for the desider corresponding object.
//! This factory is templatized, so that the three major object types of this
//! library, which are `Algorithms`, `Hierarchies`, and `Mixings`, can each
//! have their own, independent factory.
//! The factory's storage must first be filled with the appropriate builders.
//! In this library, this happens via load files for each of the three classes,
//! which are called automagically in the main run file.
//! The Identifier template parameter must be a Protobuf `enum` type, since
//! this library uses the Protocol Buffers package to shift between IDs and
//! strings. These `enum` types are described in the proto/algorithm_id.proto,
//! hierarchy_id.proto, and mixing_id.proto files.

//! @tparam Identifier       Protobuf enum type for the indentifier
//! @tparam AbstractProduct  Class name for the abstract base object

template <typename Identifier, class AbstractProduct>
class Factory {
 public:
  ~Factory() = default;

  //! Returns (and creates if nonexistent) the singleton of this class
  static Factory &Instance() {
    static Factory factory;
    return factory;
  }

  //! Creates a new instance of a specific object based on its identifier
  //! @param id Identifier for the object
  //! @return   A shared pointer to the created object
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

  //! Creates a new instance of a specific object based on its name string
  //! @param name string identifier for the object
  //! @return     A shared pointer to the created object
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
  //! @param id      Identifier to associate the builder with
  //! @param bulider Builder function for a specific object type
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

  //! Check whether the given object ID is already in the storage
  bool check_existence(const Identifier &id) const {
    return !(storage.find(id) == storage.end());
  }

private:
  using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

  Factory() = default;
  Factory(const Factory &f) = delete;
  Factory &operator=(const Factory &f) = delete;

  //! Converts a Protobuf ID into the name of the class in string form
  static std::string id_to_name(const Identifier &id) {
    return google::protobuf::internal::NameOfEnum(
        google::protobuf::GetEnumDescriptor<Identifier>(), id);
  }

  //! Converts the name of the class in string form into a Protobuf ID
  static Identifier name_to_id(const std::string &name) {
    Identifier id;
    google::protobuf::internal::ParseNamedEnum<Identifier>(
        google::protobuf::GetEnumDescriptor<Identifier>(), name, &id);
    return id;
  }

  //! Storage for algorithm builders
  std::map<Identifier, Builder> storage;
};

#endif  // BAYESMIX_RUNTIME_FACTORY_H_
