#ifndef BAYESMIX_MIXINGS_BASE_MIXING_H_
#define BAYESMIX_MIXINGS_BASE_MIXING_H_

#include <google/protobuf/message.h>

#include <Eigen/Dense>
#include <memory>

#include "abstract_mixing.h"

template <class Derived, typename State, typename Prior>
class BaseMixing : public AbstractMixing {
 protected:
  //!
  State state;
  //!
  std::shared_ptr<google::protobuf::Message> prior;
  //!
  const Eigen::MatrixXd *covariates_ptr;
  //!
  unsigned int num_components;

  //!
  void create_empty_prior() { prior.reset(new Prior); }
  //!
  virtual void initialize_state() = 0;
  //!
  std::shared_ptr<Prior> cast_prior() const {
    return std::dynamic_pointer_cast<Prior>(prior);
  }

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~BaseMixing() = default;
  BaseMixing() = default;
  // virtual std::shared_ptr<AbstractMixing> clone() const override {
  //   return std::make_shared<Derived>(static_cast<Derived const &>(*this));
  // }  // TODO keep it?

  // GETTERS AND SETTERS
  State get_state() const { return state; }
  unsigned int get_num_components() const override { return num_components; }
  google::protobuf::Message *get_mutable_prior() override {
    if (prior == nullptr) {
      create_empty_prior();
    }
    return prior.get();
  }
  void set_num_components(const unsigned int num_) override {
    num_components = num_;
  }
  void set_covariates(Eigen::MatrixXd *covar) override {
    covariates_ptr = covar;
  }
};

#endif  // BAYESMIX_MIXINGS_BASE_MIXING_H_
