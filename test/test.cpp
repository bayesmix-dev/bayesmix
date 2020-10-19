#include<iostream>
#include "Factory.hpp"

// Hyp classes
class Hyp {
public:
  virtual ~Hyp() = default;
  virtual void print_id() const = 0;
};

class NNIGFix : public Hyp {
public:
  void print_id() const override {std::cout << "NNIGFix" << std::endl;}
};

class NNWFix : public Hyp {
public:
  void print_id() const override {std::cout << "NNWFix" << std::endl;}
};



// Hier classes
class Hier {
public:
  virtual ~Hier() = default;
  virtual void print_id() const = 0;
};

class NNIG: public Hier{
public:
  void print_id() const override {std::cout << "NNIG" << std::endl;}
};

class NNW: public Hier{
public:
  void print_id() const override {std::cout << "NNW" << std::endl;}
};



// Algo classes
class Algo {
private:
  std::shared_ptr<Hier> hier;
  std::shared_ptr<Hyp> hyp;
public:
  virtual ~Algo() = default;
  virtual void print_id() const = 0;
  void set_Hier(std::shared_ptr<Hier> hier_){hier = hier_;}
  void set_Hyp(std::shared_ptr<Hyp> hyp_){hyp = hyp_;}
  void call_Hier_id() const {hier->print_id();}
  void call_Hyp_id() const {hyp->print_id();}
};

class N2: public Algo{
public:
  void print_id() const override {std::cout << "N2" << std::endl;}
};

class N8: public Algo{
public:
  void print_id() const override {std::cout << "N8" << std::endl;}
};



// Factory stuff
template<class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

__attribute__((constructor))
static void load_Hyp(){
  Factory<Hyp> &factory = Factory<Hyp>::Instance();
  Builder<Hyp> NNIGFixbuilder = [](){return std::make_shared<NNIGFix>();};
  Builder<Hyp> NNWFixbuilder  = [](){return std::make_shared<NNWFix >();};
  factory.add_builder("NNIGFix", NNIGFixbuilder);
  factory.add_builder("NNWFix" , NNWFixbuilder );
}

__attribute__((constructor))
static void load_Hier(){
  Factory<Hier> &factory = Factory<Hier>::Instance();
  Builder<Hier> NNIGbuilder = [](){return std::make_shared<NNIG>();};
  Builder<Hier> NNWbuilder  = [](){return std::make_shared<NNW >();};
  factory.add_builder("NNIG", NNIGbuilder);
  factory.add_builder("NNW" , NNWbuilder );
}

__attribute__((constructor))
static void load_Algo(){
  Factory<Algo> &factory = Factory<Algo>::Instance();
  Builder<Algo> N2builder = [](){return std::make_shared<N2>();};
  Builder<Algo> N8builder = [](){return std::make_shared<N8>();};
  factory.add_builder("N2", N2builder);
  factory.add_builder("N8", N8builder);
}



// Main
int main(int argc, char const *argv[]){
  std::string type_Algo = argv[1];
  std::string type_Hier = argv[2];
  std::string type_Hyp  = argv[3];
  Factory<Algo> &factory_Algo = Factory<Algo>::Instance();
  Factory<Hier> &factory_Hier = Factory<Hier>::Instance();
  Factory<Hyp>  &factory_Hyp  = Factory<Hyp >::Instance();
  auto algo = factory_Algo.create_object(type_Algo);
  auto hier = factory_Hier.create_object(type_Hier);
  auto hyp  = factory_Hyp.create_object(type_Hyp);
  (*algo).set_Hier(hier);
  (*algo).set_Hyp(hyp);
  (*algo).print_id();
  (*hier).print_id();
  (*hyp ).print_id();
  (*algo).call_Hier_id();
  (*algo).call_Hyp_id();
  return 0;
}
