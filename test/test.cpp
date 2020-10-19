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



// Mix classes
class Mix {
public:
  virtual ~Mix() = default;
  virtual void print_id() const = 0;
};

class DP: public Mix{
public:
  void print_id() const override {std::cout << "DP" << std::endl;}
};

class PY: public Mix{
public:
  void print_id() const override {std::cout << "PY" << std::endl;}
};



// Algo classes
class Algo {
private:
  std::shared_ptr<Hier> hier;
  std::shared_ptr<Hyp> hyp;
  std::shared_ptr<Mix> mix;
public:
  virtual ~Algo() = default;
  virtual void print_id() const = 0;
  void set_Hier(std::shared_ptr<Hier> hier_){hier = hier_;}
  void set_Hyp(std::shared_ptr<Hyp> hyp_){hyp = hyp_;}
  void set_Mix(std::shared_ptr<Mix> mix_){mix = mix_;}
  void call_Hier_id() const {hier->print_id();}
  void call_Hyp_id() const {hyp->print_id();}
  void call_Mix_id() const {mix->print_id();}
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
static void load_Mix(){
  Factory<Mix> &factory = Factory<Mix>::Instance();
  Builder<Mix> DPbuilder = [](){return std::make_shared<DP>();};
  Builder<Mix> PYbuilder = [](){return std::make_shared<PY>();};
  factory.add_builder("DP", DPbuilder);
  factory.add_builder("PY", PYbuilder);
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
// Example call: ./test N8 NNW NNWFix PY
int main(int argc, char const *argv[]){
  std::string type_Algo = argv[1];
  std::string type_Hier = argv[2];
  std::string type_Hyp  = argv[3];
  std::string type_Mix  = argv[4];
  Factory<Algo> &factory_Algo = Factory<Algo>::Instance();
  Factory<Hier> &factory_Hier = Factory<Hier>::Instance();
  Factory<Hyp> &factory_Hyp = Factory<Hyp>::Instance();
  Factory<Mix> &factory_Mix = Factory<Mix>::Instance();
  auto algo = factory_Algo.create_object(type_Algo);
  auto hier = factory_Hier.create_object(type_Hier);
  auto hyp = factory_Hyp.create_object(type_Hyp);
  auto mix = factory_Mix.create_object(type_Mix);
  (*algo).set_Hier(hier);
  (*algo).set_Hyp(hyp);
  (*algo).set_Mix(mix);
  (*algo).print_id();
  (*algo).call_Hier_id();
  (*algo).call_Hyp_id();
  (*algo).call_Mix_id();
  return 0;
}
