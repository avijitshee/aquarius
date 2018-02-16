#include "hubbard.hpp"

using namespace aquarius::input;
using namespace aquarius::task;
using namespace aquarius::op;
using namespace aquarius::tensor;
using namespace aquarius::symmetry;

namespace aquarius
{
namespace hubbard
{

HubbardTask::HubbardTask(const string& name, input::Config& config)
: Task(name, config)
{
    addProduct(Product("hubbard", "hubbard"));
}

bool HubbardTask::run(task::TaskDAG& dag, const Arena& arena)
{
    put("hubbard", new Hubbard<double>::Hubbard("hubbard", config));
}

template <typename U>
Hubbard<U>::Hubbard(const string& name,input::Config& config)
{
   nelec = config.get<int>("num_electrons");
   norb = config.get<int>("num_orbitals");
   radius = config.get<double>("radius");
   multiplicity = config.get<int>("multiplicity") ;
   dimension = config.get<int>("dimension");
   nalpha = getNumAlphaElectrons() ;
   nbeta = getNumBetaElectrons() ;
   assert(0 < nalpha && nalpha <= norb);
   assert(0 < nbeta && nbeta <= norb);
}

}
}

static const char* spec = R"!(

num_electrons int,
num_orbitals int,
radius double,
dimension? int 1,
multiplicity?
    int 1

)!";

//INSTANTIATE_SPECIALIZATIONS(aquarius::hubbard::Hubbard);
REGISTER_TASK(aquarius::hubbard::HubbardTask,"hubbard",spec);
