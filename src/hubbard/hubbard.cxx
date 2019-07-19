#include "hubbard.hpp"

using namespace aquarius::input;
using namespace aquarius::task;
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
    put("hubbard", new Hubbard("hubbard", config));
    return true;
}

Hubbard::Hubbard(const string& name,input::Config& config)
{
   nelec = config.get<int>("num_electrons");
   norb = config.get<int>("num_orbitals");
   ndoc = config.get<int>("doubly_occupied");
   multiplicity = config.get<int>("multiplicity") ;
   openshell_alpha = config.get<string>("openshell_alpha") ;
   openshell_beta = config.get<string>("openshell_beta") ;
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
dimension? int 1,
multiplicity?
    int 1,
doubly_occupied int,
openshell_alpha string, 
openshell_beta string 

)!";

//INSTANTIATE_SPECIALIZATIONS(aquarius::hubbard::Hubbard);
REGISTER_TASK(aquarius::hubbard::HubbardTask,"hubbard",spec);
