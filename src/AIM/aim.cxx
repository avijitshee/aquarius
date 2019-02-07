#include "aim.hpp"

using namespace aquarius::input;
using namespace aquarius::task;
using namespace aquarius::op;
using namespace aquarius::tensor;
using namespace aquarius::symmetry;

namespace aquarius
{
namespace aim
{

AIMTask::AIMTask(const string& name, input::Config& config)
: Task(name, config)
{
    addProduct(Product("aim", "aim"));
}

bool AIMTask::run(task::TaskDAG& dag, const Arena& arena)
{
    put("aim", new AIM<double>::AIM("aim", config));
    return true;
}

template <typename U>
AIM<U>::AIM(const string& name,input::Config& config)
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

REGISTER_TASK(aquarius::aim::AIMTask,"aim",spec);
