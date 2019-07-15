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
   alpha_elec = config.get<int>("alpha_elec");
   beta_elec = config.get<int>("beta_elec");
   norb = config.get<int>("num_orbitals");
   nalpha = getNumAlphaElectrons() ;
   nbeta = getNumBetaElectrons() ;
   assert(0 < nalpha && nalpha <= norb);
   assert(0 < nbeta && nbeta <= norb);
}

}
}

static const char* spec = R"!(

alpha_elec int,
beta_elec int,
num_orbitals int,

)!";

REGISTER_TASK(aquarius::aim::AIMTask,"aim",spec);
