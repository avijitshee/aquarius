#include "hubbard_moints.hpp"

using namespace aquarius::task;
using namespace aquarius::input;

namespace aquarius
{
namespace hubbard
{

template <typename T>
Hubbard_MOIntegrals<T>::Hubbard_MOIntegrals(const string& name, Config& config)
: Task(name, config)
{
    vector<Requirement> reqs;
    reqs += Requirement("occspace", "occ");
    reqs += Requirement("vrtspace", "vrt");
    reqs += Requirement("Ea", "Ea");
    reqs += Requirement("Eb", "Eb");
    reqs += Requirement("Fa", "Fa");
    reqs += Requirement("Fb", "Fb");
    addProduct("moints", "H", reqs);
}

INSTANTIATE_SPECIALIZATIONS(Hubbard_MOIntegrals);

}
}
