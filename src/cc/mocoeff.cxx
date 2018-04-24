#include "mocoeff.hpp"
#include "hubbard/uhf_modelH.hpp"

using namespace aquarius::task;
using namespace aquarius::input;
using namespace aquarius::hubbard;

namespace aquarius
{
namespace cc
{

template <typename T>
MOCoeffs<T>::MOCoeffs(const string& name, Config& config)
: Task(name, config)
{
    vector<Requirement> reqs;
    reqs.emplace_back("occspace", "occ");
    reqs.emplace_back("vrtspace", "vrt");
    reqs.emplace_back("Ea", "Ea");
    reqs.emplace_back("Eb", "Eb");
}

INSTANTIATE_SPECIALIZATIONS(MOCoeffs);

}
}
