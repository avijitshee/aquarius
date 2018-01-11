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

template <typename U>
Hubbard<U>::Hubbard(const string& name, input::Config& config):Task(name, config)
{
    vector<Requirement> reqs;
    addProduct("Da", "Da", reqs);
    addProduct("Db", "Db", reqs);
    addProduct("hubbard_1eints", "H", reqs);
//    addProduct("hubbard_2eints", "HH2b", reqs);
    addProduct("hubbard_S", "S", reqs);

   nelec = config.get<int>("num_electrons");
   norb = config.get<int>("num_orbitals");
   radius = config.get<double>("radius");
   multiplicity = config.get<int>("multiplicity") ;
   int d = config.get<int>("dimension");
   nocc = nelec/2;
   assert(0 < nocc && nocc <= norb);
}

template <typename U>
bool Hubbard<U>::run(TaskDAG& dag, const Arena& arena)
{

    vector<vector<U>> E(norb,vector<U>(norb));

    int nvrt = norb-nocc;


   /*In the following we define the one-electronic AO/site integrals for Hubbard model..
    */ 

    for (int i = 0;i < norb;i++)
    {
       E[i][i] = integral_diagonal[i] ;
    }

    for (int j = 1;j < norb;j++)
    {
       E[0][j] = integral_offdiagonal[j-1] ;
    }

    for (int i = 1;i < norb;i++)
    {
       E[i][0] = integral_offdiagonal[i-1] ;
    }

    auto& H = this->put("H", new SymmetryBlockedTensor<U>("Fa", arena, PointGroup::C1(), 2, {{norb},{norb}}, {NS,NS}, true));
//  auto& Fb = this->put("Fb", new SymmetryBlockedTensor<U>("Fb", arena, PointGroup::C1(), 2, {{norb},{norb}}, {NS,NS}, true));
    auto& Da = this->put("Da", new SymmetryBlockedTensor<U>("Da", arena, PointGroup::C1(), 2, {{norb},{norb}}, {NS,NS}, true));
    auto& Db = this->put("Db", new SymmetryBlockedTensor<U>("Db", arena, PointGroup::C1(), 2, {{norb},{norb}}, {NS,NS}, true));
    auto& S  = this->put("S", new SymmetryBlockedTensor<U>("S", arena, PointGroup::C1(), 2, {{norb},{norb}}, {NS,NS}, true));

    vector<tkv_pair<U>> dpairs;
    vector<tkv_pair<U>> fpairs;
    vector<tkv_pair<U>> ov_pairs;

    for (int i = 0;i < norb;i++)
    {
        ov_pairs.emplace_back(i*norb+i, 1);
    }

    for (int i = 0;i < nocc;i++)
    {
        dpairs.emplace_back(i*norb+i, 1);
    }
    for (int i = 0;i < norb;i++)
    {
      for (int j = 0;j < norb;j++)
       {
        fpairs.emplace_back(i*norb+j, E[i][j]);
       }
    }

    if (arena.rank == 0)
    {
        Da.writeRemoteData({0,0}, dpairs);
        H.writeRemoteData({0,0}, fpairs);
        S.writeRemoteData({0,0}, ov_pairs);
    }
    else
    {
        Da.writeRemoteData({0,0});
        H.writeRemoteData({0,0});
        S.writeRemoteData({0,0}, ov_pairs);
    }

    Db = Da;
//    Fb = Fa;

    return true;
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

INSTANTIATE_SPECIALIZATIONS(aquarius::hubbard::Hubbard);
REGISTER_TASK(aquarius::hubbard::Hubbard<double>,"hubbard",spec);
