#include "read_integrals.hpp"

using namespace aquarius::input;
using namespace aquarius::task;
using namespace aquarius::op;
using namespace aquarius::tensor;
using namespace aquarius::symmetry;

namespace aquarius
{
namespace aim
{

template <typename U>
ReadInts<U>::ReadInts(const string& name, input::Config& config)
: Task(name, config)
{
    vector<Requirement> reqs;
    reqs.emplace_back("aim","aim"); 
    addProduct("Da", "Da", reqs);
    addProduct("Db", "Db", reqs);
    addProduct("aim_alphaints", "Ha", reqs);
    addProduct("aim_betaints", "Hb", reqs);
    addProduct("aim_S", "S", reqs);
}

template <typename U>
bool ReadInts<U>::run(task::TaskDAG& dag, const Arena& arena)
{
    auto& aim =this->template get<AIM>("aim");

    vector<int> alpha_array ;
    vector<int> beta_array ;
    int norb = aim.getNumOrbitals() ;
    int nalpha = aim.getNumAlphaElectrons() ;
    int nbeta = aim.getNumBetaElectrons() ;
    int ndoc = aim.getDoccOrbitals() ;

    vector<vector<double>> Ea(norb,vector<double>(norb));
    vector<vector<double>> Eb(norb,vector<double>(norb));
    vector<vector<double>> Dalpha(norb,vector<double>(norb));
    vector<vector<double>> Dbeta(norb,vector<double>(norb));

    if (arena.rank == 0)
    {
     read_1e_integrals(norb) ; 
    } else

    {
           int_a.resize(norb*norb) ;
           int_b.resize(norb*norb) ;
     } 

    read_coeff() ;	
   if (coeff_exists) {
    for (int i = 0;i < norb;i++){
     for (int j = 0;j < norb;j++){
       Dalpha[i][j] = mo_coeff[i*norb+j] ;
//     for (int k = 0;k < nalpha;k++)
//     {
//       Dalpha[i][j] += mo_coeff[k*norb+i]*mo_coeff[k*norb+j] ;
//     }
     }
    }

    for (int i = 0;i < norb;i++)
    {
     for (int j = 0;j < norb;j++)
     {
        Dbeta[i][j] = mo_coeff[i*norb+j] ;
//      for (int k = 0;k < nbeta;k++)
//      {
//        Dbeta[i][j] += mo_coeff[k*norb+i]*mo_coeff[k*norb+j] ;
//      }
     }
    }
   }

    for (int i = 0;i < norb;i++){
     for (int j = 0;j < norb;j++){
     
       Ea[i][j] = int_a[i*norb+j] ;
       Eb[i][j] = int_b[i*norb+j] ;

     }
    }

    auto& Ha =  this->put("Ha", new SymmetryBlockedTensor<U>("Fa", arena, PointGroup::C1(), 2, {{norb},{norb}}, {NS,NS}, true));
    auto& Hb =  this->put("Hb", new SymmetryBlockedTensor<U>("Fb", arena, PointGroup::C1(), 2, {{norb},{norb}}, {NS,NS}, true));
    auto& Da = this->put("Da", new SymmetryBlockedTensor<U>("Da", arena, PointGroup::C1(), 2, {{norb},{norb}}, {NS,NS}, true));
    auto& Db = this->put("Db", new SymmetryBlockedTensor<U>("Db", arena, PointGroup::C1(), 2, {{norb},{norb}}, {NS,NS}, true));
    auto& S  = this->put("S", new SymmetryBlockedTensor<U>("S", arena, PointGroup::C1(), 2, {{norb},{norb}}, {NS,NS}, true));

    vector<tkv_pair<U>> dapairs;
    vector<tkv_pair<U>> dbpairs;
    vector<tkv_pair<U>> fapairs;
    vector<tkv_pair<U>> fbpairs;
    vector<tkv_pair<U>> ov_pairs;

    for (int i = 0;i < norb;i++)
    {
        ov_pairs.emplace_back(i*norb+i, 1);
    }


   if (coeff_exists) 
   {
    for (int i = 0;i < norb;i++)
    {
       for (int j = 0;j < norb;j++)
       {
        dapairs.emplace_back(i*norb+j, Dalpha[i][j]);
       }
    }

   for (int i = 0;i < norb;i++){
       for (int j = 0;j < norb;j++)
       {
        dbpairs.emplace_back(i*norb+j, Dbeta[i][j]);
       }
    }
   }
   else{
    for (int i = 0;i < nalpha;i++){
        dapairs.emplace_back(i*norb+i,1) ;
    }

    for (int i = 0;i < nbeta;i++){
        dbpairs.emplace_back(i*norb+i,1) ;
    }

   }

    for (int i = 0;i < norb;i++){
       for (int j = 0;j < norb;j++){
        fapairs.emplace_back(i*norb+j, Ea[i][j]);
        fbpairs.emplace_back(i*norb+j, Eb[i][j]);
       }
    }

    if (arena.rank == 0)
    {
        Da.writeRemoteData({0,0}, dapairs);
        Db.writeRemoteData({0,0}, dbpairs);
        Ha.writeRemoteData({0,0}, fapairs);
        Hb.writeRemoteData({0,0}, fbpairs);
        S.writeRemoteData({0,0}, ov_pairs);
    }
    else
    {
        Da.writeRemoteData({0,0});
        Db.writeRemoteData({0,0});
        Ha.writeRemoteData({0,0});
        Hb.writeRemoteData({0,0});
        S.writeRemoteData({0,0}, ov_pairs);
    }
    return true;
}

}
}

INSTANTIATE_SPECIALIZATIONS(aquarius::aim::ReadInts);
REGISTER_TASK(aquarius::aim::ReadInts<double>,"read_aim_integrals");
