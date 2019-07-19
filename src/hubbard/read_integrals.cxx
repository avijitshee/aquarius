#include "read_integrals.hpp"

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
ReadInts<U>::ReadInts(const string& name, input::Config& config)
: Task(name, config)
{
    vector<Requirement> reqs;
    reqs.emplace_back("hubbard","hubbard"); 
    addProduct("Da", "Da", reqs);
    addProduct("Db", "Db", reqs);
    addProduct("hubbard_1eints", "H", reqs);
    addProduct("hubbard_S", "S", reqs);
}

template <typename U>
bool ReadInts<U>::run(task::TaskDAG& dag, const Arena& arena)
{
    auto& hubbard =this->template get<Hubbard>("hubbard");

    vector<int> alpha_array ;
    vector<int> beta_array ;
    int norb = hubbard.getNumOrbitals() ;
    int nalpha = hubbard.getNumAlphaElectrons() ;
    int nbeta = hubbard.getNumBetaElectrons() ;
    int ndoc = hubbard.getDoccOrbitals() ;

    vector<vector<double>> E(norb,vector<double>(norb));
    vector<vector<double>> Dalpha(norb,vector<double>(norb));
    vector<vector<double>> Dbeta(norb,vector<double>(norb));

    read_1e_integrals() ; 
    read_2e_integrals() ; 
    read_coeff() ;	


   if (coeff_exists) 
   {
    for (int i = 0;i < norb;i++)
    {
     for (int j = 0;j < norb;j++)
     {
       for (int k = 0;k < nalpha;k++)
       {
         Dalpha[i][j] += mo_coeff[k*norb+i]*mo_coeff[k*norb+j] ;
       }
     }
    }

    for (int i = 0;i < norb;i++)
    {
     for (int j = 0;j < norb;j++)
     {
        for (int k = 0;k < nbeta;k++)
        {
          Dbeta[i][j] += mo_coeff[k*norb+i]*mo_coeff[k*norb+j] ;
        }
     }
    }
   }

    for (int i = 0;i < norb;i++)
    {
     for (int j = 0;j < norb;j++)
     {
       E[i][j] = integral_diagonal[i*norb+j] ;
     }
    }

    for (int i = 0;i < norb;i++)
    {
       E[i][i] -= v_onsite[i]/2.0   ;

       printf("orbital energies: %d, %f\n", i, E[i][i]) ;
    }

    auto& H =  this->put("H", new SymmetryBlockedTensor<U>("Fa", arena, PointGroup::C1(), 2, {{norb},{norb}}, {NS,NS}, true));
    auto& Da = this->put("Da", new SymmetryBlockedTensor<U>("Da", arena, PointGroup::C1(), 2, {{norb},{norb}}, {NS,NS}, true));
    auto& Db = this->put("Db", new SymmetryBlockedTensor<U>("Db", arena, PointGroup::C1(), 2, {{norb},{norb}}, {NS,NS}, true));
    auto& S  = this->put("S", new SymmetryBlockedTensor<U>("S", arena, PointGroup::C1(), 2, {{norb},{norb}}, {NS,NS}, true));

    vector<tkv_pair<U>> dapairs;
    vector<tkv_pair<U>> dbpairs;
    vector<tkv_pair<U>> fpairs;
    vector<tkv_pair<U>> ov_pairs;

    for (int i = 0;i < norb;i++)
    {
        ov_pairs.emplace_back(i*norb+i, 1);
    }

    hubbard.alphastring_to_vector(alpha_array) ;
    hubbard.betastring_to_vector(beta_array) ;

   if (coeff_exists) 
   {
    for (int i = 0;i < norb;i++)
    {
       for (int j = 0;j < norb;j++)
       {
        dapairs.emplace_back(i*norb+j, Dalpha[i][j]);
       }
    }

   for (int i = 0;i < norb;i++)
    {
       for (int j = 0;j < norb;j++)
       {
        dbpairs.emplace_back(i*norb+j, Dbeta[i][j]);
       }
    }
   }
   else
   {
    for (int i = 0;i < ndoc;i++)
    {
        dapairs.emplace_back(i*norb+i, 1);
    }

    for (int i = 0;i < alpha_array.size();i++)
    {
        dapairs.emplace_back((i+ndoc)*norb+(i+ndoc), alpha_array[i]);
    }

    for (int i = 0;i < ndoc;i++)
    {
        dbpairs.emplace_back(i*norb+i, 1);
    }

    for (int i = 0;i < beta_array.size();i++)
    {
        dbpairs.emplace_back((i+ndoc)*norb+(i+ndoc), beta_array[i]);
    }
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
        Da.writeRemoteData({0,0}, dapairs);
        Db.writeRemoteData({0,0}, dbpairs);
        H.writeRemoteData({0,0}, fpairs);
        S.writeRemoteData({0,0}, ov_pairs);
    }
    else
    {
        Da.writeRemoteData({0,0});
        Db.writeRemoteData({0,0});
        H.writeRemoteData({0,0});
        S.writeRemoteData({0,0}, ov_pairs);
    }
    return true;
}

}
}

INSTANTIATE_SPECIALIZATIONS(aquarius::hubbard::ReadInts);
REGISTER_TASK(aquarius::hubbard::ReadInts<double>,"read_integrals");
