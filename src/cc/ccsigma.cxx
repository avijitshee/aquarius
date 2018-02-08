#include "util/global.hpp"

#include "convergence/lanczos.hpp"
#include "util/iterative.hpp"
#include "operator/2eoperator.hpp"
#include "operator/space.hpp"
#include "operator/st2eoperator.hpp"
#include "operator/excitationoperator.hpp"
#include "task/task.hpp"
#include "hubbard/uhf_modelH.hpp"
#include "mocoeff.hpp"

using namespace aquarius::hubbard;
using namespace aquarius::tensor;
using namespace aquarius::task;
using namespace aquarius::input;
using namespace aquarius::op;
using namespace aquarius::symmetry;

namespace aquarius
{
namespace cc
{

template <typename U>
class CCSDSIGMA : public MOCoeffs<U>
{
    protected:
        typedef complex_type_t<U> CU;

        int orbital;
        vector<CU> omegas;
        CU omega;
        int nmax;

    public:
        CCSDSIGMA(const string& name, Config& config)
        :MOCoeffs<U>(name, config)
        {
            vector<Requirement> reqs;
            reqs.emplace_back("ccsd.ipgflanczos", "gf_ip");
            reqs.emplace_back("ccsd.eagflanczos", "gf_ea");

            double from = config.get<double>("omega_min");
            double to = config.get<double>("omega_max");
            nmax = config.get<double>("frequency_points");
            double eta = config.get<double>("eta");
            double beta = config.get<double>("beta");
            string grid_type = config.get<string>("grid");

            double delta = (to-from)/max(1,nmax-1);
            for (int i = 0;i < nmax;i++)
            {
               if (grid_type == "real") omegas.emplace_back(from+delta*i, eta);
               if (grid_type == "imaginary") omegas.emplace_back(0.,(2.0*i+1)*M_PI/beta);
            }
         } 

        bool run(TaskDAG& dag, const Arena& arena)
        {
         int nirreps = 1;
         auto& gf_ip = this->template get<vector<vector<vector<CU>>>>("gf_ip");
         auto& gf_ea = this->template get<vector<vector<vector<CU>>>>("gf_ea");

         const auto& occ = this->template get<MOSpace<U>>("occ");
         const auto& vrt = this->template get<MOSpace<U>>("vrt");

         auto& Fa = this->template get<SymmetryBlockedTensor<U>>("Fa");
         auto& Fb = this->template get<SymmetryBlockedTensor<U>>("Fb");

         const SymmetryBlockedTensor<U>& cA_ = vrt.Calpha;
         const SymmetryBlockedTensor<U>& ca_ = vrt.Cbeta;
         const SymmetryBlockedTensor<U>& cI_ = occ.Calpha;
         const SymmetryBlockedTensor<U>& ci_ = occ.Cbeta;

         vector<vector<U>> cA(nirreps), ca(nirreps), cI(nirreps), ci(nirreps);

         const vector<int>& N = occ.nao;
         const vector<int>& nI = occ.nalpha;
         const vector<int>& ni = occ.nbeta;
         const vector<int>& nA = vrt.nalpha;
         const vector<int>& na = vrt.nbeta;
 
         int norb = N[0]; 
         int nocc = nI[0]; 
 
         for (int i = 0;i < nirreps;i++)
         {
           vector<int> irreps = {i,i};
           cA_.getAllData(irreps, cA[i]);
           assert(cA[i].size() == N[i]*nA[i]);
           ca_.getAllData(irreps, ca[i]);
           assert(ca[i].size() == N[i]*na[i]);
           cI_.getAllData(irreps, cI[i]);
           assert(cI[i].size() == N[i]*nI[i]);
           ci_.getAllData(irreps, ci[i]);
           assert(ci[i].size() == N[i]*ni[i]);
         } 

          printf("print debug statement 1: %.15f\n", 10.4);

    vector<U> c_full ; 

    for (int i = 0;i < nirreps;i++)
    {
      c_full.insert(c_full.end(),cI[i].begin(),cI[i].end());
      c_full.insert(c_full.end(),cA[i].begin(),cA[i].end());
    }
 
    printf("print debug statement: %d\n", norb);

    vector<CU> gf_ao(nmax,{0.,0.}) ;

    for (int omega = 0; omega < nmax ;omega++)
    {
      vector<CU> gf_tmp(norb*norb) ;
      gf_ao[omega] = {0.,0.} ;
      for (int p = 0; p < norb ;p++)
      {
        for (int q = 0; q < norb ;q++)
        {
          if (q == p)
          {
          gf_tmp[p*norb+q] = gf_ip[p][q][omega] + gf_ea[p][q][omega] ; 
          }
          if (q != p )
          { 
            gf_tmp[p*norb+q]  = 0.5*( gf_ip[p][q][omega] - gf_ip[p][p][omega] - gf_ip[q][q][omega]) ; 
            gf_tmp[p*norb+q] += 0.5*( gf_ea[p][q][omega] - gf_ea[p][p][omega] - gf_ea[q][q][omega]) ; 
          }
          for (int i = 0; i < 1; i++) 
          {
            gf_ao[omega] = gf_ao[omega] + c_full[p*norb+i]*c_full[q*norb+i]*gf_tmp[p*norb+q] ; 
          }  
        }
      }
         printf("Real and Imaginary value of the Green's function %10f %15f %15f\n",omegas[omega].imag(), gf_ao[omega].real(), gf_ao[omega].imag());
    } 
   }
  };

}
}

static const char* spec = R"(

    frequency_points int,
    basis_type?
        enum { AO, MO },
    omega_min double,
    omega_max double,
    eta double,
    beta double, 
    grid?
        enum{ real, imaginary }
)";

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::CCSDSIGMA);
REGISTER_TASK(aquarius::cc::CCSDSIGMA<double>, "ccsdsigma",spec);
