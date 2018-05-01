#include "util/global.hpp"

#include "task/task.hpp"
#include "operator/2eoperator.hpp"
#include "operator/space.hpp"
#include "operator/st2eoperator.hpp"
#include "operator/excitationoperator.hpp"
#include "hubbard/uhf_modelH.hpp"

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
class CCSDSIGMA: public Task 
{
    protected:
        typedef complex_type_t<U> CU;

        vector<CU> omegas;
        CU omega;
        int nmax;
        int nr_impurities; 
        string gf_type ;
        vector<U> integral_diagonal ;
        vector<U> v_onsite ;

    public:
        CCSDSIGMA(const string& name, Config& config): Task(name, config)
        {
            vector<Requirement> reqs;
            reqs.emplace_back("occspace", "occ");
            reqs.emplace_back("vrtspace", "vrt");
            reqs.emplace_back("Ea", "Ea");
            reqs.emplace_back("Eb", "Eb");
            reqs.emplace_back("ccsd.ipgflanczos", "gf_ip");
            reqs.emplace_back("ccsd.eagflanczos", "gf_ea");
            this->addProduct(Product("ccsd.sigma", "sigma", reqs));


            double from = config.get<double>("omega_min");
            double to = config.get<double>("omega_max");
            nmax = config.get<double>("frequency_points");
            double eta = config.get<double>("eta");
            nr_impurities = config.get<int>("impurities"); 
            double beta = config.get<double>("beta");
            string grid_type = config.get<string>("grid");
            gf_type = config.get<string>("gftype");  

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
         auto& gf_ea = this->template get<vector<vector<vector<vector<CU>>>>>("gf_ea");
         auto& gf_ip = this->template get<vector<vector<vector<vector<CU>>>>>("gf_ip");

         const auto& occ = this->template get<MOSpace<U>>("occ");
         const auto& vrt = this->template get<MOSpace<U>>("vrt");

         auto& Ea = this->template get<vector<vector<real_type_t<U>>>>("Ea");
         auto& Eb = this->template get<vector<vector<real_type_t<U>>>>("Eb");

         const SymmetryBlockedTensor<U>& cA_ = vrt.Calpha;
         const SymmetryBlockedTensor<U>& ca_ = vrt.Cbeta;
         const SymmetryBlockedTensor<U>& cI_ = occ.Calpha;
         const SymmetryBlockedTensor<U>& ci_ = occ.Cbeta;

         vector<vector<U>> cA(nirreps), ca(nirreps), cI(nirreps), ci(nirreps);

         const vector<int>& N = occ.nao;
         int nI = occ.nalpha[0];
         int ni = occ.nbeta[0];
         int nA = vrt.nalpha[0];
         int na = vrt.nbeta[0];
 
         int maxspin = (nI == ni) ? 1 : 2 ;

         int norb = N[0]; 
 
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

       vector<U> c_full ; 
       vector<U> fock;

 /* remove all self-energy and total Green's function files those are already there in the directory..
  */  

       for (int nspin = 0; nspin < maxspin ; nspin++)   
       { 
        for (int i = 0;i < nr_impurities;i++)
        { 
          std::stringstream stream;
          stream << "gomega_"<<nspin<<"_"<<i<< ".txt";
          std::string fileName = stream.str();
          std::ifstream gfile(fileName.c_str());
          if (gfile) remove(fileName.c_str());
        } 
       }

       for (int nspin = 0; nspin < maxspin ; nspin++)   
       { 
        for (int i = 0;i < nr_impurities;i++)
        { 
          std::stringstream stream;
          stream << "self_energy_"<<nspin<<"_"<<i<< ".txt";
          std::string fileName = stream.str();
          std::ifstream sigmafile(fileName.c_str());
          if (sigmafile) remove(fileName.c_str());
        } 
       }


    /* Reading one electron integrals here. It will be nicer to read them separately within a function.. 
     */   

       std::ifstream one_diag("one_diag.txt");
       std::istream_iterator<U> start(one_diag), end;
       vector<U> diagonal(start, end);
       std::copy(diagonal.begin(), diagonal.end(),std::back_inserter(integral_diagonal)); 

       std::ifstream onsite("onsite.txt");
       std::istream_iterator<U> start_v(onsite), end_v;
       vector<U> int_onsite(start_v, end_v);
       std::copy(int_onsite.begin(), int_onsite.end(), std::back_inserter(v_onsite)) ;
 
   for (int nspin = 0;nspin < maxspin;nspin++)
   {
    for (int i = 0;i < nirreps;i++)
     {
       vector<int> irreps = {i,i};
       (nspin == 0) ? cA_.getAllData(irreps, cA[i]) : ca_.getAllData(irreps, ca[i]);
       assert(cA[i].size() == N[i]*nA[i]);
       assert(ca[i].size() == N[i]*na[i]);
       (nspin == 0) ? cI_.getAllData(irreps, cI[i]) : ci_.getAllData(irreps, ci[i]);
       assert(cI[i].size() == N[i]*nI[i]);
       assert(ci[i].size() == N[i]*ni[i]);
     }        

     c_full.clear() ;  
    for (int i = 0;i < nirreps;i++)
    {
      (nspin == 0) ? c_full.insert(c_full.begin(),cI[i].begin(),cI[i].end()) : c_full.insert(c_full.begin(),ci[i].begin(),ci[i].end());
      (nspin == 0) ? c_full.insert(c_full.end(),cA[i].begin(),cA[i].end()) : c_full.insert(c_full.end(),ca[i].begin(),ca[i].end());
    }

      fock.clear() ;

    for (int i = 0;i < nirreps;i++)
    {
       (nspin == 0) ? fock.assign(Ea[i].begin(), Ea[i].end()) : fock.assign(Eb[i].begin(), Eb[i].end());
    }

     std::ifstream iffile("g_0_omega.dat");
     if (iffile) remove("g_0_omega.dat");

    for (int omega = 0; omega < omegas.size() ;omega++)
    {
      vector<CU> gf_ao(nr_impurities*nr_impurities,{0.,0.}) ;
      vector<int> ipiv(nr_impurities) ;
      vector<CU> gf_tmp(norb*norb) ;
      vector<CU> gf_zero_inv(norb*norb) ;
      vector<CU> G_inv(nr_impurities*nr_impurities) ;

      for (int p = 0; p < norb ;p++)
      {
        for (int q = 0; q < norb ;q++)
        {
         
         if (gf_type == "symmetrized")
         {
          if (q == p)
          {
            gf_zero_inv[p*norb+q] = omegas[omega] - fock[q] ;
            gf_tmp[p*norb+q] = gf_ip[nspin][omega][p][q] + gf_ea[nspin][omega][p][q] ; 
          }
          if (q != p )
          { 
            gf_zero_inv[p*norb+q] =  {0.,0.} ;
            gf_tmp[p*norb+q]  = 0.5*( gf_ip[nspin][omega][p][q] - gf_ip[nspin][omega][p][p] - gf_ip[nspin][omega][q][q]) ; 
            gf_tmp[p*norb+q] += 0.5*( gf_ea[nspin][omega][p][q] - gf_ea[nspin][omega][p][p] - gf_ea[nspin][omega][q][q]) ; 
          }
         }
         else
         {
            gf_tmp[p*norb+q] = gf_ip[nspin][omega][p][q] + gf_ea[nspin][omega][p][q] ; 
         }  

          for (int i = 0; i < nr_impurities; i++) 
          {
           for (int j = 0; j < nr_impurities; j++) 
           {
            gf_ao[i*nr_impurities + j] = gf_ao[i*nr_impurities + j] + c_full[p*norb+i]*c_full[q*norb+j]*gf_tmp[p*norb+q] ; 
           }  
          }
        }
      }

      for (int i = 0; i < nr_impurities; i++) 
        {
         for (int j = 0; j < nr_impurities; j++) 
         {
           if (omega == 0) cout << omegas[omega].imag() << " " << i << " " << j << " " << gf_ao[i*nr_impurities+j].real() << " " << gf_ao[i*nr_impurities+j].imag() << std::endl ; 
         }  
        }

  /* calculate bare Green's function..
   */  

    std::ofstream gzero_omega;

    for (int u = 0; u < nr_impurities; u++)
     {
     for (int v = 0; v < nr_impurities; v++)
      {
       CU Hybridization{0.,0.} ;
       for (int b = nr_impurities; b < norb; b++)
       {
        Hybridization += (integral_diagonal[u*norb+b]*integral_diagonal[v*norb+b])/(omegas[omega] - integral_diagonal[b*norb+b]) ;
       }
        if (v == u) 
        {
         G_inv[u*nr_impurities+v] = omegas[omega] - integral_diagonal[u*norb+v] -  Hybridization + 0.5*v_onsite[u]  ;
        }
        else
        {
         G_inv[u*nr_impurities+v] = - integral_diagonal[u*norb+v] -  Hybridization  ;
        } 
         gzero_omega << omegas[omega].imag() << " " << u << " " << v << " " << G_inv[u*nr_impurities+v].real() << " " << G_inv[u*nr_impurities+v].imag() << std::endl ; 

      }
     }

       gzero_omega.close();

   /* Dyson Equation  
    */
   
       for (int b = 0 ; b < nr_impurities; b++)
       {
         std::stringstream stream;
          stream << "gomega_"<<nspin<<"_"<<b<< ".txt";
          std::string fileName = stream.str();
         std::ofstream gomega;
          gomega.open (fileName.c_str(), ofstream::out|std::ios::app);
          gomega << nspin << " " << b << " " << omegas[omega].imag() << " " << gf_ao[b*nr_impurities+b].real() << " " << gf_ao[b*nr_impurities+b].imag() << std::endl ; 
         gomega.close();
       }

         getrf(nr_impurities, nr_impurities, gf_ao.data(), nr_impurities, ipiv.data() ) ;

         getri(nr_impurities, gf_ao.data(), nr_impurities, ipiv.data()) ;

         axpy (nr_impurities*nr_impurities, -1.0, gf_ao.data(), 1, G_inv.data(), 1);

       for (int b = 0 ; b < nr_impurities; b++)
        {
         std::stringstream stream;
          stream << "self_energy_"<<nspin<<"_"<<b<< ".txt";
          std::string fileName = stream.str();
         std::ofstream gomega1;
          gomega1.open (fileName.c_str(), ofstream::out|std::ios::app);
          gomega1 << nspin << " " << b << " " << omegas[omega].imag() << " " << G_inv[b*nr_impurities+b].real() << " " << G_inv[b*nr_impurities+b].imag() << std::endl ; 
         gomega1.close();
       }
    } 
   }
      return true;
   }
  };

}
}

static const char* spec = R"(

    frequency_points int,
    omega_min double,
    omega_max double,
    impurities int,
    eta double,
    beta double, 
    grid?
        enum{ real, imaginary },
    gftype?
        enum{ symmetrized, nonsymmetrized }
)";

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::CCSDSIGMA);
REGISTER_TASK(aquarius::cc::CCSDSIGMA<double>, "ccsdsigma",spec);
