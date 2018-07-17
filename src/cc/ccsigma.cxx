#include "util/global.hpp"

#include "task/task.hpp"
#include "operator/2eoperator.hpp"
#include "operator/space.hpp"
#include "operator/st2eoperator.hpp"
#include "operator/excitationoperator.hpp"
#include "hubbard/uhf_modelH.hpp"
#include "scf/uhf.hpp"

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
        int gf_type ;
        double beta ;
        vector<U> integral_diagonal ;
        vector<U> v_onsite ;

    public:
        CCSDSIGMA(const string& name, Config& config): Task(name, config)
        {
            vector<Requirement> reqs;
            reqs.emplace_back("moints", "H");
            reqs.emplace_back("occspace", "occ");
            reqs.emplace_back("vrtspace", "vrt");
            reqs.emplace_back("Ea", "Ea");
            reqs.emplace_back("Eb", "Eb");
            reqs.emplace_back("Fa", "Fa");
            reqs.emplace_back("Fb", "Fb");
            reqs.emplace_back("ccsd.ipgf", "gf_ip");
            reqs.emplace_back("ccsd.eagf", "gf_ea");
            this->addProduct(Product("ccsd.sigma", "sigma", reqs));

            double from = config.get<double>("omega_min");
            double to = config.get<double>("omega_max");
            nmax = config.get<double>("frequency_points");
            double eta = config.get<double>("eta");
            nr_impurities = config.get<int>("impurities"); 
            beta = config.get<double>("beta");
            string grid_type = config.get<string>("grid");
            string gf_string = config.get<string>("gftype");  

            double delta = (to-from)/max(1,nmax-1);
            for (int i = 0;i < nmax;i++)
            {
               if (grid_type == "imaginary") omegas.emplace_back(0.,(2.0*i+1)*M_PI/beta);
               if (grid_type == "real") omegas.emplace_back(from+delta*i, eta);
            }

            if (gf_string == "symmetrized") gf_type = 1 ;
            if (gf_string == "nonsymmetrized") gf_type = 2 ;
         } 


//       inline U f_tau(U tau, U beta, U c1, U c2, U c3){
//          return -0.5*c1 + (c2*0.25)*(-beta+2.*tau) + (c3*0.25)*(beta*tau-tau*tau);
//       }

//       inline std::complex<U> f_omega(std::complex<U> iw, U c1, U c2, U c3) {
//         std::complex<ft_float_type> iwsq=iw*iw;
//         return c1/iw + c2/(iwsq) + c3/(iw*iwsq);
//       }

//       double &c1(int s1, int s2, int f){ return c1_[s1*ns_*nf_+s2*nf_+f]; }
//       double &c2(int s1, int s2, int f){ return c2_[s1*ns_*nf_+s2*nf_+f]; }
//       double &c3(int s1, int s2, int f){ return c3_[s1*ns_*nf_+s2*nf_+f]; }


        bool run(TaskDAG& dag, const Arena& arena)
        {
         int nirreps = 1;
         auto& gf_ea = this->template get<vector<vector<vector<vector<CU>>>>>("gf_ea");
         auto& gf_ip = this->template get<vector<vector<vector<vector<CU>>>>>("gf_ip");

         auto& H = this->template get<TwoElectronOperator<U>>("H");
         const auto& occ = this->template get<MOSpace<U>>("occ");
         const auto& vrt = this->template get<MOSpace<U>>("vrt");

         auto& Ea = this->template get<vector<vector<real_type_t<U>>>>("Ea");
         auto& Eb = this->template get<vector<vector<real_type_t<U>>>>("Eb");

         auto& Fa     = this->template get   <SymmetryBlockedTensor<U>>("Fa");
         auto& Fb     = this->template get   <SymmetryBlockedTensor<U>>("Fb");

         const SymmetryBlockedTensor<U>& cA_ = vrt.Calpha;
         const SymmetryBlockedTensor<U>& ca_ = vrt.Cbeta;
         const SymmetryBlockedTensor<U>& cI_ = occ.Calpha;
         const SymmetryBlockedTensor<U>& ci_ = occ.Cbeta;

         const SpinorbitalTensor<U>&   FME =   H.getIA();
         const SpinorbitalTensor<U>&   FAE =   H.getAB();
         const SpinorbitalTensor<U>&   FMI =   H.getIJ();

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
       vector<U> fockoo;
       vector<U> fockvv;
       vector<U> fockov;

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

    vector<U> density(norb*norb,{0.}) ;
    vector<U> fock(norb*norb,0.);
    U energy = 0. ; 
 
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


//      fock.clear() ;

    for (int i = 0;i < nirreps;i++)
    {
//    (nspin == 0) ? fock.assign(Ea[i].begin(), Ea[i].end()) : fock.assign(Eb[i].begin(), Eb[i].end());
//      (nspin == 0) ? Fa({i,i}).getAllData(fock) : Fb({i,i}).getAllData(fock);
//        (nspin == 0) ? FMI({i,i}).getAllData(fockoo) : FMI({i,i}).getAllData(fockoo);
//        (nspin == 0) ? FME({i,i}).getAllData(fockov) : FME({i,i}).getAllData(fockov);
//        (nspin == 0) ? FAE({i,i}).getAllData(fockvv) : FAE({i,i}).getAllData(fockvv);
    }

           FMI({nspin,0},{0,nspin})({0,0}).getAllData(fockoo); 
           FME({0,nspin},{nspin,0}).getAllData({0,0},fockov);
           FAE({nspin,0},{nspin,0}).getAllData({0,0},fockvv);

           int nocc = (nspin == 0 ? occ.nalpha[0] : occ.nbeta[0]) ;
           int nvirt = (nspin == 0 ? vrt.nalpha[0] : vrt.nbeta[0]) ;

           for (int i=0 ; i < norb ; i++){
            for (int j=0 ; j < norb ; j++){
             if ((i < nocc) && (j < nocc)) fock[i*norb+j] = fockoo[i*nocc+j] ; 
             if ((i < nocc) && (j >= nocc)) fock[i*norb+j] = fockov[i*nvirt+(j-nocc)] ; 
//             if ((i >= nocc) && (j < nocc)) fock[i*norb+j] = 1.0*fockov[(i-nocc)*nocc+j] ; 
             if ((i >= nocc) && (j >= nocc)) fock[i*norb+j] = fockvv[(i-nocc)*nvirt+(j-nocc)] ; 
             fock[j*norb+i] = fock[i*norb+j] ;
            }
           } 

      for (int p = 0; p < norb ;p++)
      {
        for (int q = 0; q < norb ;q++)
        {
          if (p==q) density[p*norb+q] += 0.5;
          density[p*norb+q] -= fock[p*norb+q]*(beta/4.);
        }
      }

     std::ifstream iffile("g_0_omega.dat");
     if (iffile) remove("g_0_omega.dat");

    for (int omega = 0; omega < omegas.size() ;omega++)
    {
      vector<CU> gf_ao(nr_impurities*nr_impurities,{0.,0.}) ;
      vector<int> ipiv(norb) ;
//      vector<int> ipiv(nr_impurities) ;
      vector<CU> gf_tmp(norb*norb,{0.,0.}) ;
      vector<CU> gf_original(norb*norb,{0.,0.}) ;
      vector<CU> gf_zero_inv(norb*norb) ;
      vector<CU> G_inv(nr_impurities*nr_impurities) ;

      for (int p = 0; p < norb ;p++)
      {
        for (int q = 0; q < norb ;q++)
        {

       if (gf_type == 1)
       {
         if (q == p)
          {
//            gf_zero_inv[p*norb+q] =1.0/(omegas[omega] - fock[p]) ;
           gf_zero_inv[p*norb+q] = (omegas[omega] - fock[p*norb+q]) ;
            gf_tmp[p*norb+q] = gf_ip[nspin][omega][p][q] + gf_ea[nspin][omega][p][q] ; 
          }
          if (q != p )
          { 
//            gf_zero_inv[p*norb+q] =  {0.,0.} ;
            gf_zero_inv[p*norb+q] = -fock[p*norb+q] ;
            gf_tmp[p*norb+q]  = 0.5*( gf_ip[nspin][omega][p][q] - gf_ip[nspin][omega][p][p] - gf_ip[nspin][omega][q][q]) ; 
            gf_tmp[p*norb+q] += 0.5*( gf_ea[nspin][omega][p][q] - gf_ea[nspin][omega][p][p] - gf_ea[nspin][omega][q][q]) ; 
          }
        }
        else 
        {
         gf_tmp[p*norb+q] = gf_ip[nspin][omega][p][q] + gf_ea[nspin][omega][p][q] ; 
        }

//          if (omega == 0) {density[p*norb+q] +=gf_tmp[p*norb+q].real() ;
//          }else{ 
//           density[p*norb+q] += (2.0/beta)*(cos(omegas[omega].imag()*beta)*gf_tmp[p*norb+q].real()-sin(omegas[omega].imag()*beta)*gf_tmp[p*norb+q].imag()s) ;
//         density[p*norb+q] += (2.0/beta)*(gf_tmp[p*norb+q].real()) ;
//         density[p*norb+q] += (2.0/beta)*(gf_zero_inv[p*norb+q].real()) ;
//           }

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
//        if (omega == 0) cout << omegas[omega].imag() << " " << i << " " << j << " " << gf_ao[i*nr_impurities+j].real() << " " << gf_ao[i*nr_impurities+j].imag() << std::endl ; 
        }  
       }

  /* calculate bare Green's function..
   */  

    std::ofstream gzero_omega;
    gzero_omega.open("g_0_omega.txt", ofstream::out);

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
         G_inv[u*nr_impurities+v] = omegas[omega] - integral_diagonal[u*norb+v] - Hybridization + 0.5*v_onsite[u]  ;
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
          gomega << omegas[omega].real() << " " << omegas[omega].imag() << " " << gf_ao[b*nr_impurities+b].real() << " " << (-1./M_PI)*gf_ao[b*nr_impurities+b].imag() << std::endl ; 
//        gomega << omegas[omega].real() << " " << omegas[omega].imag() << " " << gf_ao[b*nr_impurities+b].real() << " " << gf_ao[b*nr_impurities+b].imag() << std::endl ; 
         gomega.close();
       }

           getrf(nr_impurities, nr_impurities, gf_ao.data(), nr_impurities, ipiv.data() ) ;

           getri(nr_impurities, gf_ao.data(), nr_impurities, ipiv.data()) ;

           axpy (nr_impurities*nr_impurities, -1.0, gf_ao.data(), 1, G_inv.data(), 1);

//         copy(norb*norb, gf_tmp.data(), 1, gf_original.data(), 1) ;

//         getrf(norb, norb, gf_tmp.data(), norb, ipiv.data() ) ;

//         getri(norb, gf_tmp.data(), norb, ipiv.data()) ;

//         axpy (norb*norb, -1.0, gf_tmp.data(), 1, gf_zero_inv.data(), 1);

//        for (int p = 0; p < norb ;p++)
//        {
//         for (int q = 0; q < norb ;q++)
//          {
//             energy +=(gf_zero_inv[p*norb+q].real()*gf_original[p*norb+q].real() - gf_zero_inv[p*norb+q].imag()*gf_original[p*norb+q].imag()) ;
//          }
//        } 

//         axpy (norb*norb, 1.0, gf_tmp.data(), 1, gf_zero_inv.data(), 1);

//       ipiv.clear() ;

//       ipiv.resize(norb) ;

         getrf(norb, norb, gf_zero_inv.data(), norb, ipiv.data() ) ;

         getri(norb, gf_zero_inv.data(), norb, ipiv.data()) ;
      for (int p = 0; p < norb ;p++)
      {
        for (int q = 0; q < norb ;q++)
        {
//          density[p*norb+q] += (2.0/beta)*(cos(omegas[omega].imag()*beta)*gf_original[p*norb+q].real()-sin(omegas[omega].imag()*beta)*gf_original[p*norb+q].imag()) ;

//          density[p*norb+q] += (2.0/beta)*(cos(omegas[omega].imag()*beta)*gf_zero_inv[p*norb+q].real()-sin(omegas[omega].imag()*beta)*gf_zero_inv[p*norb+q].imag()) ;
//          if (p ==q) density[p*norb+q] += (2.0/beta)*((fock[p]/pow(omegas[omega].imag(),2))) ;
//          density[p*norb+q] += (2.0/beta)*(gf_zero_inv[p*norb+q].real()-1.0/omegas[omega]+(fock[p*norb+q]/pow(omegas[omega].imag(),2))).real() ;
//          density[p*norb+q] += (2.0/beta)*(gf_zero_inv[p*norb+q].real()-1.0/omegas[omega]).real() ;
          density[p*norb+q] += (2.0/beta)*(gf_tmp[p*norb+q].real()-1.0/omegas[omega]+(fock[p*norb+q]/pow(omegas[omega].imag(),2))).real() ;
//            density[p*norb+q] += (2.0/beta)*(gf_tmp[p*norb+q].real()-1.0/omegas[omega]).real() ;
//          density[p*norb+q] += (2.0/beta)*(gf_zero_inv[p*norb+q].real()) ;
//          density[p*norb+q] += (2.0/beta)*(gf_tmp[p*norb+q].real() - gf_zero_inv[p*norb+q].real()) ;
        }
      }

    
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

      printf("total energy: %.15f\n", (2.0/beta)*energy);

      for (int p = 0; p < norb ;p++)
      {
        for (int q = 0; q < norb ;q++)
        {

//           density[p*norb+q] += 0.5;
//           if (p==q) density[p*norb+q] += 0.5;
//           if ((p<nI) && (q<nI) && (p==q)) density[p*norb+q] += 1.;
           density[p*norb+q] *= 2 ;
//           if ((p<nI) && (q<nI) && (p==q)) density[p*norb+q] += 2;
//            density[p*norb+q] += 0.1 ;
//           cout << p << " " << q << " " << density[p*norb+q] << std::endl ; 
        }
      }

      energy *=2.0/beta ;

      for (int p = 0; p < norb ;p++)
      {
        energy += density[p*norb+p]*fock[p*norb+p] ; 
      }


      printf("total energy: %.15f\n", energy);

         vector<U> l(norb*norb);
         vector<CU> s_tmp(norb);
         vector<U> vr_tmp(norb*norb);

         int info_ener = geev('N', 'V', norb, fock.data(), norb,
                     s_tmp.data(), l.data(), norb,
                     vr_tmp.data(), norb);
         if (info_ener != 0) throw runtime_error(str("check diagonalization: Info in geev: %d", info_ener));

         cout<<" #orbital energies" <<endl ;

         for (int i=0 ; i < norb ; i++){
             printf(" %.15f\n", s_tmp[i].real());
          }

         l.clear() ;
         s_tmp.clear();
         vr_tmp.clear();

         U value = 0. ; 
         int info = geev('N', 'V', norb, density.data(), norb,
                     s_tmp.data(), l.data(), norb,
                     vr_tmp.data(), norb);
         if (info != 0) throw runtime_error(str("check diagonalization: Info in geev: %d", info));


         cout<<" #orbital occupation" <<endl ;

         for (int i=0 ; i < norb ; i++){
             printf(" %.15f\n", s_tmp[i].real());
//            value += s_tmp[i].real();
             value += density[i*norb+i];
          }

         printf("total occupancy: %.15f\n", value);

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
        enum{ imaginary, real },
    gftype?
        enum{ symmetrized, nonsymmetrized },
)";

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::CCSDSIGMA);
REGISTER_TASK(aquarius::cc::CCSDSIGMA<double>, "ccsdsigma",spec);
