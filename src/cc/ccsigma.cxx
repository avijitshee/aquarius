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

//         int norb = N[0]; 
         int norb = nI+nA; 
 
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
    vector<CU> sigma(norb*norb,{0.,0.}) ;
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
//           if ((i >= nocc) && (j < nocc)) fock[i*norb+j] = 1.0*fockov[(i-nocc)*nocc+j] ; 
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
      vector<CU> gf_zero_inv(norb*norb,{0.,0.}) ;
      vector<CU> G_inv(nr_impurities*nr_impurities) ;

      sigma.clear() ;

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

//        for (int i = 0; i < nr_impurities; i++) 
//        {
//         for (int j = 0; j < nr_impurities; j++) 
//         {
//          gf_ao[i*nr_impurities + j] = gf_ao[i*nr_impurities + j] + c_full[p*norb+i]*c_full[q*norb+j]*gf_tmp[p*norb+q] ; 
//         }  
//        }
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

//  std::ofstream gzero_omega;
//  gzero_omega.open("g_0_omega.txt", ofstream::out);

//  for (int u = 0; u < nr_impurities; u++)
//   {
//   for (int v = 0; v < nr_impurities; v++)
//    {
//     CU Hybridization{0.,0.} ;
//     for (int b = nr_impurities; b < norb; b++)
//     {
//      Hybridization += (integral_diagonal[u*norb+b]*integral_diagonal[v*norb+b])/(omegas[omega] - integral_diagonal[b*norb+b]) ;
//     }
//      if (v == u) 
//      {
//       G_inv[u*nr_impurities+v] = omegas[omega] - integral_diagonal[u*norb+v] - Hybridization + 0.5*v_onsite[u]  ;
//      }
//      else
//      {
//       G_inv[u*nr_impurities+v] = - integral_diagonal[u*norb+v] -  Hybridization  ;
//      } 
//       gzero_omega << omegas[omega].imag() << " " << u << " " << v << " " << G_inv[u*nr_impurities+v].real() << " " << G_inv[u*nr_impurities+v].imag() << std::endl ; 
//    }
//   }

//     gzero_omega.close();

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

           copy(norb*norb, gf_tmp.data(), 1, gf_original.data(), 1) ;
           copy(norb*norb, gf_zero_inv.data(), 1, sigma.data(), 1) ;

           getrf(norb, norb, gf_tmp.data(), norb, ipiv.data() ) ;

           getri(norb, gf_tmp.data(), norb, ipiv.data()) ;

           axpy (norb*norb, -1.0, gf_tmp.data(), 1, sigma.data(), 1);

          for (int p = 0; p < norb ;p++)
          {
           for (int q = 0; q < norb ;q++)
            {
               energy +=(sigma[p*norb+q].real()*gf_original[p*norb+q].real() - sigma[p*norb+q].imag()*gf_original[p*norb+q].imag()) ;
            }
          } 

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
//            density[p*norb+q] += (2.0/beta)*(gf_tmp[p*norb+q].real()-1.0/omegas[omega]+(fock[p*norb+q]/pow(omegas[omega].imag(),2))).real() ;
           if (p == q) 
           {
              density[p*norb+q] += (2.0/beta)*(gf_original[p*norb+q].real()-1.0/omegas[omega]+(fock[p*norb+q]/pow(omegas[omega].imag(),2))).real() ;
           }else{
              density[p*norb+q] += (2.0/beta)*(gf_original[p*norb+q].real()+(fock[p*norb+q]/pow(omegas[omega].imag(),2))) ;
           }
         
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

      vector<CU> sigma_xx(norb*norb,0.0) ;
 
      for (int p = 0; p < norb ;p++)
      {
        for (int q = 0; q < norb ;q++)
        {
          sigma_xx[p*norb+q] = -1.0*sigma[p*norb+q].imag()*omegas[nmax-1].imag() ; 
        }
      }

      printf("Tr(Sigma.G) energy: %.15f\n", (2.0/beta)*energy);

      for (int p = 0; p < norb ;p++)
      {
        for (int q = 0; q < norb ;q++)
        {
           density[p*norb+q] *= 1.0 ;
        }
      }

      vector<U> density_ao(norb*norb,0.) ;

      energy *=2.0/beta ;

      auto& D = this->puttmp("D", new OneElectronOperator<U>("D", arena, occ, vrt));
      auto& DIA = D.getIA();
      auto& DAI = D.getAI();
      auto& DAB = D.getAB();
      auto& DIJ = D.getIJ();

      vector<tkv_pair<U>> pairs;
//      vector<tkv_pair<U>> pair_right{{orbright_dummy, 1}};

     U val ; 

      for (int p = 0; p < nI ;p++)
      {
      for (int q = 0; q < nI ;q++)
       {
           val = density[p*norb+q];
           pairs.push_back(tkv_pair<U>(p*nI+q,val)) ;
       }
      }

     if (arena.rank == 0){
        DIJ({0,0},{0,0})({0,0}).writeRemoteData(pairs); 
        DIJ({0,1},{0,1})({0,0}).writeRemoteData(pairs); }
     else{
        DIJ({0,0},{0,0})({0,0}).writeRemoteData(); 
        DIJ({0,1},{0,1})({0,0}).writeRemoteData(); }

      pairs.clear() ;  

      for (int p = nI; p < norb ;p++)
      {
      for (int q = nI; q < norb ;q++)
       {
           val = density[p*norb+q];
           pairs.push_back(tkv_pair<U>((p-nI)*nA+(q-nI),val)) ;
       }
      }

     if (arena.rank == 0){
        DAB({0,0},{0,0})({0,0}).writeRemoteData(pairs); 
        DAB({1,0},{1,0})({0,0}).writeRemoteData(pairs);}
     else {
        DAB({0,0},{0,0})({0,0}).writeRemoteData(); 
        DAB({1,0},{1,0})({0,0}).writeRemoteData(); }

       const SpinorbitalTensor<U>& WMNIJ = H.getIJKL();
       const SpinorbitalTensor<U>& WAMEI = H.getAIBJ();

       const PointGroup& group = H.getIJKL().getGroup();
       int nirrep = group.getNumIrreps();

       auto& XI = this->puttmp("XI", new SpinorbitalTensor<U>("X(i)", arena, group, {vrt,occ}, {0,1}, {0,1}));
       SpinorbitalTensor<U> delta ("delta"  , arena, group, {vrt,occ}, {0,1}, {0,1});

      pairs.clear() ;

      for (int p = 0; p < nI ;p++)
      {
           pairs.push_back(tkv_pair<U>(p*nI+p,1.0)) ;
      }

     if (arena.rank == 0) {
       delta({0,0},{0,0})({0,0}).writeRemoteData(pairs); 
       delta({0,1},{0,1})({0,0}).writeRemoteData(pairs);}
     else { 
       delta({0,0},{0,0})({0,0}).writeRemoteData(); 
       delta({0,1},{0,1})({0,0}).writeRemoteData();} 

       XI["ii"]  = WMNIJ["jiki"]*DIJ["jk"] ;  
       XI["ii"] += WAMEI["aibi"]*DAB["ab"] ;  

       U E1B = 0.5*scalar(XI["ii"]*delta["ii"]) ; 

      printf("additional contribution to Fock, 1-body energy contribution: %.15f\n", -E1B);

      for (int p = 0; p < norb ;p++)
      {
      for (int q = 0; q < norb ;q++)
       {
         E1B -= 2.0*density[p*norb+q]*fock[p*norb+q] ; 
       }
      }

      printf("Total 1-body energy contribution: %.15f\n", -E1B);

      energy -= E1B ; 

//calculation of high-frequency tail--
     
      U e2b_hf = 0. ;
      CU omega ;
      for (int w = nmax; w < 1000000 ;w++)
      {
        omega = {0.0,(2.0*w+1)*M_PI/beta} ;      
      for (int p = 0; p < norb ;p++)
      {
        for (int q = 0; q < norb ;q++)
        {
         if (p==q) e2b_hf +=  2.0*(-(1.0/omega).imag()*(sigma_xx[p*norb+q]/omega).imag()) ; 
        }
      }
      }

     printf("high frequency tail: %.15f\n", e2b_hf/beta);
     energy +=e2b_hf/beta ;

    printf("total energy: %.15f\n", energy);

    for (int p = 0; p < norb ;p++)
      {
        for (int q = 0; q < norb ;q++)
        {
         density[p*norb+q] *= 1.0 ;
        for (int i = 0; i < norb; i++) 
        {
         for (int j = 0; j < norb; j++) 
         {
          density_ao[i*norb + j] = density_ao[i*norb + j] + c_full[p*norb+i]*c_full[q*norb+j]*density[p*norb+q] ; 
         }  
        }
      }
     }

    std::ofstream dens_ao;
    dens_ao.open("coeff.txt", ofstream::out);

      for (int p = 0; p < norb ;p++)
      {
        for (int q = 0; q < norb ;q++)
        {
//          dens_ao << setprecision(10) << density_ao[p*norb + q] << std::endl ; 
        }
      }

        dens_ao.close() ;






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
             printf(" %.15f\n", 2.0*s_tmp[i].real());
//            value += s_tmp[i].real();
             value += density[i*norb+i];
          }

         printf("total occupancy: %.15f\n", value);

      return true;
   }

   void calculate_density ()
   {



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
