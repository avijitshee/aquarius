#include "util/global.hpp"

#include "task/task.hpp"
#include "operator/2eoperator.hpp"
#include "operator/space.hpp"
#include "operator/st2eoperator.hpp"
#include "operator/excitationoperator.hpp"
#include "hubbard/uhf_modelH.hpp"
#include "AIM/uhf_modelH.hpp"
#include "scf/uhf.hpp"
#include <mkl_cblas.h>

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
        int norb ; 
        int nI, ni, nA, na ;
        string grid_type ;
        vector<U> integral_diagonal ;
        vector<U> v_onsite ;
        vector<U> fock;

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

            reqs.emplace_back("ccsd.ipalpha", "alpha_ip");
            reqs.emplace_back("ccsd.ipbeta", "beta_ip");
            reqs.emplace_back("ccsd.ipgamma", "gamma_ip");

            reqs.emplace_back("ccsd.eaalpha", "alpha_ea");
            reqs.emplace_back("ccsd.eabeta", "beta_ea");
            reqs.emplace_back("ccsd.eagamma", "gamma_ea");
            reqs.emplace_back("ccsd.ipnorm", "norm_ip");
            reqs.emplace_back("ccsd.eanorm", "norm_ea");

            this->addProduct(Product("ccsd.sigma", "sigma", reqs));

            double from = config.get<double>("omega_min");
            double to = config.get<double>("omega_max");
            nmax = config.get<double>("npoint");
            double eta = config.get<double>("eta");
            nr_impurities = config.get<int>("impurities"); 
            beta = config.get<double>("beta");
            grid_type = config.get<string>("grid");
            string gf_string = config.get<string>("gftype");  

            double delta = (to-from)/max(1,nmax-1);
            for (int i = 0;i < nmax;i++)
            {
               if (grid_type == "imaginary") omegas.emplace_back(0.,(2.0*i+1)*M_PI/beta);
               if (grid_type == "real") omegas.emplace_back(from+delta*i, eta);
            }

            ifstream ifs("wlist_sub.txt");
            if (ifs){
            omegas.clear() ; 
            string line;
            while (getline(ifs, line))
            {
             U val;
             istringstream(line) >> val;
             omegas.emplace_back(0.,val);
            }
           }

            if (gf_string == "symmetrized") gf_type = 1 ;
            if (gf_string == "nonsymmetrized") gf_type = 2 ;
         } 


        bool run(TaskDAG& dag, const Arena& arena)
        {
         int nirreps = 1;

         const auto& occ = this->template get<MOSpace<U>>("occ");
         const auto& vrt = this->template get<MOSpace<U>>("vrt");

         const auto& H = this->template get<TwoElectronOperator<U>>("H");

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
         nI = occ.nalpha[0];
         ni = occ.nbeta[0];
         nA = vrt.nalpha[0];
         na = vrt.nbeta[0];
 
         int maxspin = (nI == ni) ? 1 : 2 ;

         norb = nI+nA; 
 
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
       U piinverse = 1.0/M_PI ;

/* remove all self-energy and total Green's function files those are already there in the directory..
 */  
      if (arena.rank == 0)
      {
       std::ifstream gfile("gf_total.txt");
       if (gfile) remove("gf_total.txt");

       std::ifstream specfile("spectral_fn.txt");
       if (specfile) remove("spectral_fn.txt");

       std::ifstream sigomega("sigma_total.txt");
       if (sigomega) remove("sigma_total.txt");

       for (int nspin = 0; nspin < maxspin ; nspin++){   
        for (int i = 0;i < nr_impurities;i++){
          std::stringstream stream;
          stream << "spectral_fn_"<<nspin<<"_"<<i<< ".txt";
          std::string fileName = stream.str();
          std::ifstream gfile(fileName.c_str());
          if (gfile) remove(fileName.c_str());
        } 
       }

       for (int nspin = 0; nspin < maxspin ; nspin++){  
        for (int i = 0;i < nr_impurities;i++){
          std::stringstream stream;
          stream << "self_energy_"<<nspin<<"_"<<i<< ".txt";
          std::string fileName = stream.str();
          std::ifstream sigmafile(fileName.c_str());
          if (sigmafile) remove(fileName.c_str());
        } 
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

       vector<U> density(norb*norb,0.) ;
       vector<vector<CU>> sigma(nmax,vector<CU>(norb*norb, {0.,0.})) ;
       vector<vector<CU>> gf_imp(nmax,vector<CU>(nr_impurities*nr_impurities, {0.,0.})) ;
       U energy = 0. ; 
 
       for (int nspin = 0;nspin < maxspin;nspin++){
         for (int i = 0;i < nirreps;i++){
            vector<int> irreps = {i,i};
            (nspin == 0) ? cA_.getAllData(irreps, cA[i]) : ca_.getAllData(irreps, ca[i]);
            assert(cA[i].size() == N[i]*nA[i]);
            assert(ca[i].size() == N[i]*na[i]);
            (nspin == 0) ? cI_.getAllData(irreps, cI[i]) : ci_.getAllData(irreps, ci[i]);
            assert(cI[i].size() == N[i]*nI[i]);
            assert(ci[i].size() == N[i]*ni[i]);
         }        
     c_full.clear() ;  
    for (int i = 0;i < nirreps;i++){
      (nspin == 0) ? c_full.insert(c_full.begin(),cI[i].begin(),cI[i].end()) : c_full.insert(c_full.begin(),ci[i].begin(),ci[i].end());
      (nspin == 0) ? c_full.insert(c_full.end(),cA[i].begin(),cA[i].end()) : c_full.insert(c_full.end(),ca[i].begin(),ca[i].end());
    }

      fock.resize(norb*norb) ;

      FMI({nspin,0},{0,nspin})({0,0}).getAllData(fockoo); 
      FME({0,nspin},{nspin,0}).getAllData({0,0},fockov);
      FAE({nspin,0},{nspin,0}).getAllData({0,0},fockvv);

      int nocc = (nspin == 0 ? occ.nalpha[0] : occ.nbeta[0]) ;
      int nvirt = (nspin == 0 ? vrt.nalpha[0] : vrt.nbeta[0]) ;

      for (int i=0 ; i < norb ; i++){
       for (int j=0 ; j < norb ; j++){
        if ((i < nocc) && (j < nocc)) fock[i*norb+j] = fockoo[i*nocc+j] ; 
        if ((i < nocc) && (j >= nocc)) fock[i*norb+j] = fockov[i*nvirt+(j-nocc)] ; 
//      if ((i >= nocc) && (j < nocc)) fock[i*norb+j] = 1.0*fockov[(i-nocc)*nocc+j] ; 
        if ((i >= nocc) && (j >= nocc)) fock[i*norb+j] = fockvv[(i-nocc)*nvirt+(j-nocc)] ; 
        fock[j*norb+i] = fock[i*norb+j] ;
       }
      } 

    for (int omega = 0; omega < omegas.size() ;omega++)
    {
      vector<CU> gf_imp(nr_impurities*nr_impurities,{0.,0.}) ;
      vector<CU> gf_tmp(norb*norb,{0.,0.}) ;
      vector<CU> G_inv(nr_impurities*nr_impurities) ;

    /* calculate total Green's function
     */ 
        recalculate_gf(arena, 0., omega, gf_tmp) ;

    if (nr_impurities > 0){
    
       moao_transform_gf(gf_tmp, c_full, gf_imp) ; 

   /*  store impurity Green's function in a file
    */

    if (arena.rank == 0){
      for (int i = 0; i < nr_impurities ; i++){
         for (int j = 0; j < nr_impurities ; j++){
	      std::ofstream gomega;
	      gomega.open ("gf_total.txt", ofstream::out|std::ios::app);
	      gomega << setprecision(12) << gf_imp[i*nr_impurities+j] << std::endl ; 
	      gomega.close();
         }
      }
    }

/* spectral function to a file
 */

    if (arena.rank == 0){
     if (grid_type == "real"){
        for (int i = 0; i < nr_impurities ; i++){
          std::stringstream stream;
           stream << "spectral_fn_"<<nspin<<"_"<<i<< ".txt";
           std::string fileName = stream.str();
           std::ofstream gspec;
           gspec.open (fileName.c_str(), ofstream::out|std::ios::app);
           gspec << omegas[omega].real() << " " << -piinverse*gf_imp[i*nr_impurities+i].imag() << std::endl ;
          gspec.close();
        }
     }
    }
   }

/* calculate self-energy
 */ 
       if (nr_impurities == 0)
       {
         calculate_sigma(0., omega, gf_tmp, sigma[omega]) ;

        if (arena.rank == 0){
         if (grid_type == "real"){
	      std::ofstream specdensity;
	      specdensity.open ("spectral_fn.txt", ofstream::out|std::ios::app);
              U value = 0. ;
              for (int j = 0; j < norb ; j++)
              {
                value += gf_tmp[j*nr_impurities+j].imag() ;
              }            
	      specdensity << setprecision(12) << omegas[omega].real() << -piinverse*value << std::endl ; 
	      specdensity.close();
         } 
        }  
       }
    } 
   }

   if (arena.rank == 0){
    if (nr_impurities == 0){
      for (int i = 0; i < norb ; i++){
         for (int j = 0; j < norb ; j++){
	    std::ofstream sigomega;
	    sigomega.open ("sigma_total.txt", ofstream::out|std::ios::app);
	    sigomega << setprecision(12) << sigma[0][i*norb+j] << std::endl ; 
	    sigomega.close();
         }
      }
    }
   }

   /* bisection starts here
    */

    if (nr_impurities == 0){

      U thrs = 1.e-5 ;
      U mu = 0. ;

//     bisection_HF(mu) ; 

       energy = 0. ; 
 
       vector<vector<CU>> gf_final(nmax,vector<CU>(norb*norb)) ;

       bisection(arena, mu, sigma, gf_final, density) ;

       for (int omega = 0; omega < nmax ; omega++){
        sigma[omega].clear ();
        calculate_sigma(0., omega, gf_final[omega], sigma[omega]) ;
        energy += E2b(sigma[omega], gf_final[omega]) ; 
       }
       this->log(arena) << "Tr(Sigma.G) energy: " << (2.0/beta)*energy  << endl ;

       energy *= (2.0/beta) ;

       energy += (1.0/beta)*E2b_high_frequency(sigma[nmax-1]) ;

       this->log(arena) << "high frequency tail: " << E2b_high_frequency(sigma[nmax-1])/beta << endl;
       this->log(arena) << "total 2b energy: " << energy << endl ;

      vector<U> density_ao(norb*norb,0.) ;
      moao_transform_gf(density, c_full, density_ao) ; 

      if (arena.rank == 0) {
      std::ofstream dens_ao;
      dens_ao.open("coeff.txt", ofstream::out);

      for (int p = 0; p < norb ;p++){
       for (int q = 0; q < norb ;q++){
          dens_ao << setprecision(10) << density_ao[p*norb + q] << std::endl ; 
       }
      }
      dens_ao.close() ;
      }

      U value = 0. ; 
      if (arena.rank == 0) {
      vector<U> l(norb*norb);
      vector<CU> s_tmp(norb);
      vector<U> vr_tmp(norb*norb);

      int info = geev('N', 'V', norb, density.data(), norb,
                  s_tmp.data(), l.data(), norb,
                  vr_tmp.data(), norb);
      if (info != 0) throw runtime_error(str("check diagonalization: Info in geev: %d", info));

      this->log(arena)<<" #orbital occupation" <<endl ;
      for (int i=0 ; i < norb ; i++){
          cout << setprecision(10) << 2.0*s_tmp[i].real() << endl;
          value += density[i*norb+i];
      }}

      this->log(arena)<<"total occupancy: " << 2.0*value << endl;
    }
      return true;
   }

   void calculate_density (int omega, vector<CU> &gf_original, vector<U> &density){
   
     for (int p = 0; p < norb ;p++){
        for (int q = 0; q < norb ;q++){
           if (p == q) {
              density[p*norb+q] += (2.0/beta)*(gf_original[p*norb+q].real()-1.0/omegas[omega]+(fock[p*norb+q]/pow(omegas[omega].imag(),2))).real() ;
           }else{
              density[p*norb+q] += (2.0/beta)*(gf_original[p*norb+q].real()+(fock[p*norb+q]/pow(omegas[omega].imag(),2))) ;
           }
        }
      }
   }

    void add_density_high_frequency_tail (vector<U> &density){
      for (int p = 0; p < norb ;p++){
        for (int q = 0; q < norb ;q++){
          if (p==q) density[p*norb+q] += 0.5;
          density[p*norb+q] -= fock[p*norb+q]*(beta/4.);
        }
      }
    } 

    U trace(vector<U> &density){
     U value = 0. ;
     for (int i=0 ; i < norb ; i++){
         value += density[i*norb+i];
      }
     return value ;
    }

    void recalculate_gf(U mu, int omega, vector<CU> &gf,  const vector<CU> &sigma){
    
      vector<int> ipiv(norb) ;

//    for (int i = 0 ; i < norb*norb ; i++){
//     gf [norb*norb] = {0.,0.} ; 
//    }
      calculate_gf_zero_inv(mu, omega, gf) ;

      axpy (norb*norb, -1.0, sigma.data(), 1, gf.data(), 1);

      getrf(norb, norb, gf.data(), norb, ipiv.data()) ;

      getri(norb, gf.data(), norb, ipiv.data()) ;
    } 


    void recalculate_gf(const Arena &arena, U mu, int omega, vector<CU> &gf)
    {
      vector<CU> gf_ip_temp(norb*(norb+1)/2) ;
      vector<CU> gf_ea_temp(norb*(norb+1)/2) ;

      continued_fraction_ip (mu, omega, gf_ip_temp) ; 

      continued_fraction_ea (mu, omega, gf_ea_temp) ; 

      calculate_total_gf(arena, gf_ip_temp, gf_ea_temp, gf) ; 
    } 

    void continued_fraction_ip (U mu, int omega, vector<CU> &gf)  
    {

         auto& alpha_ip = this->template get<vector<vector<U>>>("alpha_ip");
         auto& beta_ip = this->template get<vector<vector<U>>>("beta_ip");
         auto& gamma_ip = this->template get<vector<vector<U>>>("gamma_ip");
         auto& norm_ip = this->template get<vector<U>>("norm_ip");

         int nvec_lanczos = alpha_ip[0].size() ;

         CU alpha_temp ;
         CU beta_temp ;
         CU gamma_temp ;
         CU com_one(1.,0.) ;
         CU value, value1 ;
         CU mucomplex = {mu, 0.} ;

         for (int p = 0; p < alpha_ip.size() ;p++){
              value  = {0.,0.} ;
              value1 = {0.,0.} ;

             for(int i=(nvec_lanczos-1);i >= 0;i--){  
              alpha_temp = {alpha_ip[p][i],0.} ;
              beta_temp  = {beta_ip[p][i],0.} ;
              gamma_temp = {gamma_ip[p][i],0.} ;

              value = (com_one)/(omegas[omega] + mucomplex + alpha_temp - beta_temp*gamma_temp*value1) ;                 
              value1 = value ;
             }
              gf[p] = value*norm_ip[p]  ;
         }

    }

    void continued_fraction_ea (U mu, int omega, vector<CU> &gf)  
    {
         auto& alpha_ea = this->template get<vector<vector<U>>>("alpha_ea");
         auto& beta_ea = this->template get<vector<vector<U>>>("beta_ea");
         auto& gamma_ea = this->template get<vector<vector<U>>>("gamma_ea");
         auto& norm_ea = this->template get<vector<U>>("norm_ea");

         int nvec_lanczos = alpha_ea[0].size() ;

         CU alpha_temp ;
         CU beta_temp ;
         CU gamma_temp ;
         CU com_one(1.,0.) ;
         CU value, value1 ;
         CU mucomplex = {mu, 0.} ;

         for (int p = 0; p < alpha_ea.size() ;p++){

             value  = {0.,0.} ;
             value1 = {0.,0.} ;

             for(int i=(nvec_lanczos-1);i >= 0;i--){  
              alpha_temp = {alpha_ea[p][i],0.} ;
              beta_temp  = {beta_ea[p][i],0.} ;
              gamma_temp = {gamma_ea[p][i],0.} ;

              value = (com_one)/(omegas[omega] + mucomplex - alpha_temp - beta_temp*gamma_temp*value1) ;                 
              value1 = value ;   
              }
              gf[p] = value*norm_ea[p] ;
          }
    }

    void calculate_sigma(U mu, int omega, vector<CU> &gf_total, vector<CU> &sigma)
    {
      vector<int> ipiv(norb,0) ;

   /* calculate inverse of zeroth order Green's function
    */ 
      vector<CU> gf_inv(norb*norb, {0.,0.}) ;

      copy (norb*norb, gf_total.data(),1, gf_inv.data(),1) ;

      calculate_gf_zero_inv( mu, omega, sigma) ;

      getrf(norb, norb, gf_inv.data(), norb, ipiv.data()) ;

      getri(norb, gf_inv.data(), norb, ipiv.data()) ;

      axpy (norb*norb, -1.0, gf_inv.data(), 1, sigma.data(), 1);
    }

    void calculate_gf_zero_inv(U mu, int omega, vector<CU> &gf_inv)
    {
      for (int p = 0; p < norb ;p++){
        for (int q = 0; q < norb ;q++){
         if (q == p){
           gf_inv[p*norb+q] = (omegas[omega] + mu - fock[p*norb+q]) ;
          }
          else
          { 
            gf_inv[p*norb+q] = -fock[p*norb+q] ;
          }
        }
      } 
    }

   void calculate_total_gf(const Arena &arena, vector<CU> &gf_ip_temp, vector<CU> &gf_ea_temp, vector<CU> &gf_total) {
    
     int tot_size = norb*(norb+1)/2  ;

     if (gf_type == 1){
      for (int p = 0; p < norb ;p++){
        for (int q = p; q < norb ;q++){
         if (q == p){
           gf_total[p*norb+q] = gf_ip_temp[(p*norb+q)-p*(p+1)/2] + gf_ea_temp[(p*norb+q)-p*(p+1)/2] ; 
          }
          else{
            gf_total[p*norb+q]  = 0.5*( gf_ip_temp[(p*norb+q)-p*(p+1)/2] - gf_ip_temp[(p*norb+p)-p*(p+1)/2] - gf_ip_temp[(q*norb+q)-q*(q+1)/2]) ; 
            gf_total[p*norb+q] += 0.5*( gf_ea_temp[(p*norb+q)-p*(p+1)/2] - gf_ea_temp[(p*norb+p)-p*(p+1)/2] - gf_ea_temp[(q*norb+q)-q*(q+1)/2]) ; 
          }
// symmetrize gf_total   
            gf_total[q*norb+p] = gf_total[p*norb+q] ; 
        }
      } 
     }
     else {
      for (int p = 0; p < norb ;p++){
        for (int q = 0; q < norb ;q++){
          gf_total[p*norb+q] = gf_ip_temp[(p*norb+q)-p*(p+1)/2] + gf_ea_temp[(p*norb+q)-p*(p+1)/2] ; 
          gf_total[q*norb+p] = gf_total[p*norb+q] ; 
        }
      } 
     }
    } 

   void bisection(const Arena &arena, U mu, vector<vector<CU>> &sigma, vector<vector<CU>> &gf, vector<U> &density) 
   {
    int nelec = ni + nI;
    int nspin = 0. ;
    U threshold = 1.e-7 ;
    U mu_min=-3., mu_max=3.;
    U mu_lower, mu_upper ;

    mu_lower = mu_min ; 
    mu_upper = mu_max ; 

    do
    {
      mu=mu_lower+(mu_upper-mu_lower)/2.;
     for (int p = 0; p < norb*norb ;p++){
      density[p] = 0. ;
     }

     for (int omega = 0; omega < nmax ; omega++)
     {  
        recalculate_gf( mu, omega, gf[omega], sigma[omega]) ;
  //    recalculate_gf( arena, mu, omega, gf[omega]) ;
    
      calculate_density(omega,gf[omega],density) ;
     }

      add_density_high_frequency_tail (density) ;
      
      if (arena.rank == 0) cout << "print trace "<< setprecision(8) << 2.0*trace(density)<< endl ;
      if (2.0*trace(density) > nelec){
       mu_upper = mu ;
       }else
       {
        mu_lower = mu ;
       } 
    } while(abs(nelec-2.0*trace(density)) > threshold); 
  
    if (arena.rank == 0){ 
     cout << "bisection has converged" << endl ;
     cout << "chemical potential: " << mu << endl ;
     cout << "total number of electrons: " << 2.0*trace(density) << endl ;} 
   } 

   void bisection_HF(U mu) 
   {
    int nelec = ni + nI;
    int nspin = 0. ;
    U threshold = 1.e-5 ;
    U mu_min=-3., mu_max=3.;
    U mu_lower, mu_upper ;

    mu_lower = mu_min ; 
    mu_upper = mu_max ; 

    vector<U> density(norb*norb) ;
    do
    {
      mu=mu_lower+(mu_upper-mu_lower)/2.;
     for (int p = 0; p < norb*norb ;p++)
     {
      density[p] = 0. ;
     }

     for (int omega = 0; omega < nmax ; omega++)
     {  

       vector<CU> gf_inv(norb*norb, {0.,0.}) ;
       vector<int> ipiv(norb,0) ;

       calculate_gf_zero_inv( mu, omega, gf_inv) ;

       getrf(norb, norb, gf_inv.data(), norb, ipiv.data()) ;

       getri(norb, gf_inv.data(), norb, ipiv.data()) ;
    
      calculate_density(omega,gf_inv,density) ;
     }

      add_density_high_frequency_tail (density) ;
      
      cout << "print trace "<< setprecision(8) << 2.0*trace(density)<< endl ;
      if (2.0*trace(density) > nelec){
       mu_upper = mu ;
       }else
       {
        mu_lower = mu ;
       } 
    } while(abs(nelec-2.0*trace(density)) > threshold); 
    
     cout << "bisection has converged" << endl ;
     cout << "chemical potential: " << mu << endl ;
     cout << "total number of electrons: " << 2.0*trace(density) << endl ;
   } 

   U E2b(vector<CU> &sigma, vector<CU> &gf_original)
   {
      U twob_energy = 0. ;
      for (int p = 0; p < norb ;p++) 
       for (int q = 0; q < norb ;q++) 
        
           twob_energy +=(sigma[p*norb+q].real()*gf_original[p*norb+q].real() - sigma[p*norb+q].imag()*gf_original[p*norb+q].imag()) ;

      return twob_energy ;
   } 
      
   U E1b(const Arena& arena, const MOSpace<U>& occ, const MOSpace<U>& vrt, const TwoElectronOperator<U>& H, vector<U> &density)
   {
      auto& D = this->puttmp("D", new OneElectronOperator<U>("D", arena, occ, vrt));
      auto& DIA = D.getIA();
      auto& DAI = D.getAI();
      auto& DAB = D.getAB();
      auto& DIJ = D.getIJ();

      U energy_1b ; 

      vector<tkv_pair<U>> pairs;

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

       energy_1b = 0.5*scalar(XI["ii"]*delta["ii"]) ; 

       printf("additional contribution to Fock, 1-body energy contribution: %.15f\n", -energy_1b);

      for (int p = 0; p < norb ;p++)
      {
      for (int q = 0; q < norb ;q++)
       {
         energy_1b -= 2.0*density[p*norb+q]*fock[p*norb+q] ; 
       }
      }
       printf("Total 1-body energy contribution: %.15f\n", -energy_1b);

      return energy_1b ;
   } 

   U E2b_high_frequency(vector<CU> &sigma)
   {
      vector<CU> sigma_xx(norb*norb,0.0) ;
 
      for (int p = 0; p < norb ;p++)
      {
        for (int q = 0; q < norb ;q++)
        {
          sigma_xx[p*norb+q] = -1.0*sigma[p*norb+q].imag()*omegas[nmax-1].imag() ; 
        }
      }

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

     return e2b_hf ;
   } 

   void bare_gf(int omega, vector<CU> G_inv)
   {
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
   }


    void moao_transform_gf(vector<CU>& g_mo, vector<U>& c_mo, vector<CU>& g_imp)
    {
       vector<U> g_mo_real (norb*norb, 0.) ;
       vector<U> g_mo_imag (norb*norb,0.) ;
       vector<U> g_imp_real(nr_impurities*nr_impurities, 0.) ;
       vector<U> g_imp_imag(nr_impurities*nr_impurities, 0.) ;
       vector<U> c_small(nr_impurities*norb, 0.) ;

   /*  Extract in Row major  a section of c_mo array
    */

       for (int i = 0; i < norb ; i++)
       {
         for (int j = 0; j < nr_impurities ; j++)
         {
            c_small [i*nr_impurities+j] = c_mo[i*norb+j] ; 
         } 
       }

       for (int i = 0; i < norb ; i++)
       {
         for (int j = 0; j < norb ; j++)
         {
            g_mo_real [i*norb+j] = g_mo[i*norb+j].real() ; 
            g_mo_imag [i*norb+j] = g_mo[i*norb+j].imag() ; 
         } 
       }

        vector<U> buf(norb*nr_impurities,0.) ;

        cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans, nr_impurities, norb, norb, 1.0, c_small.data(), nr_impurities, g_mo_real.data(), norb, 0.0, buf.data(), norb);
        cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, nr_impurities, nr_impurities, norb, 1.0, buf.data(), norb, c_small.data(), nr_impurities, 1.0, g_imp_real.data(), nr_impurities);

        buf.clear() ; 

        cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans, nr_impurities, norb, norb, 1.0, c_small.data(), nr_impurities, g_mo_imag.data(), norb, 0.0, buf.data(), norb);
        cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, nr_impurities, nr_impurities, norb, 1.0, buf.data(), norb, c_small.data(), nr_impurities, 1.0, g_imp_imag.data(), nr_impurities);

    /*combine real and imaginary part to produce total complex matrix
     */ 

       for (int i = 0; i < nr_impurities ; i++)
       {
         for (int j = 0; j < nr_impurities ; j++)
         {
            g_imp [i*nr_impurities+j] = {g_imp_real[i*nr_impurities+j],g_imp_imag[i*nr_impurities+j]} ; 
         } 
       }

    }

    void moao_transform_gf(vector<U>& g_mo, vector<U>& c_mo, vector<U>& g_imp)
    {
         vector<U> buf(norb*norb,0.) ;

          cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans, norb, norb, norb, 1.0, c_mo.data(), norb, g_mo.data(), norb, 0.0, buf.data(), norb);
          cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, norb, norb, norb, 1.0, buf.data(), norb, c_mo.data(), norb, 1.0, g_imp.data(), norb);

       /* feed in fortran order
        */
//         gemm('T', 'N', norb, norb, norb, 1.0, c_mo.data(), norb, g_mo.data(), norb, 1.0, buf.data(), norb);
//         gemm('N', 'N', norb, norb, norb, 1.0, buf.data(), norb, c_mo.data(), norb, 1.0, g_imp.data(), norb);
    }
  };

}
}

static const char* spec = R"(

    npoint int,
    omega_min double ?
        double -10.0,
    omega_max double ?
        double 10.0,
    impurities ?
        int 0,
    eta ?
        double .001,
    beta ?
        double 100, 
    grid?
        enum{ imaginary, real },
    gftype?
        enum{ symmetrized, nonsymmetrized },
)";

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::CCSDSIGMA);
REGISTER_TASK(aquarius::cc::CCSDSIGMA<double>, "ccsdsigma",spec);
