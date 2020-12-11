#include "util/global.hpp"

#include "task/task.hpp"
#include "operator/2eoperator.hpp"
#include "operator/space.hpp"
#include "operator/st2eoperator.hpp"
#include "operator/excitationoperator.hpp"
#include "hubbard/uhf_modelH.hpp"
#include "AIM/uhf_modelH.hpp"
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
        int maxspin = 2 ;
        vector<vector<CU>> omegas;
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

            omegas.resize(maxspin) ; 

            double delta = (to-from)/max(1,nmax-1);
            for (int s = 0;s < maxspin;s++){
            for (int i = 0;i < nmax;i++)
            {
               if (grid_type == "imaginary") omegas[s].emplace_back(0.,(2.0*i+1)*M_PI/beta);
               if (grid_type == "real") omegas[s].emplace_back(from+delta*i, eta);
             }
            }

            ifstream ifsa("wlist_sub_0.txt");
            if (ifsa){
            omegas[0].clear() ; 
            string line;
            while (getline(ifsa, line))
            {
             U val;
             istringstream(line) >> val;
             omegas[0].emplace_back(0.,val);
            }
           }

            ifstream ifsb("wlist_sub_1.txt");
            if (ifsb){
            omegas[1].clear() ; 
            string line;
            while (getline(ifsb, line))
            {
             U val;
             istringstream(line) >> val;
             omegas[1].emplace_back(0.,val);
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
 
//         int maxspin = (nI == ni) ? 1 : 2 ;
//         int maxspin = 2 ;

         norb = nI+nA; 
 
         for (int i = 0;i < nirreps;i++){
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

       U piinverse = 1.0/M_PI ;
      if (arena.rank == 0){
       std::ifstream gfile("gf_total.txt");
       if (gfile) remove("gf_total.txt");
      }


      if (arena.rank == 0){
       std::ifstream gmofile("gf_mo.txt");
       if (gmofile) remove("gf_mo.txt");
      }

       vector<vector<vector<CU>>> sigma(maxspin,vector<vector<CU>>(nmax,vector<CU>(norb*norb, {0.,0.}))) ;
       U energy = 0. ; 
 
      for (int nspin = 0;nspin < maxspin;nspin++){

        vector<vector<CU>> gf_imp(nmax,vector<CU>(nr_impurities*nr_impurities, {0.,0.})) ;
        vector<U> fock;
        vector<U> density(norb*norb,0.) ;

        vector<U> c_full ; 
        vector<U> fockoo;
        vector<U> fockvv;
        vector<U> fockov;

        int other = (nspin == 0 ? 1 : 0) ;

        for (int i = 0;i < nirreps;i++){
           (nspin == 0) ? c_full.insert(c_full.begin(),cI[i].begin(),cI[i].end()) : c_full.insert(c_full.begin(),ci[i].begin(),ci[i].end());
           (nspin == 0) ? c_full.insert(c_full.end(),cA[i].begin(),cA[i].end()) : c_full.insert(c_full.end(),ca[i].begin(),ca[i].end());
        } 

          fock.resize(norb*norb) ;

          FMI({0,other},{0,other})({0,0}).getAllData(fockoo); 
          FME({0,other},{other,0}).getAllData({0,0},fockov);
          FAE({other,0},{other,0}).getAllData({0,0},fockvv);

        int nocc = (nspin == 0 ? occ.nalpha[0] : occ.nbeta[0]) ;
        int nvirt = (nspin == 0 ? vrt.nalpha[0] : vrt.nbeta[0]) ;

        for (int i=0 ; i < norb ; i++){
         for (int j=0 ; j < norb ; j++){
          if ((i < nocc) && (j < nocc))   fock[i*norb+j] = fockoo[i*nocc+j] ; 
          if ((i < nocc) && (j >= nocc))  fock[i*norb+j] = fockov[i*nvirt+(j-nocc)] ; 
          if ((i >= nocc) && (j >= nocc)) fock[i*norb+j] = fockvv[(i-nocc)*nvirt+(j-nocc)] ; 
          fock[j*norb+i] = fock[i*norb+j] ;
         }
        } 


    U mu_HF = 0. ;

    if (nr_impurities == 0){

        bisection_HF(mu_HF, nspin, fock) ; 
    }


      std::ofstream gomega;
      gomega.open ("gf_total.txt", ofstream::out|std::ios::app);


      std::ofstream gmo;
      gmo.open ("gf_mo.txt", ofstream::out|std::ios::app);


      for (int omega = 0; omega < omegas[nspin].size() ;omega++){
      
       vector<CU> gf_imp(nr_impurities*nr_impurities,{0.,0.}) ;
       vector<CU> gf_tmp(norb*norb,{0.,0.}) ;

    /* calculate total Green's function
     */ 
       recalculate_gf(arena, nspin, 0., omega, gf_tmp) ;
          
       if (nr_impurities == 0){
        calculate_sigma(mu_HF, omegas[nspin][omega], fock, gf_tmp, sigma[nspin][omega]) ;
       } 
       if (nr_impurities > 0){
    
       moao_transform_gf(gf_tmp, c_full, gf_imp) ; 

   /*  store impurity Green's function in a file
    */

       if (arena.rank == 0){
        for (int i = 0; i < nr_impurities ; i++){
         for (int j = 0; j < nr_impurities ; j++){
	      gomega << setprecision(12) << gf_imp[i*nr_impurities+j] << std::endl ; 
         }}
       }

       if (arena.rank == 0){
        for (int i = 0; i < norb ; i++){
         for (int j = 0; j < norb ; j++){
             gmo << setprecision(12) << gf_tmp[i*norb+j].real() << " " << gf_tmp[i*norb+j].imag() << std::endl ;
         }}
       }

      }
     } 

      gomega.close();
      gmo.close() ;
   /* bisection starts here
    */

    if (nr_impurities == 0){

       U thrs = 1.e-5 ;
       U mu = 0. ;

         energy = 0. ; 

         vector<vector<CU>> gf_final(nmax,vector<CU>(norb*norb)) ;
         bisection(arena, nspin, mu, fock, sigma[nspin], gf_final, density) ;

         for (int omega = 0; omega < nmax ; omega++){
          sigma[nspin][omega].clear() ;
          calculate_sigma(mu, omegas[nspin][omega], fock, gf_final[omega], sigma[nspin][omega]) ;
          energy += E2b(sigma[nspin][omega], gf_final[omega]) ; 
         }
       }
       this->log(arena) << "Tr(Sigma.G) energy: " << (1.0/beta)*energy  << endl ;

       energy *= (1.0/beta) ;

       energy += (1.0/beta)*E2b_high_frequency(nspin,sigma[nspin][nmax-1]) ;

       this->log(arena) << "high frequency tail: " << E2b_high_frequency(nspin,sigma[nspin][nmax-1])/beta << endl;
       this->log(arena) << "total 2b energy: " << energy << endl ;

      vector<U> density_ao(norb*norb,0.) ;
        moao_transform_gf(density, c_full, density_ao) ; 

      if (arena.rank == 0) {
      std::ofstream dens_ao;
      dens_ao.open("density.txt", ofstream::out);

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
          cout << setprecision(10) << s_tmp[i].real() << endl;
          value += density[i*norb+i];
      }}

      this->log(arena)<<"total occupancy: " << value << endl;
    }
      return true;
   }

   void calculate_density (CU omega, vector<U> &fock, vector<CU> &gf_original, vector<U> &density){
   
     for (int p = 0; p < norb ;p++){
        for (int q = 0; q < norb ;q++){
           if (p == q) {
              density[p*norb+q] += (2.0/beta)*(gf_original[p*norb+q].real()-1.0/omega+(fock[p*norb+q]/pow(omega.imag(),2))).real() ;
           }else{
              density[p*norb+q] += (2.0/beta)*(gf_original[p*norb+q].real()+(fock[p*norb+q]/pow(omega.imag(),2))) ;
           }
        }
      }
   }

    void add_density_high_frequency_tail (vector<U> &fock, vector<U> &density){
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

    void recalculate_gf(U mu, CU omega, vector<U> &fock, vector<CU> &gf,  const vector<CU> &sigma){
    
      vector<int> ipiv(norb) ;

//    for (int i = 0 ; i < norb*norb ; i++){
//     gf [norb*norb] = {0.,0.} ; 
//    }
      calculate_gf_zero_inv(fock, mu, omega, gf) ;

      axpy (norb*norb, -1.0, sigma.data(), 1, gf.data(), 1);

      getrf(norb, norb, gf.data(), norb, ipiv.data()) ;

      getri(norb, gf.data(), norb, ipiv.data()) ;
    } 

    void recalculate_gf(const Arena &arena, int nspin, U mu, int omega, vector<CU> &gf)
    {
      vector<CU> gf_ip_temp(norb*(norb+1)/2) ;
      vector<CU> gf_ea_temp(norb*(norb+1)/2) ;

      continued_fraction_ip (mu,nspin,omega, gf_ip_temp) ; 

      continued_fraction_ea (mu,nspin,omega, gf_ea_temp) ; 

      calculate_total_gf(arena, gf_ip_temp, gf_ea_temp, gf) ; 
    } 

    void continued_fraction_ip (U mu, int nspin, int omega, vector<CU> &gf)  
    {
         auto& alpha_ip = this->template get<vector<vector<vector<U>>>>("alpha_ip");
         auto& beta_ip = this->template get<vector<vector<vector<U>>>>("beta_ip");
         auto& gamma_ip = this->template get<vector<vector<vector<U>>>>("gamma_ip");
         auto& norm_ip = this->template get<vector<vector<U>>>("norm_ip");

         int nvec_lanczos = alpha_ip[nspin][0].size() ;

         CU alpha_temp ;
         CU beta_temp ;
         CU gamma_temp ;
         CU com_one(1.,0.) ;
         CU value, value1 ;
         CU mucomplex = {mu, 0.} ;

         for (int p = 0; p < alpha_ip[nspin].size() ;p++){
              value  = {0.,0.} ;
              value1 = {0.,0.} ;

             for(int i=(alpha_ip[nspin][p].size()-1);i >= 0;i--){  
              alpha_temp = {alpha_ip[nspin][p][i],0.} ;
              beta_temp  = {beta_ip[nspin][p][i],0.} ;
              gamma_temp = {gamma_ip[nspin][p][i],0.} ;

              value = (com_one)/(omegas[nspin][omega] + mucomplex + alpha_temp - beta_temp*gamma_temp*value1) ;                 
              value1 = value ;
             }
              gf[p] = value*norm_ip[nspin][p]  ;
         }
    }

    void continued_fraction_ea (U mu, int nspin, int omega, vector<CU> &gf)  
    {
         auto& alpha_ea = this->template get<vector<vector<vector<U>>>>("alpha_ea");
         auto& beta_ea = this->template get<vector<vector<vector<U>>>>("beta_ea");
         auto& gamma_ea = this->template get<vector<vector<vector<U>>>>("gamma_ea");
         auto& norm_ea = this->template get<vector<vector<U>>>("norm_ea");

         CU alpha_temp ;
         CU beta_temp ;
         CU gamma_temp ;
         CU com_one(1.,0.) ;
         CU value, value1 ;
         CU mucomplex = {mu, 0.} ;

         for (int p = 0; p < alpha_ea[nspin].size() ;p++){

             value  = {0.,0.} ;
             value1 = {0.,0.} ;

             for(int i=(alpha_ea[nspin][p].size()-1);i >= 0;i--){  
              alpha_temp = {alpha_ea[nspin][p][i],0.} ;
              beta_temp  = {beta_ea[nspin][p][i],0.} ;
              gamma_temp = {gamma_ea[nspin][p][i],0.} ;

              value = (com_one)/(omegas[nspin][omega] + mucomplex - alpha_temp - beta_temp*gamma_temp*value1) ;                 
              value1 = value ;   
              }
              gf[p] = value*norm_ea[nspin][p] ;
          }
    }

    void calculate_sigma(U mu, CU omega, vector<U> &fock, vector<CU> &gf_total, vector<CU> &sigma)
    {
      vector<int> ipiv(norb,0) ;

   /* calculate inverse of zeroth order Green's function
    */ 
      vector<CU> gf_inv(norb*norb, {0.,0.}) ;

      copy (norb*norb, gf_total.data(),1, gf_inv.data(),1) ;

      calculate_gf_zero_inv(fock, mu, omega, sigma) ;

      getrf(norb, norb, gf_inv.data(), norb, ipiv.data()) ;

      getri(norb, gf_inv.data(), norb, ipiv.data()) ;

      axpy (norb*norb, -1.0, gf_inv.data(), 1, sigma.data(), 1);
    }

    void calculate_gf_zero_inv(vector<U> &fock, U mu, CU omega, vector<CU> &gf_inv){
    
      for (int p = 0; p < norb ;p++){
        for (int q = 0; q < norb ;q++){
         if (q == p){
           gf_inv[p*norb+q] = (omega + mu - fock[p*norb+q]) ;
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

   void bisection(const Arena &arena, int nspin, U mu, vector<U> &fock, vector<vector<CU>> &sigma, vector<vector<CU>> &gf, vector<U> &density) 
   {
    int nelec ;
    U threshold = 1.e-5 ;
    U mu_min=-3., mu_max=3.;
    U mu_lower, mu_upper ;

    mu_lower = mu_min ; 
    mu_upper = mu_max ; 

    nelec = (nspin == 0 ?  nI : ni) ;

    do
    {
      mu=mu_lower+(mu_upper-mu_lower)/2.;
     for (int p = 0; p < norb*norb ;p++){
      density[p] = 0. ;
     }

     for (int omega = 0; omega < omegas[nspin].size() ; omega++)
     {  
       recalculate_gf( mu, omegas[nspin][omega], fock, gf[omega], sigma[omega]) ;

//       recalculate_gf( arena,nspin, mu, omega, gf[omega]) ;
    
       calculate_density(omegas[nspin][omega],fock,gf[omega],density) ;
     }

      add_density_high_frequency_tail (fock,density) ;

      if (arena.rank == 0) cout << "print trace "<< setprecision(8) << trace(density)<< endl ;
      if (trace(density) > nelec){
       mu_upper = mu ;
       }else
       {
        mu_lower = mu ;
       } 
    } while(abs(nelec-trace(density)) > threshold); 
  
    if (arena.rank == 0){ 
     cout << "bisection has converged" << endl ;
     cout << "chemical potential: " << mu << endl ;
     cout << "total number of electrons: " << trace(density) << endl ;} 
   } 

   void bisection_HF(U mu, int nspin, vector<U> &fock) 
   {
    int nelec ;
    U threshold = 1.e-8 ;
    U mu_min=-3., mu_max=3.;
    U mu_lower, mu_upper ;

    mu_lower = mu_min ; 
    mu_upper = mu_max ; 

    nelec = (nspin == 0 ?  nI : ni) ;

    cout << "Nspin" << " " << nspin << " nelec " << nelec << endl ; 

    vector<U> density(norb*norb) ;
    do
    {
      mu=mu_lower+(mu_upper-mu_lower)/2.;
     for (int p = 0; p < norb*norb ;p++){
      density[p] = 0. ;
     }

     for (int omega = 0; omega < omegas[nspin].size() ; omega++)
     {  
       vector<CU> gf_inv(norb*norb, {0.,0.}) ;
       vector<int> ipiv(norb,0) ;

       calculate_gf_zero_inv(fock, mu, omegas[nspin][omega], gf_inv) ;

       getrf(norb, norb, gf_inv.data(), norb, ipiv.data()) ;

       getri(norb, gf_inv.data(), norb, ipiv.data()) ;
    
       calculate_density(omegas[nspin][omega], fock, gf_inv,density) ;
     }

      add_density_high_frequency_tail (fock, density) ;
      
      cout << "print trace "<< setprecision(8) << trace(density)<< endl ;
      if (trace(density) > nelec){
       mu_upper = mu ;
       }else
       {
        mu_lower = mu ;
       } 
    } while(abs(nelec-trace(density)) > threshold); 
    
     cout << "bisection has converged" << endl ;
     cout << "chemical potential: " << mu << endl ;
     cout << "total number of electrons: " << trace(density) << endl ;
   } 

   U E2b(vector<CU> &sigma, vector<CU> &gf_original){
   
      U twob_energy = 0. ;
      for (int p = 0; p < norb ;p++) 
       for (int q = 0; q < norb ;q++) 
           twob_energy +=(sigma[p*norb+q].real()*gf_original[p*norb+q].real() - sigma[p*norb+q].imag()*gf_original[p*norb+q].imag()) ;

      return twob_energy ;
   } 
      
   U E1b(const Arena& arena, const MOSpace<U>& occ, const MOSpace<U>& vrt, vector<U> &fock, const TwoElectronOperator<U>& H, vector<U> &density)
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

   U E2b_high_frequency(int nspin,vector<CU> &sigma)
   {
//      vector<CU> sigma_xx(norb*norb,0.0) ;
      vector<CU> sigma_xx(norb*norb,{0.,0.}) ;
 
      for (int p = 0; p < norb ;p++)
      {
        for (int q = 0; q < norb ;q++)
        {
          sigma_xx[p*norb+q] = -1.0*sigma[p*norb+q].imag()*omegas[nspin][nmax-1].imag() ; 
        }
      }

      U e2b_hf = 0. ;
      CU omega ;

      for (int p = 0; p < norb ;p++){
        for (int q = 0; q < norb ;q++){
         if (p==q) e2b_hf +=  (-beta/8.)*(sigma_xx[p*norb+q]).imag() ; 
        }
      }

      for (int w = nmax; w < 10000000 ;w++)
      {
        omega = {0.0,(2.0*w+1)*M_PI/beta} ;      
       for (int p = 0; p < norb ;p++){
         e2b_hf +=  1.0*(-(1.0/omega).imag()*(sigma_xx[p*norb+p]/omega).imag()) ; 
       }
      }

     return e2b_hf ;
   } 

   void bare_gf(CU omega, vector<CU> G_inv)
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
        Hybridization += (integral_diagonal[u*norb+b]*integral_diagonal[v*norb+b])/(omega - integral_diagonal[b*norb+b]) ;
       }
        if (v == u) 
        {
         G_inv[u*nr_impurities+v] = omega - integral_diagonal[u*norb+v] - Hybridization + 0.5*v_onsite[u]  ;
        }
        else
        {
         G_inv[u*nr_impurities+v] = - integral_diagonal[u*norb+v] -  Hybridization  ;
        } 
         gzero_omega << omega.imag() << " " << u << " " << v << " " << G_inv[u*nr_impurities+v].real() << " " << G_inv[u*nr_impurities+v].imag() << std::endl ; 
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

   /*  Extract in Row major a section of c_mo array
    */
       for (int i = 0; i < norb ; i++){
         for (int j = 0; j < nr_impurities ; j++){
            c_small [i*nr_impurities+j] = c_mo[i*norb+j] ; 
         } 
       }

       for (int i = 0; i < norb ; i++){
         for (int j = 0; j < norb ; j++){
            g_mo_real [i*norb+j] = g_mo[i*norb+j].real() ; 
            g_mo_imag [i*norb+j] = g_mo[i*norb+j].imag() ; 
         } 
       }

        vector<U> buf(norb*nr_impurities,0.) ;

        c_dgemm('T', 'N', nr_impurities, norb, norb, 1.0, c_small.data(), norb, g_mo_real.data(), norb, 0.0, buf.data(), nr_impurities);
        c_dgemm('N', 'N', nr_impurities, nr_impurities, norb, 1.0, buf.data(), norb, c_small.data(), norb, 1.0, g_imp_real.data(), nr_impurities);

        buf.clear() ; 

        c_dgemm('T', 'N', nr_impurities, norb, norb, 1.0, c_small.data(), norb, g_mo_imag.data(), norb, 0.0, buf.data(), nr_impurities);
        c_dgemm('N', 'N', nr_impurities, nr_impurities, norb, 1.0, buf.data(), norb, c_small.data(), norb, 1.0, g_imp_imag.data(), nr_impurities);


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

          c_dgemm('T', 'N', norb, norb, norb, 1.0, c_mo.data(), norb, g_mo.data(), norb, 0.0, buf.data(), norb);
          c_dgemm('N', 'N', norb, norb, norb, 1.0, buf.data(), norb, c_mo.data(), norb, 1.0, g_imp.data(), norb);

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
