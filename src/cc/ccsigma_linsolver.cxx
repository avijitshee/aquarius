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
class CCSDSIGMA_LINSOLVER: public Task 
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
        CCSDSIGMA_LINSOLVER(const string& name, Config& config): Task(name, config)
        {
            vector<Requirement> reqs;
            reqs.emplace_back("moints", "H");
            reqs.emplace_back("occspace", "occ");
            reqs.emplace_back("vrtspace", "vrt");
            reqs.emplace_back("Ea", "Ea");
            reqs.emplace_back("Eb", "Eb");
            reqs.emplace_back("Fa", "Fa");
            reqs.emplace_back("Fb", "Fb");

            reqs.emplace_back("cc.gf", "gf");

            this->addProduct(Product("cc.sigma_linsolver", "gf_total", reqs));

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


       ofstream ofile;
       ofile.open ("omega.txt");
       for (int i = 0; i < nmax ; i++){
              CU grid = {from+delta*i, eta} ;
	      ofile << setprecision(12) << grid << std::endl ; 
       }
       ofile.close();


            ifstream ifsa("wlist_sub.txt");
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

            ifstream ifsb("wlist_sub.txt");
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
      if (arena.rank == 0){
       std::ifstream cfile("coeff.txt");
       if (cfile) remove("coeff.txt");
      }


     auto& gf = this->template get<vector<vector<vector<CU>>>>("gf");

 
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

       if (arena.rank == 0){
        for (int i = 0; i < norb ; i++){
         for (int j = 0; j < norb ; j++){
	      std::ofstream cfile;
	      cfile.open ("coeff.txt", ofstream::out|std::ios::app);
	      cfile << nspin << " " <<  i << " " << j << " " << setprecision(12) << c_full[i*norb+j] << std::endl ; 
	      cfile.close();
         }}
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

      for (int omega = 0; omega < omegas[nspin].size() ;omega++){
      
       vector<CU> gf_imp(nr_impurities*nr_impurities,{0.,0.}) ;

      vector<int> ipiv(nr_impurities) ;

    /* calculate total Green's function
     */ 
          
       if (nr_impurities > 0){
    
       moao_transform_gf(gf[nspin][omega], c_full, gf_imp) ; 

       vector<CU> gf_inv(gf_imp) ;
       getrf(nr_impurities, nr_impurities, gf_inv.data(), norb, ipiv.data()) ;

       getri(nr_impurities, gf_inv.data(), nr_impurities, ipiv.data()) ;

   /*  store impurity Green's function in a file
    */

       if (arena.rank == 0){
        for (int i = 0; i < nr_impurities ; i++){
         for (int j = 0; j < nr_impurities ; j++){
	      std::ofstream gomega;
	      gomega.open ("gf_total.txt", ofstream::out|std::ios::app);
	      gomega << setprecision(12) << gf_imp[i*nr_impurities+j] << endl ; 
	      gomega.close();
         }}
       }
      }
        if (arena.rank == 0){
        for (int i = 0; i < norb ; i++){
         for (int j = 0; j < norb ; j++){
              std::ofstream gomega;
              gomega.open ("gf_mo.txt", ofstream::out|std::ios::app);
              gomega << nspin << " " << i << " " << j << " " << setprecision(12) << gf[nspin][omega][i*norb+j].real() << " " << gf[nspin][omega][i*norb+j].imag() << endl ; 
              gomega.close();
         }}}
     } 
    }
      return true;
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

    void moao_transform_gf(vector<CU>& g_mo, vector<U>& c_mo, vector<CU>& g_imp)
    {
       vector<U> g_mo_real (norb*norb, 0.) ;
       vector<U> g_mo_imag (norb*norb,0.) ;
       vector<U> g_imp_real(nr_impurities*nr_impurities, 0.) ;
       vector<U> g_imp_imag(nr_impurities*nr_impurities, 0.) ;
       vector<U> c_small(nr_impurities*norb, 0.) ;

   /*  Extract in Row major a section of c_mo array
    */
//     for (int i = 0; i < norb ; i++){
//       for (int j = 0; j < nr_impurities ; j++){
//          c_small [i*nr_impurities+j] = c_mo[i*norb+j] ; 
//       } 
//     }

   /* Extract in colum major a section of c_mo array
    */
       for (int i = 0; i < norb ; i++){
         for (int j = 0; j < nr_impurities ; j++){
            c_small [j*norb+i] = c_mo[i*norb+j] ; 
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
        c_dgemm('N', 'N', nr_impurities, nr_impurities, norb, 1.0, buf.data(), nr_impurities, c_small.data(), norb, 1.0, g_imp_real.data(), nr_impurities);

        buf.clear() ; 

        c_dgemm('T', 'N', nr_impurities, norb, norb, 1.0, c_small.data(), norb, g_mo_imag.data(), norb, 0.0, buf.data(), nr_impurities);
        c_dgemm('N', 'N', nr_impurities, nr_impurities, norb, 1.0, buf.data(), nr_impurities, c_small.data(), norb, 1.0, g_imp_imag.data(), nr_impurities);

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

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::CCSDSIGMA_LINSOLVER);
REGISTER_TASK(aquarius::cc::CCSDSIGMA_LINSOLVER<double>, "ccsdsigma_linsolver",spec);
