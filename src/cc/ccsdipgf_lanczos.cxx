#include "util/global.hpp"

#include "time/time.hpp"
#include "task/task.hpp"
#include "convergence/lanczos.hpp"
#include "util/iterative.hpp"
#include "operator/2eoperator.hpp"
#include "operator/st2eoperator.hpp"
#include "operator/excitationoperator.hpp"
#include "operator/deexcitationoperator.hpp"
#include "operator/denominator.hpp"

using namespace aquarius::tensor;
using namespace aquarius::task;
using namespace aquarius::input;
using namespace aquarius::op;
using namespace aquarius::convergence;
using namespace aquarius::symmetry;

namespace aquarius
{
namespace cc
{

template <typename U>
class CCSDIPGF_LANCZOS : public Iterative<U>
{
    protected:
        typedef U X ; 
        typedef complex_type_t<U> CU;
        Config lanczos_config;
        int orbital;
        int orbstart;
        int orbend;
        vector<CU> omegas;
        CU omega;
        vector<U> old_value ;
        string orb_range ;

    public:
        CCSDIPGF_LANCZOS(const string& name, Config& config)
        : Iterative<U>(name, config), lanczos_config(config.get("lanczos"))
        {
            vector<Requirement> reqs;
            reqs.emplace_back("ccsd.T", "T");
            reqs.emplace_back("ccsd.L", "L");
            reqs.emplace_back("ccsd.Hbar", "Hbar");
            this->addProduct(Product("ccsd.ipgf", "gf_ip", reqs));
            this->addProduct(Product("ccsd.ipalpha", "alpha_ip", reqs));
            this->addProduct(Product("ccsd.ipbeta", "beta_ip", reqs));
            this->addProduct(Product("ccsd.ipgamma", "gamma_ip", reqs));
            this->addProduct(Product("ccsd.ipnorm", "norm_ip", reqs));
            orbital = config.get<int>("orbital");
            double from = config.get<double>("omega_min");
            double to = config.get<double>("omega_max");
            int n = config.get<double>("npoint");
            double eta = config.get<double>("eta");
            double beta = config.get<double>("beta");
            string grid_type = config.get<string>("grid");
            orb_range = config.get<string>("orbital_range");

            double delta = (to-from)/max(1,n-1);
            for (int i = 0;i < n;i++)
            {
             if (grid_type == "real") omegas.emplace_back(from+delta*i, eta);
             if (grid_type == "imaginary") omegas.emplace_back(0.,(2.0*i+1)*M_PI/beta);
            }
        }

        bool run(TaskDAG& dag, const Arena& arena)
        {
            auto& H = this->template get<STTwoElectronOperator<U>>("Hbar");

            const PointGroup& group = H.getABIJ().getGroup();
            int nirrep = group.getNumIrreps();

            const Space& occ = H.occ;
            const Space& vrt = H.vrt;

            int nI = occ.nalpha[0];
            int ni = occ.nbeta[0];
            int nA = vrt.nalpha[0];
            int na = vrt.nbeta[0];
            int nvec_lanczos; 
            CU value ;
            CU value1 ;

            int maxspin = (nI == ni) ? 1 : 2 ;

            auto& T = this->template get<ExcitationOperator  <U,2>>("T");
            auto& L = this->template get<DeexcitationOperator<U,2>>("L");

           /* vector LL means Left Lanczos and RL means Right Lanczos
            */

            auto& gf_ip = this-> put("gf_ip", new vector<vector<vector<vector<CU>>>>) ;
            auto& alpha_ip = this-> put("alpha_ip", new vector<vector<U>>) ;
            auto& beta_ip = this-> put("beta_ip", new vector<vector<U>>) ;
            auto& gamma_ip = this-> put("gamma_ip", new vector<vector<U>>) ;
            auto& norm_ip = this-> put("norm_ip", new vector<U>) ;
            
            SpinorbitalTensor<U> Dij("D(ij)", arena, group, {vrt,occ}, {0,1}, {0,1});
            SpinorbitalTensor<U> Gijak("G(ij,ak)", arena, group, {vrt,occ}, {0,2}, {1,1});

            SpinorbitalTensor<U> Nij("N(ij)", arena, group, {vrt,occ}, {0,1}, {0,1});

            Dij["ij"]  =     L(1)["ie"  ]*T(1)["ej"  ];
            Dij["ij"] += 0.5*L(2)["imef"]*T(2)["efjm"];

            Gijak["ijak"] = L(2)["ijae"]*T(1)["ek"];

           if (orb_range == "full") 
           { orbstart = 0 ;
             orbend = nI + nA ;  
             gf_ip.resize(maxspin);
             alpha_ip.resize(orbend*orbend) ;
             beta_ip.resize(orbend*orbend) ;
             gamma_ip.resize(orbend*orbend) ;

            for (int nspin = 0;nspin < maxspin;nspin++)
             {
              gf_ip[nspin].resize(omegas.size());
             }  

            for (int nspin = 0;nspin < maxspin;nspin++)
             {
              for (int i = 0;i < omegas.size();i++)
               {
                gf_ip[nspin][i].resize(orbend);
               }
             }
             for (int nspin = 0;nspin < maxspin;nspin++)
             for (int i = 0;i < omegas.size();i++)
             for (int j = 0;j < orbend;j++)
             {
               gf_ip[nspin][i][j].resize(orbend);
             }

           } 

           if (orb_range == "diagonal") 
           { orbstart = orbital-1 ;
             orbend = orbital;  
             gf_ip.resize(maxspin);

             alpha_ip.resize(orbend*orbend) ;
             beta_ip.resize(orbend*orbend) ;
             gamma_ip.resize(orbend*orbend) ;

            for (int nspin = 0;nspin < maxspin;nspin++)
             {
              gf_ip[nspin].resize(omegas.size());
             }  
            for (int nspin = 0;nspin < maxspin;nspin++)
             {
              for (int i = 0;i < omegas.size();i++)
               {
                gf_ip[nspin][i].resize(1);
               }
             }
             for (int nspin = 0;nspin < maxspin;nspin++)
             for (int i = 0;i < omegas.size();i++)
             for (int j = 0;j < 1;j++)
             {
               gf_ip[nspin][i][j].resize(1);
             }
           } 

          vector<CU> spec_func(omegas.size()) ;

        for (int nspin = 0; nspin < maxspin ; nspin++)   
         {
         for (int orbleft = orbstart; orbleft < orbend ; orbleft++)   
          {
           for (int orbright = orbstart; orbright < orbend ; orbright++)   
            {
              old_value.clear() ;             

            printf("Computing Green's function element:  %d %d\n", orbleft, orbright ) ;

            bool isalpha_right = false;
            bool isvrt_right = false;
            bool isalpha_left = false;
            bool isvrt_left = false;

            int orbleft_dummy = orbleft ;
            int orbright_dummy = orbright ;

            if (nspin == 0)
            {
                isalpha_left = true;
                if ((orbleft ) >= nI)
                {
                    isvrt_left = true;
                    orbleft_dummy = orbleft - nI;
                }
            }
            else
            {
                if (orbleft >= ni)
                {
                    isvrt_left = true;
                    orbleft_dummy = orbleft - ni;
                }
            }

            if (nspin == 0)
            {
                isalpha_right = true;
                if ((orbright ) >= nI)
                {
                    isvrt_right = true;
                    orbright_dummy = orbright - nI;
                }
            }
            else
            {
                if (orbright >= ni)
                {
                    isvrt_right = true;
                    orbright_dummy = orbright - ni;
                }
            }

            auto& RL = this->puttmp("RL", new ExcitationOperator  <U,1,2>("RL", arena, occ, vrt, isalpha_right ? -1 : 1));
            auto& LL = this->puttmp("LL", new DeexcitationOperator  <U,1,2>("LL", arena, occ, vrt, isalpha_left ? 1 : -1));
            auto& Z = this->puttmp("Z", new ExcitationOperator  <U,1,2>("Z", arena, occ, vrt, isalpha_right ? -1 : 1));
            auto& Y = this->puttmp("Y", new DeexcitationOperator  <U,1,2>("Y", arena, occ, vrt, isalpha_left ? 1 : -1));
            auto& b  = this->puttmp("b",  new ExcitationOperator  <U,1,2>("b",  arena, occ, vrt, isalpha_right ? -1 : 1));
            auto& e  = this->puttmp("e",  new DeexcitationOperator<U,1,2>("e",  arena, occ, vrt, isalpha_left ? 1 : -1));

            auto& XE = this->puttmp("XE", new SpinorbitalTensor<U>("X(e)", arena, group, {vrt,occ}, {0,0}, {1,0}, isalpha_right ? -1 : 1));
            auto& XEA = this->puttmp("XEA", new SpinorbitalTensor<U>("X(ea)", arena, group, {vrt,occ}, {1,0}, {0,0}, isalpha_left ? 1 : -1));
            auto& alpha = this-> puttmp("alpha", new unique_vector<U>()) ;
            auto& beta  = this-> puttmp("beta", new unique_vector<U>()) ;
            auto& gamma = this-> puttmp("gamma", new unique_vector<U>());

            SpinorbitalTensor<U> ap ("ap"  , arena, group, {vrt,occ}, {0,0}, {isvrt_right, !isvrt_right}, isalpha_right ? -1 : 1);
            SpinorbitalTensor<U> apt("ap^t", arena, group, {vrt,occ}, {isvrt_left, !isvrt_left}, {0,0}, isalpha_left ? 1 : -1);

            vector<tkv_pair<U>> pair_left{{orbleft_dummy, 1}};
            vector<tkv_pair<U>> pair_right{{orbright_dummy, 1}};

            CTFTensor<U>& tensor1 = ap({0,0}, {isvrt_right && isalpha_right, !isvrt_right && isalpha_right})({0});
            if (arena.rank == 0)
                tensor1.writeRemoteData(pair_right);
            else
                tensor1.writeRemoteData();

            CTFTensor<U>& tensor2 = apt({isvrt_left && isalpha_left, !isvrt_left && isalpha_left}, {0,0})({0});
            if (arena.rank == 0)
                tensor2.writeRemoteData(pair_left);
            else
                tensor2.writeRemoteData();

            if ((isvrt_right) && (isvrt_left))
            {
                /*
                 *  ab...    abe...
                 * b  (e) = t
                 *  ijk...   ijk...
                 */

                b(1)[  "i"] = T(1)[  "ei"]*ap["e"];
                b(2)["aij"] = T(2)["aeij"]*ap["e"];

                /*
                 *  ijk...   ijk...
                 * e  (e) = l
                 *  ab...    abe...
                 */

                e(1)[  "i"] = L(1)[  "ie"]*apt["e"];
                e(2)["ija"] = L(2)["ijae"]*apt["e"];

                if (orbright != orbleft)
                {
                 b(1)[  "i"] += T(1)[  "ei"]*apt["e"];
                 b(2)["aij"] += T(2)["aeij"]*apt["e"];

                 e(1)[  "i"] += L(1)[  "ie"]*ap["e"];
                 e(2)["ija"] += L(2)["ijae"]*ap["e"];
                }
            }
            else if((isvrt_right) && (!isvrt_left))
            {
                /*
                 *  ab...    abe...
                 * b  (e) = t
                 *  q
                 *  ijk...   ijk...
                 */
                b(1)[  "i"]  = apt["i"]; //new
                b(1)[  "i"] += T(1)[  "ei"]*ap["e"];
                b(2)["aij"] = T(2)["aeij"]*ap["e"];

                /*
                 *  ijk...           ij...     ijk...
                 * e  (m) = d  (1 + l     ) + G
                 *  ab...    km      ab...     abm...
                 */
                e(1)[  "i"]  =               apt["i"];

                e(1)[  "i"] -=   Dij[  "im"]*apt["m"];
                e(2)["ija"]  =  L(1)[  "ia"]*apt["j"];
                e(2)["ija"] -= Gijak["ijam"]*apt["m"];

                e(1)[  "i"] += L(1)[  "ie"]*ap["e"];  //new
                e(2)["ija"] += L(2)["ijae"]*ap["e"];   //new

            }
            else if((!isvrt_right) && (isvrt_left))
            {
                /*
                 * b (m) = d
                 *  i       im
                 */
                b(1)["i"] = ap["i"];
                b(1)[  "i"] += T(1)[  "ei"]*apt["e"]; //new
                b(2)["aij"] = T(2)["aeij"]*apt["e"];  //new
                /*
                 *  ijk...   ijk...
                 * e  (e) = l
                 *  ab...    abe...
                 */
                e(1)[  "i"] =               ap["i"];
                e(1)[  "i"] += L(1)[  "ie"]*apt["e"];
                e(2)["ija"] = L(2)["ijae"]*apt["e"];

                e(1)[  "i"] -=   Dij[  "im"]*ap["m"];
                e(2)["ija"] +=  L(1)[  "ia"]*ap["j"];
                e(2)["ija"] -= Gijak["ijam"]*ap["m"];

            } 
            else
            {
                /*
                 * b (m) = d
                 *  i       im
                 */
                b(1)["i"] = ap["i"];

                /*
                 *  ijk...           ij...     ijk...
                 * e  (m) = d  (1 + l     ) + G
                 *  ab...    km      ab...     abm...
                 */
                e(1)[  "i"]   =               apt["i"];

                e(1)[  "i"] -=   Dij[  "im"]*apt["m"];
                e(2)["ija"]  =  L(1)[  "ia"]*apt["j"];
                e(2)["ija"] -= Gijak["ijam"]*apt["m"];

              if (orbright != orbleft)
              {
                b(1)["i"] += apt["i"];  //new
                e(1)[  "i"]  +=               ap["i"]; //extra
                e(1)[  "i"] -=   Dij[  "im"]*ap["m"];  //extra
                e(2)["ija"]  +=  L(1)[  "ia"]*ap["j"]; //extra
                e(2)["ija"] -= Gijak["ijam"]*ap["m"]; //extra
              }
            }

              auto& D = this->puttmp("D", new Denominator<U>(H));
           
              int number_of_vectors = nI*nI*nA + nI ; 
              this->puttmp("lanczos", new Lanczos<U,X>(lanczos_config,number_of_vectors));
               
                RL = b ;
                LL = e ;

             /* Evaluate norm 
              */ 

              U norm = sqrt(aquarius::abs(scalar(RL*LL))); 

              norm_ip.emplace_back(norm*norm) ;

              printf("print norm: %10f\n", norm);
              RL /= norm;
              LL /= norm;

              Iterative<U>::run(dag, arena);

              nvec_lanczos = alpha.size() ; 

             for (int ndim = 0;ndim < orbend*orbend ;ndim++)
             {
              alpha_ip[ndim].resize(nvec_lanczos) ;
              beta_ip[ndim].resize(nvec_lanczos) ;
              gamma_ip[ndim].resize(nvec_lanczos) ;
             }  


  /*Define full trdiagonal matrix 
   */  
         vector<U> Tdiag(nvec_lanczos*nvec_lanczos);

         for (int i=0 ; i < nvec_lanczos ; i++){
          for (int j=0 ; j < nvec_lanczos ; j++){
            if (j==i) Tdiag[i*nvec_lanczos + j] = alpha[i] ; 
            if (j==(i-1))Tdiag[i*nvec_lanczos + j] = beta[j];
            if (j==(i+1))Tdiag[i*nvec_lanczos + j] = gamma[i];
          }
         }    

         for (int i=0 ; i < nvec_lanczos ; i++){
            alpha_ip[orbleft*orbend+orbright][i] = alpha[i] ;
            beta_ip[orbleft*orbend+orbright][i] = beta[i] ;
            gamma_ip[orbleft*orbend+orbright][i] = gamma[i] ;
         }
 
  /*
   * Diagonalize the tridiagonal matrix to see if that produces EOM-IP values..
   */

            vector<U> l(nvec_lanczos*nvec_lanczos);
            vector<CU> s_tmp(nvec_lanczos);
            vector<U> vr_tmp(nvec_lanczos*nvec_lanczos);

            int info = geev('N', 'V', nvec_lanczos, Tdiag.data(), nvec_lanczos,
                        s_tmp.data(), l.data(), nvec_lanczos,
                        vr_tmp.data(), nvec_lanczos);
            if (info != 0) throw runtime_error(str("check diagonalization: Info in geev: %d", info));

            for (int i=0 ; i < nvec_lanczos ; i++){
//              printf("real eigenvalues: %.15f\n", s_tmp[i].real());
//              printf("imaginary eigenvalues: %.15f\n", s_tmp[i].imag());
             }

             std::ifstream iffile("gomega_ip.dat");
             if (iffile) remove("gomega_ip.dat");

            U piinverse = 1/M_PI ;

            int omega_counter = 0 ;

            for (auto& o : omegas)
            {
              value  = {0.,0.} ;
              value1 = {0.,0.} ;

              CU alpha_temp ;
              CU beta_temp ;
              CU gamma_temp ;
              CU com_one(1.,0.) ;
              omega = {o.real(),o.imag()} ;
//            omega = {-o.real(),o.imag()} ;

//             this->log(arena) << "Computing Green's function at " << fixed << setprecision(6) << o << endl ;

            /*Evaluate continued fraction 
             */

             for(int i=(nvec_lanczos-1);i >= 0;i--){  
              alpha_temp = {alpha[i],0.} ;
              beta_temp  = {beta[i],0.} ;
              gamma_temp = {gamma[i],0.} ;

              value = (com_one)/(omega + alpha_temp - beta_temp*gamma_temp*value1) ;                 
              value1 = value ;
             }

             if(orbright==orbleft) spec_func[omega_counter] += value*norm*norm;

             if (orb_range == "full") gf_ip[nspin][omega_counter][orbleft][orbright] = value*norm*norm ;
             if (orb_range == "diagonal") gf_ip[nspin][omega_counter][0][0] = value*norm*norm ;

//              printf("real value : %.15f\n", value.real()*norm*norm);
//              printf("imaginary value : %.15f\n", value.imag()*norm*norm);
              omega_counter += 1 ;
             }
          }
        }
       }
             U piinverse = 1/M_PI ;
             std::ofstream gomega;
               gomega.open ("gomega_ip.dat", ofstream::out|std::ios::app);

            for (int i=0 ; i < omegas.size() ; i++){
               gomega << omegas[i].real() << " " << -1/M_PI*spec_func[i].imag() << std::endl ; 
             }

             gomega.close();

            return true;
        }

        void iterate(const Arena& arena)
        {
            const auto& H = this->template get<STTwoElectronOperator<U>>("Hbar");
            const SpinorbitalTensor<U>&   FME =   H.getIA();
            const SpinorbitalTensor<U>&   FAE =   H.getAB();
            const SpinorbitalTensor<U>&   FMI =   H.getIJ();
            const SpinorbitalTensor<U>& WMNEF = H.getIJAB();
            const SpinorbitalTensor<U>& WMNIJ = H.getIJKL();
            const SpinorbitalTensor<U>& WMNEJ = H.getIJAK();
            const SpinorbitalTensor<U>& WAMIJ = H.getAIJK();
            const SpinorbitalTensor<U>& WAMEI = H.getAIBJ();

            auto& T = this->template get<ExcitationOperator<U,2>>("T");

            auto& XE = this->template gettmp<SpinorbitalTensor<U>>("XE");
            auto& XEA = this->template gettmp<SpinorbitalTensor<U>>("XEA");
            auto& D = this->template gettmp<Denominator<U>>("D");
            auto& lanczos = this->template gettmp<Lanczos<U,X>>("lanczos");
            auto& RL = this->template gettmp< ExcitationOperator<U,1,2>>("RL");
            auto& LL = this->template gettmp< DeexcitationOperator<U,1,2>>("LL");
            auto& Z  = this->template gettmp<  ExcitationOperator<U,1,2>>("Z");
            auto& Y  = this->template gettmp<  DeexcitationOperator<U,1,2>>("Y");
            auto& b  = this->template gettmp<  ExcitationOperator<U,1,2>>("b");
            auto& e  = this->template gettmp<DeexcitationOperator<U,1,2>>("e");
            auto& alpha = this->template gettmp<unique_vector<U>> ("alpha");
            auto& beta  = this->template gettmp<unique_vector<U>> ("beta");
            auto& gamma = this->template gettmp<unique_vector<U>> ("gamma");

            int nvec_lanczos; 
            U value ;
            U delta_value ;
            U value1 ;
            U alpha_temp ;
            U beta_temp ;
            U gamma_temp ;

            //printf("<Rr|Ri>: %.15f\n", scalar(Rr*Ri));

            //printf("<B|Rr>: %.15f\n", scalar(b*Rr));
            //printf("<B|Ri>: %.15f\n", scalar(b*Ri));

                XE[  "e"]    = -0.5*WMNEF["mnfe"]*RL(2)[ "fmn"];

                Z(1)[  "i"]  =       -FMI[  "mi"]*RL(1)[   "m"];
                Z(1)[  "i"] +=        FME[  "me"]*RL(2)[ "emi"];
                Z(1)[  "i"] -=  0.5*WMNEJ["mnei"]*RL(2)[ "emn"];
                Z(2)["aij"]  =     -WAMIJ["amij"]*RL(1)[   "m"];
                Z(2)["aij"] +=        FAE[  "ae"]*RL(2)[ "eij"];
                Z(2)["aij"] -=        FMI[  "mi"]*RL(2)[ "amj"];
                Z(2)["aij"] +=         XE[   "e"]*T (2)["aeij"];
                Z(2)["aij"] +=  0.5*WMNIJ["mnij"]*RL(2)[ "amn"];
                Z(2)["aij"] -=      WAMEI["amei"]*RL(2)[ "emj"];

          /*Left hand matrix-vector product : Q^T Hbar
           *We will use Y array for the left hand residual..   
           */ 
                XEA[  "e"]  = -0.5*T(2)["efnm"]*LL(2)["mnf"];

                Y(1)[  "i"]  =       -FMI[  "im"]*LL(1)[  "m"];
                Y(1)[  "i"] -=  0.5*WAMIJ["eimn"]*LL(2)["mne"];
                Y(2)["ija"] =       FAE[  "ea"]*LL(2)["ije"];
                Y(2)["ija"] -=       FMI[  "im"]*LL(2)["mja"];
                Y(2)["ija"] += 0.5*WMNIJ["ijmn"]*LL(2)["mna"];
                Y(2)["ija"] -=     WMNEJ["ijam"]*LL(1)[  "m"];
                Y(2)["ija"] +=       XEA[  "e"]*WMNEF["ijae"];
                Y(2)["ija"] -=     WAMEI["eiam"]*LL(2)["mje"];
                Y(2)["ija"]  +=       FME[  "ia"]*LL(1)[  "j"];
            
            lanczos.extrapolate_tridiagonal(RL, LL, Z, Y, D, alpha, beta, gamma);

              value  = 1. ;
              value1 = 0. ;
              nvec_lanczos = alpha.size() ; 

            if (nvec_lanczos > 2) {
             for(int i=(nvec_lanczos-1);i >= 0;i--){  
              alpha_temp = alpha[i] ;
              beta_temp  = beta[i] ;
              gamma_temp = gamma[i] ;
              value = 1.0/(alpha_temp - beta_temp*gamma_temp*value1) ;                 
              value1 = value ;
              }
             }
             
              old_value.push_back(value) ;

              if (nvec_lanczos <= 2) {
                delta_value = 1.0 ;}
              else{
                delta_value = old_value[nvec_lanczos-2] - value ;
              }

              this->conv() = max(pow(beta[beta.size()-1],2), pow(gamma[gamma.size()-1],2));
//            this->conv() = aquarius::abs(delta_value) ;

        }
};

}
}

static const char* spec = R"(

orbital int,
npoint int,
omega_min double,
omega_max double,
eta double,
grid?
  enum{ real, imaginary },
orbital_range?
  enum{ diagonal, full},
beta?
   double 100.0 , 
convergence?
    double 1e-12,
max_iterations?
    int 150,
conv_type?
    enum { MAXE, RMSE, MAE },
lanczos?
{
    order?
            int 10,
    compaction?
            enum { discrete, continuous },
}

)";

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::CCSDIPGF_LANCZOS);
REGISTER_TASK(aquarius::cc::CCSDIPGF_LANCZOS<double>, "ccsdipgf_lanczos",spec);
