#include "util/global.hpp"

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
class CCSDEAGF_LANCZOS : public Iterative<U>
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
        CCSDEAGF_LANCZOS(const string& name, Config& config)
        : Iterative<U>(name, config), lanczos_config(config.get("lanczos"))
        {
            vector<Requirement> reqs;
            reqs.emplace_back("ccsd.T", "T");
            reqs.emplace_back("ccsd.L", "L");
            reqs.emplace_back("ccsd.Hbar", "Hbar");
            this->addProduct("ccsd.eagflanczos", "gf_ea", reqs);

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
             if (grid_type == "imaginary") omegas.emplace_back(0.,2.0*i*M_PI/beta);
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

            auto& T = this->template get<ExcitationOperator  <U,2>>("T");
            auto& L = this->template get<DeexcitationOperator<U,2>>("L");


            auto& gf_ea = this-> puttmp("gf_ea", new vector<vector<vector<CU>>>) ;

            SpinorbitalTensor<U> Dab("D(ab)", arena, group, {vrt,occ}, {1,0}, {1,0});
            SpinorbitalTensor<U> Gieab("G(am,ef)", arena, group, {vrt,occ}, {1,1}, {2,0});

            SpinorbitalTensor<U> Nij("N(ij)", arena, group, {vrt,occ}, {0,1}, {0,1});

            Dab["ab"]  =     -L(1)["mb"  ]*T(1)["am"  ];
            Dab["ab"] -= 0.5*L(2)["kmbe"]*T(2)["aekm"];

            Gieab["amef"]  = -L(2)["nmef"]*T(1)[  "an"];


            bool isalpha_right = false;
            bool isvrt_right = true;
            bool isalpha_left = false;
            bool isvrt_left = true;

           if (orb_range == "full") 
           { orbstart = 0 ;
             orbend = nI + nA ;  
             gf_ea.resize(omegas.size());
             for (int i = 0;i < omegas.size();i++)
             {
               gf_ea[i].resize(orbend);
             }
             for (int i = 0;i < omegas.size();i++)
             for (int j = 0;j < orbend;j++)
             {
               gf_ea[i][j].resize(orbend);
             }
           } 

           if (orb_range == "diagonal") 
           { orbstart = orbital-1 ;
             orbend = orbital;  
             gf_ea.resize(omegas.size());
             for (int i = 0;i < omegas.size();i++)
             {
               gf_ea[i].resize(1);
             }
             for (int i = 0;i < omegas.size();i++)
             for (int j = 0;j < 1;j++)
             {
               gf_ea[i][j].resize(1);
             }
           }

          for (int orbleft = orbstart; orbleft < orbend ; orbleft++)   
          {
           for (int orbright = orbstart; orbright < orbend ; orbright++)   
            {

             printf("Computing Green's function element:  %d %d\n", orbleft, orbright ) ;

            int orbleft_dummy = orbleft ;
            int orbright_dummy = orbright ;

            if ((orbleft+1) > 0)
            {
                isalpha_left = true;
                if ((orbleft ) >= nI)
                {
                    isvrt_left = false;
                    orbleft_dummy = orbleft - nI;
                }
            }
            else
            {
                orbleft = -orbital-2;
                if (orbleft >= ni)
                {
                    isvrt_left = false;
                    orbleft -= ni;
                }
            }

            if ((orbright+1) > 0)
            {
                isalpha_right = true;
                if ((orbright ) >= nI)
                {
                    isvrt_right = false;
                    orbright_dummy = orbright - nI;
                }
            }
            else
            {
                orbright = -orbital-2;
                if (orbright >= ni)
                {
                    isvrt_right = false;
                    orbright -= ni;
                }
            }

//          bool isalpha = false;
//          bool isvrt = false;
//          if (orbital > nI)
//          {
//             if (orbital <= nI)
//              {
//                  isvrt = true;
//                  orbital --;
//              }
//              isalpha = true;
//              orbital -= (nI+1);
//              
//          }
//          else
//          {
//              orbital = -orbital-1;
//              if (orbital >= ni)
//              {
//                  isvrt = true;
//                  orbital -= ni;
//              }
//          }

           /* vector LL means Left Lanczos and RL means Right Lanczos
            */
            auto& RL = this->puttmp("RL", new ExcitationOperator  <U,2,1>("RL", arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& LL = this->puttmp("LL", new DeexcitationOperator  <U,2,1>("LL", arena, occ, vrt, isalpha_left ? -1 : 1));
            auto& Z = this->puttmp("Z", new ExcitationOperator  <U,2,1>("Z", arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& Y = this->puttmp("Y", new DeexcitationOperator  <U,2,1>("Y", arena, occ, vrt, isalpha_left ? -1 : 1));
            auto& b  = this->puttmp("b",  new ExcitationOperator  <U,2,1>("b",  arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& e  = this->puttmp("e",  new DeexcitationOperator<U,2,1>("e",  arena, occ, vrt, isalpha_left ? -1 : 1));

            auto& XMI = this->puttmp("XMI", new SpinorbitalTensor<U>("X(mi)", arena, group, {vrt,occ}, {0,1}, {0,0}, isalpha_left ? 1 : -1));
            auto& GIM = this->puttmp("GIM", new SpinorbitalTensor<U>("G(im)", arena, group, {vrt,occ}, {0,0}, {0,1}, isalpha_right ? -1 : 1));
            auto& alpha = this-> puttmp("alpha", new unique_vector<U>()) ;
            auto& beta  = this-> puttmp("beta", new unique_vector<U>()) ;
            auto& gamma = this-> puttmp("gamma", new unique_vector<U>());

            SpinorbitalTensor<U> ap ("ap"  , arena, group, {vrt,occ}, {!isvrt_right, isvrt_right}, {0,0}, isalpha_right ? 1 : -1);
            SpinorbitalTensor<U> apt("ap^t", arena, group, {vrt,occ}, {0,0}, {!isvrt_left, isvrt_left}, isalpha_left ? -1 : 1);

//            vector<tkv_pair<U>> pairs{{orbital, 1}};

            vector<tkv_pair<U>> pair_left{{orbleft_dummy, 1}};
            vector<tkv_pair<U>> pair_right{{orbright_dummy, 1}};

//          vector<tkv_pair<U>> pairs;

//            for (int vec = 0; vec < nI; vec++) {
//              pairs.push_back(tkv_pair<U>(vec, vec)) ;
//            }

            CTFTensor<U>& tensor1 = ap({!isvrt_right && isalpha_right, isvrt_right && isalpha_right}, {0,0})({0});
            if (arena.rank == 0)
                tensor1.writeRemoteData(pair_right);
            else
                tensor1.writeRemoteData();

            CTFTensor<U>& tensor2 = apt({0,0}, {!isvrt_left && isalpha_left, isvrt_left && isalpha_left})({0});
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
                b(1)[  "a"] = -T(1)[  "ak"]*ap["k"];
                b(2)["abi"] = -T(2)["abik"]*ap["k"];

                /*
                 *  ijk...   ijk...
                 * e  (e) = l
                 *  ab...    abe...
                 */
                e(1)[  "a"]  = -L(1)[  "ka"]*apt["k"];
                e(2)["iab"]  = -L(2)["ikab"]*apt["k"];
            }

            else if((isvrt_right) && (!isvrt_left))
            {
                /*
                 *  ab...    abe...
                 * b  (e) = t
                 *  ijk...   ijk...
                 */
                b(1)[  "a"] = -T(1)[  "ak"]*ap["k"];
                b(2)["abi"] = -T(2)["abik"]*ap["k"];

                /*
                 *  ijk...           ij...     ijk...
                 * e  (m) = d  (1 + l     ) + G
                 *  ab...    km      ab...     abm...
                 */
                e(1)[  "a"]  =               apt["a"];
                e(1)[  "a"]  +=   Dab[  "ea"]*apt["e"];
                e(2)["iab"]  =  L(1)[  "ia"]*apt["b"];
                e(2)["iab"]  += Gieab["eiba"]*apt["e"];
            }
            else if((!isvrt_right) && (isvrt_left))
            {
                /*
                 * b (m) = d
                 *  i       im
                 */
                b(1)["a"] = ap["a"];

                /*
                 *  ijk...   ijk...
                 * e  (e) = l
                 *  ab...    abe...
                 */
                e(1)[  "a"]  = -L(1)[  "ka"]*apt["k"];
                e(2)["iab"]  = -L(2)["ikab"]*apt["k"];
            }
            else
            {
                /*
                 * b (m) = d
                 *  i       im
                 */
                b(1)["a"] = ap["a"];

                /*
                 *  ijk...           ij...     ijk...
                 * e  (m) = d  (1 + l     ) + G
                 *  ab...    km      ab...     abm...
                 */
                e(1)[  "a"]  =               apt["a"];
                e(1)[  "a"]  +=   Dab[  "ea"]*apt["e"];
                e(2)["iab"]  =  L(1)[  "ia"]*apt["b"];
                e(2)["iab"]  += Gieab["eiba"]*apt["e"];
            }


            //printf("<E|E>: %.15f\n", scalar(e*e));
            //printf("<B|B>: %.15f\n", scalar(b*b));
            //printf("<E|B>: %.15f\n", scalar(e(1)["m"]*b(1)["m"])+0.5*scalar(e(2)["mne"]*b(2)["emn"]));

              auto& D = this->puttmp("D", new Denominator<U>(H));
           
              int number_of_vectors = nI*nA*nA + nA ; 
              this->puttmp("lanczos", new Lanczos<U,X>(lanczos_config,number_of_vectors));

             /* Evaluate norm 
              */ 
//              RL(1) = b(1) ;
//              RL(1) += e(1) ;
//              RL(1) = 0.5*RL(1) ;
//              LL(1) = e(1) ;
//              LL(1) += b(1) ;
//              LL(1) = 0.5*LL(1) ;

//              RL(2)["abi"] = e(2)["iab"] ;
//              RL(2)["abi"] += b(2)["abi"] ;
//              RL(2)["abi"] = 0.5*RL(2)["abi"] ;
//              LL(2)["iab"] = b(2)["abi"] ;
//              LL(2)["iab"] += e(2)["iab"] ;
//              LL(2)["iab"] = 0.5*LL(2)["iab"] ;

              RL = b ;
              LL = e ;
              U norm = sqrt(aquarius::abs(scalar(RL*LL))); 
              RL /= norm;
              LL /= norm;

              printf("print norm: %10f\n", norm);

              Iterative<U>::run(dag, arena);

              nvec_lanczos = alpha.size() ; 

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

            /*
             * Diagonalize the tridiagonal matrix to see if that produces EOM-EA values..
             */


            vector<U>  l(nvec_lanczos*nvec_lanczos);
            vector<CU> s_tmp(nvec_lanczos);
            vector<U>  vr_tmp(nvec_lanczos*nvec_lanczos);

            int info = geev('N', 'V', nvec_lanczos, Tdiag.data(), nvec_lanczos,
                        s_tmp.data(), l.data(), nvec_lanczos,
                        vr_tmp.data(), nvec_lanczos);
            if (info != 0) throw runtime_error(str("check diagonalization: Info in geev: %d", info));

            for (int i=0 ; i < nvec_lanczos ; i++){
              printf("real eigenvalues: %.15f\n", s_tmp[i].real());
              printf("imaginary eigenvalues: %.15f\n", s_tmp[i].imag());
             }

            std::ifstream iffile("gomega.dat");
            if (iffile) remove("gomega.dat");

            U pi = 2*acos(0.0);

            U piinverse = 1.0/M_PI ;

            for (auto& o : omegas)
            {

            /*Evaluate continued fraction 
             */

              value  = {0.,0.} ;
              value1 = {0.,0.} ;

              CU alpha_temp ;
              CU beta_temp ;
              CU gamma_temp ;
              CU com_one(1.,0.) ;
              omega = {o.real(),o.imag()} ;

              this->log(arena) << "Computing Green's function at " << fixed << setprecision(6) << o << endl ;

             for(int i=(nvec_lanczos-1);i >= 0;i--){  
              alpha_temp = {alpha[i],0.} ;
              beta_temp  = {beta[i],0.} ;
              gamma_temp = {gamma[i],0.} ;

//            value = (1.0)/(o.real() - alpha[i] - beta[i+1]*gamma[i+1]*value1) ;                 
              value = (com_one)/(omega - alpha_temp - beta_temp*gamma_temp*value1) ;                 
              value1 = value ;
             }

             std::ofstream gomega;
//             gomega.open ("gomega.dat", std::ofstream::out);
             gomega.open ("gomega.dat", ofstream::out|std::ios::app);
//             gomega << o.real() << " " << -piinverse*value.imag()*norm*norm << std::endl ; 
              gomega << o.real() << " " << piinverse*value.imag()*norm*norm << std::endl ; 
             gomega.close();

              printf("real value at least: %.15f\n", value.real()*norm*norm);
              printf("imaginary value at least: %.15f\n", value.imag()*norm*norm);

//              Nij = value*Nij ;
//              printf("value: %.15f\n", value);
             }
            }
           } 
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
            const SpinorbitalTensor<U>& WAMEF = H.getAIBC();
            const SpinorbitalTensor<U>& WABEJ = H.getABCI();
            const SpinorbitalTensor<U>& WABEF = H.getABCD();

            auto& T = this->template get<ExcitationOperator<U,2>>("T");

            auto& XMI = this->template gettmp<SpinorbitalTensor<U>>("XMI");
            auto& GIM = this->template gettmp<SpinorbitalTensor<U>>("GIM");

            auto& D = this->template gettmp<Denominator<U>>("D");
//          auto& lanczos = this->template gettmp<Lanczos<unique_vector<U>>>("lanczos");
//          auto& lanczos = this->template gettmp<Lanczos<ExcitationOperator<U,2,1>>>("lanczos");
            auto& lanczos = this->template gettmp<Lanczos<U,X>>("lanczos");

            auto& RL = this->template gettmp< ExcitationOperator<U,2,1>>("RL");
            auto& LL = this->template gettmp< DeexcitationOperator<U,2,1>>("LL");
            auto& Z  = this->template gettmp<  ExcitationOperator<U,2,1>>("Z");
            auto& Y  = this->template gettmp<  DeexcitationOperator<U,2,1>>("Y");
            auto& b  = this->template gettmp<  ExcitationOperator<U,2,1>>("b");
            auto& e  = this->template gettmp<DeexcitationOperator<U,2,1>>("e");

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

            printf("<RL|RL>: %.15f\n", scalar(RL*RL));
            printf("<LL1|LL1>: %.15f\n", scalar(LL(1)["m"]*LL(1)["m"]));
            printf("<LL|LL>: %.15f\n", scalar(LL*LL));
            printf("<LL|RL>: %.15f\n", scalar(RL*LL));
            //printf("<Rr|Ri>: %.15f\n", scalar(Rr*Ri));

            //printf("<B|Rr>: %.15f\n", scalar(b*Rr));
            //printf("<B|Ri>: %.15f\n", scalar(b*Ri));

                XMI[  "m"] = -0.5*WMNEF["mnef"]*RL(2)["efn"];

                Z(1)[  "a"]  =       FAE[  "ae"]*RL(1)[  "e"];
                Z(1)[  "a"] -=       FME[  "me"]*RL(2)["aem"];
                Z(1)[  "a"] -= 0.5*WAMEF["amef"]*RL(2)["efm"];

                Z(2)["abi"]   =     WABEJ["baei"]*RL(1)[  "e"];
                Z(2)["abi"]  +=       FAE[  "ae"]*RL(2)["ebi"];
                Z(2)["abi"]  -=       FMI[  "mi"]*RL(2)["abm"];
                Z(2)["abi"]  -=       XMI[  "m"]*T(2)["abim"];
                Z(2)["abi"]  += 0.5*WABEF["abef"]*RL(2)["efi"];
                Z(2)["abi"]  -=     WAMEI["amei"]*RL(2)["ebm"];


          /*Left hand matrix-vector product : Q^T Hbar
           *We will use Y array for the left hand residual..   
           */ 
            GIM[ "m"]   =  -0.5*T(2)["efmo"]*LL(2)["oef"];

            Y(1)[ "a"]  =       FAE[  "ea"]*LL(1)[  "e"];
            Y(1)[ "a"] -= 0.5*WABEJ["efam"]*LL(2)["mef"];

            Y(2)["iab"]  =       FME[  "ia"]*LL(1)[  "b"];
            Y(2)["iab"] +=     WAMEF["eiba"]*LL(1)[  "e"];
            Y(2)["iab"] +=       FAE[  "ea"]*LL(2)["ieb"];
            Y(2)["iab"] -=       FMI[  "im"]*LL(2)["mab"];
            Y(2)["iab"] += 0.5*WABEF["efab"]*LL(2)["ief"];
            Y(2)["iab"] -=     WAMEI["eibm"]*LL(2)["mae"];
            Y(2)["iab"] -=     WMNEF["miba"]* GIM[  "m"];
 
            printf("<Z1|Z1>: %.15f\n", scalar(Z(1)*Z(1)));
            printf("<Z2|Z2>: %.15f\n", 0.5*scalar(Z(2)*Z(2)));
            printf("<Z|Z>: %.15f\n",   scalar(Z*Z));
            printf("<Y1|Y1>: %.15f\n", scalar(Y(1)*Y(1)));
            printf("<Y2|Y2>: %.15f\n", 0.5* scalar(Y(2)*Y(2)));
            printf("<Y|Y>: %.15f\n", scalar(Y*Y));
            //printf("<Z2|Z2>: %.15f\n", 0.5*scalar(Z(2)*Z(2)));
            //printf("<Z|Z>: %.15f\n", scalar(Z*Z));
            //printf("<Zi|Zi>: %.15f\n", scalar(Zi*Zi));

            //printf("<Ur|Ur>: %.15f\n", scalar(Z*Z));
            //printf("<Ui|Ui>: %.15f\n", scalar(Zi*Zi));
            
            lanczos.extrapolate_tridiagonal(RL, LL, Z, Y, D, alpha, beta, gamma);

              value  = 1. ;
              value1 = 0. ;
              nvec_lanczos = alpha.size() ; 

            if (nvec_lanczos > 2) {
             for(int i=(nvec_lanczos-1);i >= 0;i--){  
              alpha_temp = alpha[i] ;
              beta_temp  = beta[i] ;
              gamma_temp = gamma[i] ;
              value = 1.0/(alpha_temp + beta_temp*gamma_temp*value1) ;                 
              value1 = value ;
              }
             }
             
              old_value.push_back(value) ;

              if (nvec_lanczos <= 2) {
                delta_value = 1.0 ;}
              else{
                delta_value = old_value[nvec_lanczos-2] - value ;
              }

              printf("have passed this step 1: %.10f\n", beta[beta.size()-1]);
              printf("old_value: %.10f\n", old_value[nvec_lanczos-1]);
              printf("new value: %.10f\n", value);

//            this->conv() = max(pow(beta[beta.size()-1],2), pow(gamma[gamma.size()-1],2));
           this->conv() = max(pow(beta[beta.size()-1],2), pow(gamma[gamma.size()-1],2));
//             this->conv() = aquarius::abs(delta_value) ;

//            lanczos.getSolution(alpha, beta, gamma);
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

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::CCSDEAGF_LANCZOS);
REGISTER_TASK(aquarius::cc::CCSDEAGF_LANCZOS<double>, "ccsdeagf_lanczos",spec);
