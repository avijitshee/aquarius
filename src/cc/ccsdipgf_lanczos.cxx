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
class CCSDIPGF_LANCZOS : public Iterative<U>
{
    protected:
        typedef U X ; 
        typedef complex_type_t<U> CU;
        Config lanczos_config;
        int orbital;
        vector<CU> omegas;
        CU omega;

    public:
        CCSDIPGF_LANCZOS(const string& name, Config& config)
        : Iterative<U>(name, config), lanczos_config(config.get("lanczos"))
        {
            vector<Requirement> reqs;
            reqs.emplace_back("ccsd.T", "T");
            reqs.emplace_back("ccsd.L", "L");
            reqs.emplace_back("ccsd.Hbar", "Hbar");
            this->addProduct("ccsd.ipgflanczos", "gf", reqs);

            orbital = config.get<int>("orbital");
            double from = config.get<double>("omega_min");
            double to = config.get<double>("omega_max");
            int n = config.get<double>("npoint");
            double eta = config.get<double>("eta");

            double delta = (to-from)/max(1,n-1);
            for (int i = 0;i < n;i++)
            {
                omegas.emplace_back(from+delta*i, eta);
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


            auto& T = this->template get<ExcitationOperator  <U,2>>("T");
            auto& L = this->template get<DeexcitationOperator<U,2>>("L");


            bool isalpha = false;
            bool isvrt = false;
            if (orbital > 0)
            {
                isalpha = true;
                orbital--;
                if (orbital >= nI)
                {
                    isvrt = true;
                    orbital -= nI;
                }
            }
            else
            {
                orbital = -orbital-1;
                if (orbital >= ni)
                {
                    isvrt = true;
                    orbital -= ni;
                }
            }


           /* vector LL means Left Lanczos and RL means Right Lanczos
            */
            auto& RL = this->puttmp("RL", new ExcitationOperator  <U,1,2>("RL", arena, occ, vrt, isalpha ? -1 : 1));
            auto& LL = this->puttmp("LL", new DeexcitationOperator  <U,1,2>("LL", arena, occ, vrt, isalpha ? 1 : -1));
            auto& Z = this->puttmp("Z", new ExcitationOperator  <U,1,2>("Z", arena, occ, vrt, isalpha ? -1 : 1));
            auto& Y = this->puttmp("Y", new DeexcitationOperator  <U,1,2>("Y", arena, occ, vrt, isalpha ? 1 : -1));
            auto& b  = this->puttmp("b",  new ExcitationOperator  <U,1,2>("b",  arena, occ, vrt, isalpha ? -1 : 1));
            auto& e  = this->puttmp("e",  new DeexcitationOperator<U,1,2>("e",  arena, occ, vrt, isalpha ? 1 : -1));


            auto& XE = this->puttmp("XE", new SpinorbitalTensor<U>("X(e)", arena, group, {vrt,occ}, {0,0}, {1,0}, isalpha ? -1 : 1));
            auto& XEA = this->puttmp("XEA", new SpinorbitalTensor<U>("XA(e)", arena, group, {vrt,occ}, {0,0}, {1,0}, isalpha ? -1 : 1));
//          auto& alpha = this-> puttmp("alpha", new vector<unique_vector<U>>()) ;
//          auto& beta  = this-> puttmp("beta", new vector<unique_vector<U>>())  ;
//          auto& gamma = this-> puttmp("gamma", new  vector<unique_vector<U>>());
            auto& alpha = this-> puttmp("alpha", new unique_vector<U>()) ;
            auto& beta  = this-> puttmp("beta", new unique_vector<U>()) ;
            auto& gamma = this-> puttmp("gamma", new unique_vector<U>());

            SpinorbitalTensor<U> Dij("D(ij)", arena, group, {vrt,occ}, {0,1}, {0,1});
            SpinorbitalTensor<U> Gijak("G(ij,ak)", arena, group, {vrt,occ}, {0,2}, {1,1});

            SpinorbitalTensor<U> Nij("N(ij)", arena, group, {vrt,occ}, {0,1}, {0,1});


            Dij["ij"]  =     L(1)["ie"  ]*T(1)["ej"  ];
            Dij["ij"] += 0.5*L(2)["imef"]*T(2)["efjm"];

            Gijak["ijak"] = L(2)["ijae"]*T(1)["ek"];

            SpinorbitalTensor<U> ap ("ap"  , arena, group, {vrt,occ}, {0,0}, {isvrt, !isvrt}, isalpha ? -1 : 1);
            SpinorbitalTensor<U> apt("ap^t", arena, group, {vrt,occ}, {isvrt, !isvrt}, {0,0}, isalpha ? 1 : -1);

            vector<tkv_pair<U>> pairs{{orbital, 1}};

            CTFTensor<U>& tensor1 = ap({0,0}, {isvrt && isalpha, !isvrt && isalpha})({0});
            if (arena.rank == 0)
                tensor1.writeRemoteData(pairs);
            else
                tensor1.writeRemoteData();

            CTFTensor<U>& tensor2 = apt({isvrt && isalpha, !isvrt && isalpha}, {0,0})({0});
            if (arena.rank == 0)
                tensor2.writeRemoteData(pairs);
            else
                tensor2.writeRemoteData();

            if (isvrt)
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
                e(1)[  "i"]  =               apt["i"];
                e(1)[  "i"] -=   Dij[  "im"]*apt["m"];
                e(2)["ija"]  =  L(1)[  "ia"]*apt["j"];
                e(2)["ija"] -= Gijak["ijam"]*apt["m"];
            }


            //printf("<E|E>: %.15f\n", scalar(e*e));
            //printf("<B|B>: %.15f\n", scalar(b*b));
            //printf("<E|B>: %.15f\n", scalar(e(1)["m"]*b(1)["m"])+0.5*scalar(e(2)["mne"]*b(2)["emn"]));

              auto& D = this->puttmp("D", new Denominator<U>(H));
           
              int number_of_vectors = nI*nI*nA + nI ; 
              this->puttmp("lanczos", new Lanczos<U,X>(lanczos_config,number_of_vectors));

              RL = b;
              LL = e;
  
              U norm = sqrt(scalar(LL*RL)); 
              RL /= norm;
              LL /= norm;

             /* Evaluate norm 
              */ 

              Nij["ij"] = e(1)[  "i"]*b(1)[  "j"]  ;
              Nij["ij"] -= e(2)["ime"]*b(2)["ejm"];

              Iterative<U>::run(dag, arena);

              nvec_lanczos = alpha.size() ; 

            for (auto& o : omegas)
            {

            /*Evaluate continued fraction 
             */

              value = {0.,0.} ;

             for(int i=(nvec_lanczos-1);i >= 0;--i){  
              value = (1.0,0.0)/(o - alpha[i] - beta[i+1]*gamma[i+1]*value) ;                 
             }
             
             this->log(arena) << "Computing Green's function at " << fixed << setprecision(6) << o << endl ;

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

            auto& T = this->template get<ExcitationOperator<U,2>>("T");

            auto& XE = this->template gettmp<SpinorbitalTensor<U>>("XE");
            auto& XEA = this->template gettmp<SpinorbitalTensor<U>>("XEA");

            auto& D = this->template gettmp<Denominator<U>>("D");
//            auto& lanczos = this->template gettmp<Lanczos<unique_vector<U>>>("lanczos");
//           auto& lanczos = this->template gettmp<Lanczos<ExcitationOperator<U,1,2>>>("lanczos");
            auto& lanczos = this->template gettmp<Lanczos<U,X>>("lanczos");

            auto& RL = this->template gettmp< ExcitationOperator<U,1,2>>("RL");
            auto& LL = this->template gettmp< DeexcitationOperator<U,1,2>>("LL");
            auto& Z = this->template  gettmp<  ExcitationOperator<U,1,2>>("Z");
            auto& Y = this->template  gettmp<  DeexcitationOperator<U,1,2>>("Y");
            auto& b  = this->template gettmp<  ExcitationOperator<U,1,2>>("b");
            auto& e  = this->template gettmp<DeexcitationOperator<U,1,2>>("e");

//          auto& alpha = this->template gettmp<vector<unique_vector<U>>> ("alpha");
//          auto& beta  = this->template gettmp<vector<unique_vector<U>>>  ("beta");
//          auto& gamma = this->template gettmp<vector<unique_vector<U>>> ("gamma");

            auto& alpha = this->template gettmp<unique_vector<U>> ("alpha");
            auto& beta  = this->template gettmp<unique_vector<U>> ("beta");
            auto& gamma = this->template gettmp<unique_vector<U>> ("gamma");

            printf("<RL|RL>: %.15f\n", scalar(RL*RL));
            printf("<LL|RL>: %.15f\n", scalar(LL*RL));
            //printf("<Rr|Ri>: %.15f\n", scalar(Rr*Ri));

            //printf("<B|Rr>: %.15f\n", scalar(b*Rr));
            //printf("<B|Ri>: %.15f\n", scalar(b*Ri));

                  XE[  "e"]  = -0.5*WMNEF["mnfe"]*RL(2)[ "fmn"];

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
                XEA[  "e"]  = -0.5*T(2)["efmn"]*LL(2)["mnf"];

                Y(1)[  "i"] -=       FMI[  "im"]*LL(1)[  "m"];
                Y(1)[  "i"] -= 0.5*WAMIJ["eimn"]*LL(2)["mne"];

                Y(2)["ija"] +=       FAE[  "ea"]*LL(2)["ije"];
                Y(2)["ija"] -=       FMI[  "im"]*LL(2)["mja"];
                Y(2)["ija"] += 0.5*WMNIJ["ijmn"]*LL(2)["mna"];

                Y(2)["ija"] +=       XEA[  "e"]*WMNEF["ijeb"];

            printf("<Z(2)|Z(2)>: %.15f\n", scalar(Z(2)*Z(2)));
            printf("<Y1|Y1>: %.15f\n", scalar(Y(1)*Y(1)));
            //printf("<Z2|Z2>: %.15f\n", 0.5*scalar(Z(2)*Z(2)));
            //printf("<Z|Z>: %.15f\n", scalar(Z*Z));
            //printf("<Zi|Zi>: %.15f\n", scalar(Zi*Zi));

            //printf("<Ur|Ur>: %.15f\n", scalar(Z*Z));
            //printf("<Ui|Ui>: %.15f\n", scalar(Zi*Zi));
            
            lanczos.extrapolate_tridiagonal(RL, LL, Z, Y, D, alpha, beta, gamma);

            printf("have passed this step 1: %.10f\n", beta[beta.size()-1]);

            this->conv() = max(beta[beta.size()-1], gamma[gamma.size()-1]);

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
convergence?
    double 1e-9,
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
