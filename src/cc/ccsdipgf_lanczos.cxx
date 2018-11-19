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
        int element_start ;
        int element_end ;
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
            this->addProduct(Product("ccsd.ipgf", "gf_ip", reqs));

            element_start = config.get<int>("element_start");
            element_end = config.get<int>("element_end");
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
            int orbleft ;
            int orbright ;

            int maxspin = (nI == ni) ? 1 : 2 ;

            auto& T = this->template get<ExcitationOperator  <U,2>>("T");
            auto& L = this->template get<DeexcitationOperator<U,2>>("L");

            vector<vector<U>> alpha_ip ;
            vector<vector<U>> beta_ip ;
            vector<vector<U>> gamma_ip ;
            vector<U> norm_ip ;

           /* vector LL means Left Lanczos and RL means Right Lanczos
            */

            SpinorbitalTensor<U> Dij("D(ij)", arena, group, {vrt,occ}, {0,1}, {0,1});
            SpinorbitalTensor<U> Gijak("G(ij,ak)", arena, group, {vrt,occ}, {0,2}, {1,1});

            SpinorbitalTensor<U> Nij("N(ij)", arena, group, {vrt,occ}, {0,1}, {0,1});

            Dij["ij"]  =     L(1)["ie"  ]*T(1)["ej"  ];
            Dij["ij"] += 0.5*L(2)["imef"]*T(2)["efjm"];

            Gijak["ijak"] = L(2)["ijae"]*T(1)["ek"];

            this->log(arena) << "element start and element end for each process " << element_start << " " << element_end << endl ; 
             alpha_ip.resize(element_end-element_start+1) ;
             beta_ip.resize(element_end-element_start+1) ;
             gamma_ip.resize(element_end-element_start+1) ;

             vector<int> array1((nI+nA)*((nI+nA)+1)/2);
             vector<int> array2((nI+nA)*((nI+nA)+1)/2);
             vector< pair <int,int> > get_index ; 

             int x = 0 ; 
             for (int orbleft = 0; orbleft < (nI+nA) ; orbleft++){   
                for (int orbright = orbleft; orbright < (nI+nA) ; orbright++){   
                    array1[x] = orbleft ;
                    array2[x] = orbright ;
                    x += 1 ;
                }
             }

             for (int i = 0; i < (nI+nA)*((nI+nA)+1)/2 ; i++){   
                 get_index.push_back( make_pair(array1[i],array2[i]) );
             }

        int uppertriangle ;

        for (int nspin = 0; nspin < maxspin ; nspin++){

         uppertriangle = 0 ;

         for (int orbs = element_start; orbs < element_end ; orbs++){   

            orbleft = get_index[orbs].first ;
            orbright = get_index[orbs].second ;

            this->log(arena) << "Computing Green's function element: " << orbleft << " "<< orbright << endl ;

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
            auto& b = this->puttmp("b", new ExcitationOperator  <U,1,2>("b", arena, occ, vrt, isalpha_right ? -1 : 1));
            auto& e = this->puttmp("e", new DeexcitationOperator<U,1,2>("e", arena, occ, vrt, isalpha_left ? 1 : -1));

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
                tensor1.writeRemoteData(pair_right);
            if (arena.rank == 0)
                tensor1.writeRemoteData(pair_right);
            else
                tensor1.writeRemoteData();

            CTFTensor<U>& tensor2 = apt({isvrt_left && isalpha_left, !isvrt_left && isalpha_left}, {0,0})({0});
                tensor2.writeRemoteData(pair_left);
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

                e(1)[  "i"] += L(1)[  "ie"]*ap["e"];  
                e(2)["ija"] += L(2)["ijae"]*ap["e"];  

            }
            else if((!isvrt_right) && (isvrt_left))
            {
                /*
                 * b (m) = d
                 *  i       im
                 */
                b(1)["i"] = ap["i"];
                b(1)[  "i"] += T(1)[  "ei"]*apt["e"]; 
                b(2)["aij"] = T(2)["aeij"]*apt["e"];  
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
                b(1)["i"] += apt["i"]; 
                e(1)[  "i"]  +=               ap["i"]; 
                e(1)[  "i"] -=   Dij[  "im"]*ap["m"];  
                e(2)["ija"]  +=  L(1)[  "ia"]*ap["j"]; 
                e(2)["ija"] -= Gijak["ijam"]*ap["m"]; 
              }
            }

              auto& D = this->puttmp("D", new Denominator<U>(H));
           
              this->puttmp("lanczos", new Lanczos<U,X>(lanczos_config));
               
                RL = b ;
                LL = e ;

             /* Evaluate norm 
              */ 

              U norm = sqrt(aquarius::abs(scalar(RL*LL))); 

              norm_ip.emplace_back(norm*norm) ;

              this->log(arena) << "print norm: " << norm << endl ;
              RL /= norm;
              LL /= norm;

              Iterative<U>::run(dag, arena);

              nvec_lanczos = alpha.size() ; 

              alpha_ip[uppertriangle].resize(nvec_lanczos) ;
              beta_ip[uppertriangle].resize(nvec_lanczos) ;
              gamma_ip[uppertriangle].resize(nvec_lanczos) ;

              for (int i=0 ; i < nvec_lanczos ; i++){
                 alpha_ip[uppertriangle][i] = alpha[i] ;
                 beta_ip[uppertriangle][i] = beta[i]   ;
                 gamma_ip[uppertriangle][i] = gamma[i] ;
              }
              uppertriangle +=1 ;
          }
        }

         if (arena.rank == 0)
         {
             stringstream stream1;
             stream1 << "alpha_ip_"<<element_start<<"_"<<element_end<< ".txt";
             string fileName1 = stream1.str();
             std::ofstream alphafile;
             alphafile.open (fileName1, ofstream::out);

             stringstream stream2;
             stream2 << "beta_ip_"<<element_start<<"_"<<element_end<< ".txt";
             string fileName2 = stream2.str();
             std::ofstream betafile;
             betafile.open (fileName2, ofstream::out);

             stringstream stream3;
             stream3 << "gamma_ip_"<<element_start<<"_"<<element_end<< ".txt";
             string fileName3 = stream3.str();
             std::ofstream gammafile;
             gammafile.open (fileName3, ofstream::out);

             stringstream stream4;
             stream4 << "norm_ip_"<<element_start<<"_"<<element_end<< ".txt";
             string fileName4 = stream4.str();
             std::ofstream normfile;
             normfile.open (fileName4, ofstream::out);

             for (int i=element_start ; i < element_end ; i++){
                for (int j=0 ; j < alpha_ip[0].size() ; j++){
                   alphafile << i << " " << j << " " << setprecision(12) <<  alpha_ip[i-element_start][j] << endl ;
                   betafile << i << " " << j << " " << setprecision(12) << beta_ip[i-element_start][j] << endl ;
                   gammafile << i << " " << j << " " << setprecision(12) << gamma_ip[i-element_start][j] << endl ;
                }
                   normfile << setprecision(12) <<  norm_ip[i-element_start] << endl ;
             }

             alphafile.close();
             betafile.close();
             gammafile.close();
             normfile.close();
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

              this->conv() = max(pow(beta[beta.size()-1],2), pow(gamma[gamma.size()-1],2));
        }

};

}
}

static const char* spec = R"(

element_start ?
   int 0,
element_end int,
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
