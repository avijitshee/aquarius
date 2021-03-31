#include "util/global.hpp"

#include "time/time.hpp"
#include "task/task.hpp"
#include "convergence/lanczos.hpp"
#include "util/iterative.hpp"
#include "operator/2eoperator.hpp"
#include "operator/st2eoperator.hpp"
#include "operator/space.hpp"
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
class CCSDTEAGF_LANCZOS_D_T : public Iterative<U>
{
    protected:
        typedef U X ; 
        typedef complex_type_t<U> CU;
        Config lanczos_config;
        int element_start ;
        int element_end ;

    public:
        CCSDTEAGF_LANCZOS_D_T(const string& name, Config& config)
        : Iterative<U>(name, config), lanczos_config(config.get("lanczos"))
        {
            vector<Requirement> reqs;
            reqs.emplace_back("ccsd.T", "T");
            reqs.emplace_back("ccsd.L", "L");
            reqs.emplace_back("ccsd.Hbar", "Hbar");
            this->addProduct(Product("ccsdt.eagf", "gf_ea", reqs));

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

            int maxspin =  2 ;

            auto& T = this->template get<ExcitationOperator  <U,2>>("T");
            auto& L = this->template get<DeexcitationOperator<U,2>>("L");

            vector<vector<vector<U>>> alpha_ea ;
            vector<vector<vector<U>>> beta_ea ;
            vector<vector<vector<U>>> gamma_ea ;
            vector<vector<U>> norm_ea ;

            SpinorbitalTensor<U> Dab("D(ab)", arena, group, {vrt,occ}, {1,0}, {1,0});
            SpinorbitalTensor<U> Gieab("G(am,ef)", arena, group, {vrt,occ}, {1,1}, {2,0});

            SpinorbitalTensor<U> Nij("N(ij)", arena, group, {vrt,occ}, {0,1}, {0,1});

            Dab["ab"]  =     -L(1)["mb"  ]*T(1)["am"  ];
            Dab["ab"] -= 0.5*L(2)["kmbe"]*T(2)["aekm"];

            Gieab["amef"]  = -L(2)["nmef"]*T(1)[  "an"];

             alpha_ea.resize(maxspin) ;
             beta_ea.resize(maxspin) ;
             gamma_ea.resize(maxspin) ;
             norm_ea.resize(maxspin) ;
           
            for (int nspin = 0; nspin < maxspin ; nspin++){
              alpha_ea[nspin].resize(element_end-element_start+1) ;
              beta_ea[nspin].resize(element_end-element_start+1) ;
              gamma_ea[nspin].resize(element_end-element_start+1) ;
            }

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

     /* start calculating all GF elements..
      */  

        int uppertriangle ;

        for (int nspin = 0; nspin < maxspin ; nspin++){
            uppertriangle = 0 ;
         for (int orbs = element_start; orbs < element_end ; orbs++){   

            orbleft = get_index[orbs].first ;
            orbright = get_index[orbs].second ;

            this->log(arena) << "Computing Green's function element: " << orbleft << " "<< orbright << endl ;

            bool isalpha_right = false;
            bool isvrt_right = true;
            bool isalpha_left = false;
            bool isvrt_left = true;

            int orbleft_dummy = orbleft ;
            int orbright_dummy = orbright ;

            if (nspin == 0)
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
                if (orbleft >= ni)
                {
                    isvrt_left = false;
                    orbleft_dummy = orbleft - ni;
                }
            }

            if (nspin == 0)
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
                if (orbright >= ni)
                {
                    isvrt_right = false;
                    orbright_dummy = orbright - ni;
                }
            }

  /* vector LL means Left Lanczos and RL means Right Lanczos, in case you are wondering!  
   */
            auto& RL = this->puttmp("RL", new ExcitationOperator  <U,3,2>("RL", arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& LL = this->puttmp("LL", new DeexcitationOperator  <U,3,2>("LL", arena, occ, vrt, isalpha_left ? -1 : 1));
            auto& Z = this->puttmp("Z", new ExcitationOperator  <U,3,2>("Z", arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& Y = this->puttmp("Y", new DeexcitationOperator  <U,3,2>("Y", arena, occ, vrt, isalpha_left ? -1 : 1));
            auto& b  = this->puttmp("b",  new ExcitationOperator  <U,3,2>("b",  arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& e  = this->puttmp("e",  new DeexcitationOperator<U,3,2>("e",  arena, occ, vrt, isalpha_left ? -1 : 1));

            auto& XMI = this->puttmp("XMI", new SpinorbitalTensor<U>("X(mi)", arena, group, {vrt,occ}, {0,1}, {0,0}, isalpha_left ? 1 : -1));

            auto& XMCI = this->puttmp("XMCI", new SpinorbitalTensor<U>("X(mc,i)", arena, group, {vrt,occ}, {1,1}, {0,1}, isalpha_right ? -1 : 1));
            auto& XACE = this->puttmp("XACE", new SpinorbitalTensor<U>("X(ac,e)", arena, group, {vrt,occ}, {2,0}, {1,0}, isalpha_right ? -1 : 1));

            auto& GIM = this->puttmp("GIM", new SpinorbitalTensor<U>("G(im)", arena, group, {vrt,occ}, {0,0}, {0,1}, isalpha_right ? -1 : 1));

            auto& XIMC = this->puttmp("XIMC", new SpinorbitalTensor<U>("X(i,mc)", arena, group, {vrt,occ}, {0,1}, {1,1}, isalpha_left ? 1 : -1));
            auto& XEAC = this->puttmp("XEAC", new SpinorbitalTensor<U>("X(e,ac)", arena, group, {vrt,occ}, {1,0}, {2,0}, isalpha_left ? 1 : -1));
            auto& XEMO = this->puttmp("XEMO", new SpinorbitalTensor<U>("X(e,mo)", arena, group, {vrt,occ}, {1,0}, {0,2}, isalpha_right ? -1 : 1));

            auto& alpha = this-> puttmp("alpha", new unique_vector<U>()) ;
            auto& beta  = this-> puttmp("beta", new unique_vector<U>()) ;
            auto& gamma = this-> puttmp("gamma", new unique_vector<U>());

            SpinorbitalTensor<U> ap ("ap"  , arena, group, {vrt,occ}, {!isvrt_right, isvrt_right}, {0,0}, isalpha_right ? 1 : -1);
            SpinorbitalTensor<U> apt("ap^t", arena, group, {vrt,occ}, {0,0}, {!isvrt_left, isvrt_left}, isalpha_left ? -1 : 1);

            vector<tkv_pair<U>> pair_left{{orbleft_dummy, 1}};
            vector<tkv_pair<U>> pair_right{{orbright_dummy, 1}};

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

                if (orbright != orbleft)
                {
                 b(1)[  "a"] -= T(1)[  "ak"]*apt["k"];
                 b(2)["abi"] -= T(2)["abik"]*apt["k"];

                 e(1)[  "a"] -= L(1)[  "ka"]*ap["k"];
                 e(2)["iab"] -= L(2)["ikab"]*ap["k"];
                }
            }

            else if((isvrt_right) && (!isvrt_left))
            {
                /*
                 *  ab...    abe...
                 * b  (e) = t
                 *  ijk...   ijk...
                 */

                b(1)[  "a"]  =              apt["a"]; 
                b(1)[  "a"] -= T(1)[  "ak"]*ap["k"];
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
                e(3)["ijabc"] =  L(2)[  "ijab"]*apt["c"];

                e(1)[  "a"]  -= L(1)[  "ka"]*ap["k"];  
                e(2)["iab"]  -= L(2)["ikab"]*ap["k"]; 

            }
            else if((!isvrt_right) && (isvrt_left))
            {
                /*
                 * b (m) = d
                 *  i       im
                 */
                b(1)["a"] = ap["a"];
                b(1)[  "a"] -= T(1)[  "ak"]*apt["k"]; 
                b(2)["abi"] = -T(2)["abik"]*apt["k"]; 

                /*
                 *  ijk...   ijk...
                 * e  (e) = l
                 *  ab...    abe...
                 */

                e(1)[  "a"]   =               ap["a"];
                e(1)[  "a"]  -= L(1)[  "ka"]*apt["k"];
                e(2)["iab"]  = -L(2)["ikab"]*apt["k"];

                e(1)[  "a"]  +=   Dab[  "ea"]*ap["e"];
                e(2)["iab"]  +=  L(1)[  "ia"]*ap["b"];
                e(2)["iab"]  += Gieab["eiba"]*ap["e"];
                e(3)["ijabc"] =  L(2)[  "ijab"]*ap["c"];
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
                e(3)["ijabc"]  =  L(2)[  "ijab"]*apt["c"];

               if (orbright != orbleft)
              {
                b(1)["a"] += apt["a"];  
                e(1)[  "a"]  +=               ap["a"]; 
                e(1)[  "a"]  +=   Dab[  "ea"]*ap["e"];
                e(2)["iab"]  +=  L(1)[  "ia"]*ap["b"];
                e(2)["iab"]  += Gieab["eiba"]*ap["e"];
                e(3)["ijabc"] +=  L(2)[  "ijab"]*ap["c"];
              }
            }

              int number_of_vectors = nA*nA*nI + nA ; 
              this->puttmp("lanczos", new Lanczos<U,X>(lanczos_config, number_of_vectors));

              RL = b ;
              LL = e ;
            
             /* Evaluate norm 
              */ 

              U norm = sqrt(aquarius::abs(scalar(RL*LL))); 

              norm_ea[nspin].emplace_back(norm*norm) ;

              RL /= norm;
              LL /= norm;

              this->log(arena) << "print norm: " << norm << endl ;

              Iterative<U>::run(dag, arena);

              nvec_lanczos = alpha.size() ; 

              alpha_ea[nspin][uppertriangle].resize(nvec_lanczos) ;
              beta_ea[nspin][uppertriangle].resize(nvec_lanczos) ;
              gamma_ea[nspin][uppertriangle].resize(nvec_lanczos) ;

              for (int i=0 ; i < nvec_lanczos ; i++){
                 alpha_ea[nspin][uppertriangle][i] = alpha[i] ;
                 beta_ea[nspin][uppertriangle][i] = beta[i] ;
                 gamma_ea[nspin][uppertriangle][i] = gamma[i] ;
              }
              uppertriangle +=1 ;
            }
          } 

         if (arena.rank == 0)
         {

             for (int nspin = 0 ; nspin < maxspin ; nspin++){
             stringstream stream1;
             stream1 << "alpha_ea_"<<nspin<<"_"<<element_start<<"_"<<element_end<< ".txt";
             string fileName1 = stream1.str();
             std::ofstream alphafile;
             alphafile.open (fileName1, ofstream::out);

             stringstream stream2;
             stream2 << "beta_ea_"<<nspin<<"_"<<element_start<<"_"<<element_end<< ".txt";
             string fileName2 = stream2.str();
             std::ofstream betafile;
             betafile.open (fileName2, ofstream::out);

             stringstream stream3;
             stream3 << "gamma_ea_"<<nspin<<"_"<<element_start<<"_"<<element_end<< ".txt";
             string fileName3 = stream3.str();
             std::ofstream gammafile;
             gammafile.open (fileName3, ofstream::out);

             stringstream stream4;
             stream4 << "norm_ea_"<<nspin<<"_"<<element_start<<"_"<<element_end<< ".txt";
             string fileName4 = stream4.str();
             std::ofstream normfile;
             normfile.open (fileName4, ofstream::out);

             stringstream stream5;
             stream5 << "lanczos_ea_"<<nspin<<"_"<<element_start<<"_"<<element_end<< ".txt";
             string fileName5 = stream5.str();
             std::ofstream lanczoseafile;
             lanczoseafile.open (fileName5, ofstream::out);

              for (int i=element_start ; i < element_end ; i++){
                for (int j=0 ; j < alpha_ea[nspin][i-element_start].size() ; j++){
                   alphafile << nspin << " " << i << " " << j << " " << setprecision(12) <<  alpha_ea[nspin][i-element_start][j] << endl ;
                   betafile << nspin << " " << i << " " << j << " " << setprecision(12) << beta_ea[nspin][i-element_start][j] << endl ;
                   gammafile << nspin << " " << i << " " << j << " " << setprecision(12) << gamma_ea[nspin][i-element_start][j] << endl ;
                }
                   normfile << nspin << " " << setprecision(12) <<  norm_ea[nspin][i-element_start] << endl ;
                   lanczoseafile << nspin << " " << i << " " << alpha_ea[nspin][i-element_start].size() << endl ;
               }
             alphafile.close();
             betafile.close();
             gammafile.close();
             normfile.close();
             lanczoseafile.close();

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
            auto& XMCI = this->template gettmp<SpinorbitalTensor<U>>("XMCI");
            auto& XACE = this->template gettmp<SpinorbitalTensor<U>>("XACE");

            auto& GIM = this->template gettmp<SpinorbitalTensor<U>>("GIM");
            auto& XIMC = this->template gettmp<SpinorbitalTensor<U>>("XIMC");
            auto& XEAC = this->template gettmp<SpinorbitalTensor<U>>("XEAC");
            auto& XEMO = this->template gettmp<SpinorbitalTensor<U>>("XEMO");

            auto& lanczos = this->template gettmp<Lanczos<U,X>>("lanczos");

            auto& RL = this->template gettmp< ExcitationOperator<U,3,2>>("RL");
            auto& LL = this->template gettmp< DeexcitationOperator<U,3,2>>("LL");
            auto& Z  = this->template gettmp<  ExcitationOperator<U,3,2>>("Z");
            auto& Y  = this->template gettmp<  DeexcitationOperator<U,3,2>>("Y");
            auto& b  = this->template gettmp<  ExcitationOperator<U,3,2>>("b");
            auto& e  = this->template gettmp<DeexcitationOperator<U,3,2>>("e");

            auto& alpha = this->template gettmp<unique_vector<U>> ("alpha");
            auto& beta  = this->template gettmp<unique_vector<U>> ("beta");
            auto& gamma = this->template gettmp<unique_vector<U>> ("gamma");

                XMI[  "m"] = -0.5*WMNEF["mnef"]*RL(2)["efn"];

                XMCI[  "mci"]  =      WAMEI["cmei"]*RL(1)[     "e"];
                XMCI[  "mci"] +=      WMNEJ["nmei"]*RL(2)[   "ecn"];
                XMCI[  "mci"] +=  0.5*WMNEF["mnef"]*RL(3)[ "efcin"];
                XMCI[  "mci"] +=  0.5*WAMEF["cmef"]*RL(2)[  "efi"];

                XACE[  "ace"]  =      WABEF["acef"]*RL(1)[     "f"];
                XACE[  "ace"] +=      WAMEF["amef"]*RL(2)[   "fcm"];
                XACE[  "ace"] -=  0.5*WMNEF["mnef"]*RL(3)[ "afcmn"];

                Z(1)[  "a"]  =       FAE[  "ae"]*RL(1)[  "e"];
                Z(1)[  "a"] -=       FME[  "me"]*RL(2)["aem"];
                Z(1)[  "a"] -= 0.5*WAMEF["amef"]*RL(2)["efm"];
                Z(1)[  "a"] += 0.25*WMNEF["mnef"]*RL(3)[ "efamn"];

                Z(2)["abi"]   =     WABEJ["baei"]*RL(1)[  "e"];
                Z(2)["abi"]  +=       FAE[  "ae"]*RL(2)["ebi"];
                Z(2)["abi"]  -=       FMI[  "mi"]*RL(2)["abm"];
                Z(2)["abi"]  -=       XMI[  "m"]*T(2)["abim"];
                Z(2)["abi"]  += 0.5*WABEF["abef"]*RL(2)["efi"];
                Z(2)["abi"]  -=     WAMEI["amei"]*RL(2)["ebm"];
                Z(2)["abi"]  +=        FME[  "me"]*RL(3)[ "eabmi"];
                Z(2)["abi"]  +=  0.5*WAMEF["amef"]*RL(3)[ "efbim"];
                Z(2)["abi"]  -=  0.5*WMNEJ["mnei"]*RL(3)[ "eabmn"];

                Z(3)["abcij"]  =      WABEJ["abej"]*RL(2)[   "eci"];
                Z(3)["abcij"] -=      WAMIJ["amij"]*RL(2)[   "bcm"];
                Z(3)["abcij"] +=        FAE[  "ae"]*RL(3)[ "ebcij"];
                Z(3)["abcij"] -=        FMI[  "mi"]*RL(3)[ "abcmj"];
                Z(3)["abcij"] -=      WAMEI["amei"]*RL(3)[ "ebcmj"];
                Z(3)["abcij"] +=  0.5*WABEF["abef"]*RL(3)[ "efcij"];
                Z(3)["abcij"] +=  0.5*WMNIJ["mnij"]*RL(3)[ "abcmn"];
                Z(3)["abcij"] -=       XMCI[ "mci"]*T(2)[  "abmj"];
                Z(3)["abcij"] -=       XACE[ "ace"]*T(2)[  "beji"];


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

            /* new intermediates for SDT
             */

                XEAC[  "eac"]  =        -0.5*T(2)[  "efmj"]* LL(3)["mjafc"];
                XIMC[  "imc"]  =        +0.5*T(2)[  "efmn"]* LL(3)["inefc"];

                XEMO[    "amo"] =     -T(2)[  "afno"]*XIMC[  "nmf"];


                Y(1)[  "a"] -=  WAMEI["eman"]*XIMC["nme"]; //correct 
                Y(1)[  "a"] -=  0.5*WABEF[  "efga"]*XEAC[  "gef"]; //correct

                Y(1)[  "a"] -=  0.5*WMNEF[  "miea"]*XEMO[  "emi"]; //correct

                Y(2)[  "iab"] -=     WAMEF[  "emba"]*XIMC[  "ime"]; //correct 
                Y(2)[  "iab"] +=     WMNEJ["mian"]*XIMC[  "nmb"]; //correct

                Y(2)[  "iab"] +=     WAMEF["fibe"]*XEAC[  "efa"]; //correct 

                Y(2)[  "iab"] -=        0.5*WAMIJ[  "eimn"]* LL(3)["mneab"];
                Y(2)[  "iab"] +=        0.5*WABEJ[  "efam"]* LL(3)["imefb"];

                Y(3)["ijabc"]  =            WMNEF[  "ijab"]* LL(1)[    "c"];
                Y(3)["ijabc"] +=              FME[    "ia"]* LL(2)[  "jbc"];
                Y(3)["ijabc"] +=            WAMEF[  "ejab"]* LL(2)[  "iec"];
                Y(3)["ijabc"] -=            WMNEJ[  "ijam"]* LL(2)[  "mbc"];

                Y(3)["ijabc"] +=            WMNEF[  "ijeb"]*XEAC[  "eac"]; // done...
                Y(3)["ijabc"] -=            WMNEF[  "mjab"]*XIMC[  "imc"]; // done.. 

                Y(3)["ijabc"] +=              FAE[    "ea"]* LL(3)["ijebc"];
                Y(3)["ijabc"] -=              FMI[    "im"]* LL(3)["mjabc"];
                Y(3)["ijabc"] +=        0.5*WABEF[  "efab"]* LL(3)["ijefc"];
                Y(3)["ijabc"] +=        0.5*WMNIJ[  "ijmn"]* LL(3)["mnabc"];
                Y(3)["ijabc"] -=            WAMEI[  "eibm"]* LL(3)["mjaec"];

 
            lanczos.extrapolate_tridiagonal(RL, LL, Z, Y, alpha, beta, gamma);

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

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::CCSDTEAGF_LANCZOS_D_T);
REGISTER_TASK(aquarius::cc::CCSDTEAGF_LANCZOS_D_T<double>, "ccsdteagf_lanczos_d_t",spec);
