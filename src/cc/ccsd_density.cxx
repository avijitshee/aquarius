#include "util/global.hpp"

#include "time/time.hpp"
#include "task/task.hpp"
#include "operator/2eoperator.hpp"
#include "operator/excitationoperator.hpp"
#include "operator/deexcitationoperator.hpp"

using namespace aquarius::op;
using namespace aquarius::input;
using namespace aquarius::tensor;
using namespace aquarius::task;
using namespace aquarius::time;

namespace aquarius
{
namespace cc
{

template <typename U>
class CCSDDensity : public Task
{
    protected:
        typedef complex_type_t<U> CU;
    public:
        CCSDDensity(const string& name, Config& config)
        : Task(name, config)
        {
            vector<Requirement> reqs;
            reqs.push_back(Requirement("moints", "H"));
            reqs.push_back(Requirement("ccsd.T", "T"));
            reqs.push_back(Requirement("ccsd.L", "L"));
            this->addProduct(Product("ccsd.D", "D", reqs));
        }

        bool run(TaskDAG& dag, const Arena& arena)
        {
            const auto& H = this->template get<TwoElectronOperator <U  >>("H");
            const auto& T = this->template get<ExcitationOperator  <U,2>>("T");
            const auto& L = this->template get<DeexcitationOperator<U,2>>("L");

            const Space& occ = H.occ;
            const Space& vrt = H.vrt;

            auto& D = this->put("D", new TwoElectronOperator<U>("D", arena, occ, vrt));

            auto& DIA = D.getIA();
            auto& DAI = D.getAI();
            auto& DAB = D.getAB();
            auto& DIJ = D.getIJ();

            auto& GIJAB = D.getIJAB();
            auto& GIJAK = D.getIJAK();
            auto& GAIBC = D.getAIBC();
            auto& GABCD = D.getABCD();
            auto& GIJKL = D.getIJKL();
            auto& GAIBJ = D.getAIBJ();
            auto& GAIJK = D.getAIJK();
            auto& GABCI = D.getABCI();
            auto& GABIJ = D.getABIJ();

            SpinorbitalTensor<U> Tau(T(2));
            Tau["abij"] += 0.5*T(1)["ai"]*T(1)["bj"];

              DIA[  "ia"]  =       L(1)[  "ia"];

              DAB[  "ab"]  = -0.5* L(2)["mnbe"]*T(2)["aemn"];

              DIJ[  "ij"]  =       L(1)[  "ie"]*T(1)[  "ej"];
              DIJ[  "ij"] +=  0.5* L(2)["imef"]*T(2)["efjm"];

              DAI[  "ai"]  =       L(1)[  "me"]*T(2)["aeim"];
              DAI[  "ai"] -=        DIJ[  "mi"]*T(1)[  "am"];
              DAI[  "ai"] +=        DAB[  "ae"]*T(1)[  "ei"];

              DAB[  "ab"] -=       L(1)[  "mb"]*T(1)[  "am"];

            GIJAB["ijab"]  =       L(2)["ijab"];

            GIJAK["ijak"]  =       L(2)["ijae"]*T(1)[  "ek"];

            GAIBC["aibc"]  =      -L(2)["mibc"]*T(1)[  "am"];

            GIJKL["ijkl"]  =  0.5* L(2)["ijef"]* Tau["efkl"];

            GABCD["abcd"]  =  0.5* L(2)["mncd"]* Tau["abmn"];

            GAIBJ["aibj"]  = -0.5* L(2)["imbe"]*T(2)["aejm"];
            GAIBJ["aibj"] -=      GIJAK["mibj"]*T(1)[  "am"];

            GAIJK["aijk"]  =       L(1)[  "ie"]*T(2)["aejk"];
            GAIJK["aijk"] +=      GIJAK["miek"]*T(2)["aejm"];
            GAIJK["aijk"] +=  0.5*GAIBC["aief"]* Tau["efjk"];
            GAIJK["aijk"] +=        DIJ[  "ik"]*T(1)[  "aj"];

            GABCI["abci"]  =      -L(1)[  "mc"]*T(2)["abmi"];
            GABCI["abci"] +=      GAIBC["amce"]*T(2)["beim"];
            GABCI["abci"] +=  0.5*GIJAK["mnci"]* Tau["abmn"];
            GABCI["abci"] +=        DAB[  "ac"]*T(1)[  "bi"];

            GABIJ["abij"]  =                     Tau["abij"];
            GABIJ["abij"] +=        DAI[  "ai"]*T(1)[  "bj"];
            GABIJ["abij"] +=  0.5*GIJKL["mnij"]* Tau["abmn"];
            GABIJ["abij"] -=        DIJ[  "mi"]*T(2)["abmj"];
            GABIJ["abij"] +=        DAB[  "ae"]*T(2)["ebij"];
            GABIJ["abij"] -=      GAIBJ["amei"]*T(2)["ebmj"];

            GAIBJ["aibj"] -=  0.5* L(2)["imbe"]*T(2)["aejm"];
            GAIBJ["aibj"] -=       L(1)[  "ib"]*T(1)[  "aj"];

              DAI[  "ai"] +=                    T(1)[  "ai"];

              DAB *= -1;
              DIJ *= -1;
            GIJAK *= -1;
            GAIBC *= -1;
            GAIJK *= -1;
            GABCI *= -1;

// build full CCSD density matrix and then diagonalize to obtain natural orbital occupancies..


        U value = 0. ; 
        int norb = occ.nalpha[0] + vrt.nalpha[0] ; 
        vector<U> density(norb*norb, 0.) ;

        for (int spin : {1,0})
        {
           vector<U> oo ;
           vector<U> ov ;
           vector<U> vo ;
           vector<U> vv ;
              
           DAI({spin,0},{0,spin})({0,0}).getAllData(vo); 
           DIA({0,spin},{spin,0}).getAllData({0,0}, ov);
           DAB({spin,0},{spin,0}).getAllData({0,0}, vv);
           DIJ({0,spin},{0,spin}).getAllData({0,0}, oo);

           int nocc = (spin == 1 ? occ.nalpha[0] : occ.nbeta[0]) ;
           int nvirt = (spin == 1 ? vrt.nalpha[0] : vrt.nbeta[0]) ;

           norb = nocc + nvirt ; 

           for (int i=0 ; i < nocc ; i++){
             oo[i*nocc+i] += 1.0 ;
           }
 
           for (int i=0 ; i < norb ; i++){
            for (int j=0 ; j < norb ; j++){
             if ((i < nocc) && (j < nocc)) density[i*norb+j] += 1.0*oo[i*nocc+j] ; 
             if ((i < nocc) && (j >= nocc)) density[i*norb+j] += 1.0*ov[i*nvirt+(j-nocc)] ; 
             if ((i >= nocc) && (j < nocc)) density[i*norb+j] += 1.0*vo[(i-nocc)*nocc+j] ; 
             if ((i >= nocc) && (j >= nocc)) density[i*norb+j] += 1.0*vv[(i-nocc)*nvirt+(j-nocc)] ; 
            }
           } 
          }

//symmetrize density matrix


            vector<U> l(norb*norb);
            vector<CU> s_tmp(norb);
            vector<U> vr_tmp(norb*norb);

            int info = geev('N', 'V', norb, density.data(), norb,
                        s_tmp.data(), l.data(), norb,
                        vr_tmp.data(), norb);
            if (info != 0) throw runtime_error(str("check diagonalization: Info in geev: %d", info));


            cout<<" #orbital occupation" <<endl ;

            for (int i=0 ; i < norb ; i++){
                printf(" %.15f\n", s_tmp[i].real());
//                value += s_tmp[i].real();
                value += density[i*norb+i];
             }

            printf("total occupancy: %.15f\n", value);
          


            U EIA = scalar(DIA*H.getIA());
            U EAI = scalar(DAI*H.getAI());
            U EAB = scalar(DAB*H.getAB());
            U EIJ = scalar(DIJ*H.getIJ());

            U EIJAB = 0.25*scalar(GIJAB*H.getIJAB());
            U EIJAK =  0.5*scalar(GIJAK*H.getIJAK());
            U EAIBC =  0.5*scalar(GAIBC*H.getAIBC());
            U EABCD = 0.25*scalar(GABCD*H.getABCD());
            U EIJKL = 0.25*scalar(GIJKL*H.getIJKL());
            U EAIBJ =      scalar(GAIBJ*H.getAIBJ());
            U EAIJK =  0.5*scalar(GAIJK*H.getAIJK());
            U EABCI =  0.5*scalar(GABCI*H.getABCI());
            U EABIJ = 0.25*scalar(GABIJ*H.getABIJ());

            printf("|AI|: %20.15f\n", DAI({0,0},{0,0}).norm(2));
            printf("|IA|: %20.15f\n", DIA({0,0},{0,0}).norm(2));
            printf("|AB|: %20.15f\n", DAB({0,0},{0,0}).norm(2));
            printf("|IJ|: %20.15f\n", DIJ({0,0},{0,0}).norm(2));
            printf("|IJAK|: %20.15f\n", GIJAK({0,1},{0,1}).norm(2));
            printf("|AIJK|: %20.15f\n", GAIJK({0,1},{0,1}).norm(2));
            printf("|AIBC|: %20.15f\n", GAIBC({1,0},{1,0}).norm(2));
            printf("|ABCI|: %20.15f\n", GABCI({1,0},{1,0}).norm(2));

            printf("\n");
            printf("EAB: %20.15f\n", EAB);
            printf("EIJ: %20.15f\n", EIJ);
            printf("EABCD: %20.15f\n", EABCD);
            printf("EIJKL: %20.15f\n", EIJKL);
            printf("EAIBJ: %20.15f\n", EAIBJ);
            printf("EAIJK: %20.15f\n", EAIJK+EIJAK);
            printf("EABCI: %20.15f\n", EABCI+EAIBC);
            printf("EABIJ: %20.15f\n", EABIJ+EIJAB);

            U E = EIA+EAI+EAB+EIJ+
                  EIJAB+EABIJ+EABCI+EAIBC+EAIJK+EIJAK+
                  EABCD+EIJKL+EAIBJ;

            printf("\n");
            printf("E: %20.15f\n", E);

            return true;
        }
        bool greater(CU i, CU j) { return i > j; }
};

}
}

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::CCSDDensity);
REGISTER_TASK(aquarius::cc::CCSDDensity<double>,"ccsd_density","");
