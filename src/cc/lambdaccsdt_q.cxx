#include "util/global.hpp"

#include "operator/st2eoperator.hpp"
#include "operator/deexcitationoperator.hpp"
#include "operator/excitationoperator.hpp"
#include "convergence/diis.hpp"
#include "util/iterative.hpp"
#include "task/task.hpp"

using namespace aquarius::op;
using namespace aquarius::input;
using namespace aquarius::tensor;
using namespace aquarius::task;
using namespace aquarius::time;
using namespace aquarius::symmetry;
using namespace aquarius::convergence;

namespace aquarius
{
namespace cc
{

template <typename U>
class LambdaCCSDT_Q : public Iterative<U>
{
    protected:
        DIIS<DeexcitationOperator<U,3>> diis;

    public:
        LambdaCCSDT_Q(const string& name, Config& config)
        : Iterative<U>(name, config), diis(config.get("diis"))
        {
            vector<Requirement> reqs;
            reqs.push_back(Requirement("moints", "H"));
            reqs.push_back(Requirement("ccsdt.Hbar", "Hbar"));
            reqs.push_back(Requirement("ccsdt.T", "T"));
            this->addProduct(Product("double", "energy", reqs));
            this->addProduct(Product("double", "convergence", reqs));
            this->addProduct(Product("ccsdt.L", "L", reqs));
        }

        bool run(TaskDAG& dag, const Arena& arena)
        {
            const auto& H = this->template get<TwoElectronOperator<U>>("H");
            const auto& Hbar = this->template get<STTwoElectronOperator<U>>("Hbar");

            const Space& occ = H.occ;
            const Space& vrt = H.vrt;
            const PointGroup& group = occ.group;

            this->put   (    "L", new DeexcitationOperator<U,3>("L", arena, occ, vrt));
            this->puttmp(    "Q", new ExcitationOperator  <U,3>("Q", arena, occ, vrt));
            this->puttmp(    "D", new Denominator         <U  >(H));
            this->puttmp(    "Z", new DeexcitationOperator<U,3>("Z", arena, occ, vrt));
            this->puttmp(  "DIJ", new SpinorbitalTensor   <U  >(   "D(ij)",   H.getIJ()));
            this->puttmp(  "DAB", new SpinorbitalTensor   <U  >(   "D(ab)",   H.getAB()));
            this->puttmp(  "DAI", new SpinorbitalTensor   <U  >(   "D(ai)",   H.getAI()));
            this->puttmp("GABCD", new SpinorbitalTensor   <U  >("G(ab,cd)", H.getABCD()));
            this->puttmp("GAIBJ", new SpinorbitalTensor   <U  >("G(ai,bj)", H.getAIBJ()));
            this->puttmp("GIJKL", new SpinorbitalTensor   <U  >("G(ij,kl)", H.getIJKL()));
            this->puttmp("GAIBC", new SpinorbitalTensor   <U  >("G(ai,bc)", H.getAIBC()));
            this->puttmp("GIJAK", new SpinorbitalTensor   <U  >("G(ij,ak)", H.getIJAK()));

            auto& T = this->template get   <ExcitationOperator  <U,3>>("T");
            auto& Q = this->template gettmp<ExcitationOperator  <U,3>>("Q");
            auto& L = this->template get   <DeexcitationOperator<U,3>>("L");
            auto& Z = this->template gettmp<DeexcitationOperator<U,3>>("Z");
            auto& D = this->template gettmp<Denominator         <U  >>("D");

            const SpinorbitalTensor<U>& VABIJ = H.getABIJ();
            const SpinorbitalTensor<U>& VAMEF = H.getAIBC();
            const SpinorbitalTensor<U>& VABEJ = H.getABCI();
            const SpinorbitalTensor<U>& VABEF = H.getABCD();
            const SpinorbitalTensor<U>& VMNIJ = H.getIJKL();
            const SpinorbitalTensor<U>& VMNEJ = H.getIJAK();
            const SpinorbitalTensor<U>& VAMIJ = H.getAIJK();
            const SpinorbitalTensor<U>& VAMEI = H.getAIBJ();

            SpinorbitalTensor<U> WABCEJK("W(abc,ejk)", arena, group, {vrt,occ}, {3,0}, {1,2});
            SpinorbitalTensor<U> WABMIJK("W(abm,ijk)", arena, group, {vrt,occ}, {2,1}, {0,3});

            SpinorbitalTensor<U> T4("T4", arena, group, {vrt,occ}, {4,0}, {0,4});
            SpinorbitalTensor<U> Z4("L4", arena, group, {vrt,occ}, {4,0}, {0,4});

            WABCEJK[  "abcejk"]  = 0.5*VABEF[  "abef"]*T(2)[  "fcjk"];

            WABMIJK[  "abmijk"]  =     VAMEI[  "amek"]*T(2)[  "ebij"];
            WABMIJK[  "abmijk"] -= 0.5*VMNIJ[  "nmjk"]*T(2)[  "abin"];

                 T4["abcdijkl"]  =     VABEJ[  "abej"]*T(3)["ecdikl"];
                 T4["abcdijkl"] -=     VAMIJ[  "amij"]*T(3)["bcdmkl"];

                 Z4 = T4;

                 //T4["abcdijkl"] +=   WABCEJK["abcejk"]*T(2)[  "edil"];
                 //T4["abcdijkl"] -=   WABMIJK["abmijk"]*T(2)[  "cdml"];

                 //Z4["abcdijkl"] +=     VABIJ[  "abij"]*T(2)[  "cdkl"];

            T4.weight({&D.getDA(), &D.getDI()}, {&D.getDa(), &D.getDi()});

            U E = (1.0/576.0)*scalar(Z4*T4);

            this->log(arena) << printos("energy: %18.15f\n", E) << endl;

            Q(1) = 0;
            Q(2) = 0;

            Q(3)["abcijk"]  = T4["abefijkm"]*VAMEF["cmef"];
            Q(3)["abcijk"] -= T4["abceijmn"]*VMNEJ["nmek"];

            Z(0) = 0;
            L(0) = 1;
            L(1)[    "ia"] = T(1)[    "ai"];
            L(2)[  "ijab"] = T(2)[  "abij"];
            L(3)["ijkabc"] = T(3)["abcijk"];

            Iterative<U>::run(dag, arena);

            this->put("energy", new U(this->energy()));
            this->put("convergence", new U(this->conv()));

            return true;
        }

        void iterate(const Arena& arena)
        {
            const auto& H = this->template get<STTwoElectronOperator<U>>("Hbar");

            const SpinorbitalTensor<U>&   FME =   H.getIA();
            const SpinorbitalTensor<U>&   FAE =   H.getAB();
            const SpinorbitalTensor<U>&   FMI =   H.getIJ();
            const SpinorbitalTensor<U>& WMNEF = H.getIJAB();
            const SpinorbitalTensor<U>& WAMEF = H.getAIBC();
            const SpinorbitalTensor<U>& WABEJ = H.getABCI();
            const SpinorbitalTensor<U>& WABEF = H.getABCD();
            const SpinorbitalTensor<U>& WMNIJ = H.getIJKL();
            const SpinorbitalTensor<U>& WMNEJ = H.getIJAK();
            const SpinorbitalTensor<U>& WAMIJ = H.getAIJK();
            const SpinorbitalTensor<U>& WAMEI = H.getAIBJ();

            auto& T = this->template get   <ExcitationOperator  <U,3>>("T");
            auto& Q = this->template gettmp<ExcitationOperator  <U,3>>("Q");
            auto& L = this->template get   <DeexcitationOperator<U,3>>("L");
            auto& D = this->template gettmp<Denominator         <U  >>("D");
            auto& Z = this->template gettmp<DeexcitationOperator<U,3>>("Z");

            auto&   DIJ = this->template gettmp<SpinorbitalTensor<U>>(  "DIJ");
            auto&   DAB = this->template gettmp<SpinorbitalTensor<U>>(  "DAB");
            auto&   DAI = this->template gettmp<SpinorbitalTensor<U>>(  "DAI");
            auto& GABCD = this->template gettmp<SpinorbitalTensor<U>>("GABCD");
            auto& GAIBJ = this->template gettmp<SpinorbitalTensor<U>>("GAIBJ");
            auto& GIJKL = this->template gettmp<SpinorbitalTensor<U>>("GIJKL");
            auto& GAIBC = this->template gettmp<SpinorbitalTensor<U>>("GAIBC");
            auto& GIJAK = this->template gettmp<SpinorbitalTensor<U>>("GIJAK");

            /***************************************************************************
             *
             * Intermediates for Lambda-CCSD
             */
            DIJ["ij"]  =  0.5*T(2)["efjm"]*L(2)["imef"];
            DAB["ab"]  = -0.5*T(2)["aemn"]*L(2)["mnbe"];
            /*
             **************************************************************************/

            /***************************************************************************
             *
             * Lambda-CCSD iteration
             */
            Z(1)[  "ia"]  =       FME[  "ia"];
            Z(1)[  "ia"] +=       FAE[  "ea"]*L(1)[  "ie"];
            Z(1)[  "ia"] -=       FMI[  "im"]*L(1)[  "ma"];
            Z(1)[  "ia"] -=     WAMEI["eiam"]*L(1)[  "me"];
            Z(1)[  "ia"] += 0.5*WABEJ["efam"]*L(2)["imef"];
            Z(1)[  "ia"] -= 0.5*WAMIJ["eimn"]*L(2)["mnea"];
            Z(1)[  "ia"] -=     WMNEJ["inam"]* DIJ[  "mn"];
            Z(1)[  "ia"] -=     WAMEF["fiea"]* DAB[  "ef"];

            Z(2)["ijab"]  =     WMNEF["ijab"];
            Z(2)["ijab"] +=       FME[  "ia"]*L(1)[  "jb"];
            Z(2)["ijab"] +=     WAMEF["ejab"]*L(1)[  "ie"];
            Z(2)["ijab"] -=     WMNEJ["ijam"]*L(1)[  "mb"];
            Z(2)["ijab"] +=       FAE[  "ea"]*L(2)["ijeb"];
            Z(2)["ijab"] -=       FMI[  "im"]*L(2)["mjab"];
            Z(2)["ijab"] += 0.5*WABEF["efab"]*L(2)["ijef"];
            Z(2)["ijab"] += 0.5*WMNIJ["ijmn"]*L(2)["mnab"];
            Z(2)["ijab"] +=     WAMEI["eiam"]*L(2)["mjbe"];
            Z(2)["ijab"] -=     WMNEF["mjab"]* DIJ[  "im"];
            Z(2)["ijab"] +=     WMNEF["ijeb"]* DAB[  "ea"];
            /*
             **************************************************************************/

            /***************************************************************************
             *
             * Intermediates for Lambda-CCSDT
             */
              DIJ[  "ij"]  =  (1.0/12.0)*T(3)["efgjmn"]* L(3)["imnefg"];
              DAB[  "ab"]  = -(1.0/12.0)*T(3)["aefmno"]* L(3)["mnobef"];

            GABCD["abcd"]  =   (1.0/6.0)*T(3)["abemno"]* L(3)["mnocde"];
            GAIBJ["aibj"]  =       -0.25*T(3)["aefjmn"]* L(3)["imnbef"];
            GIJKL["ijkl"]  =   (1.0/6.0)*T(3)["efgklm"]* L(3)["ijmefg"];

            GIJAK["ijak"]  =         0.5*T(2)[  "efkm"]* L(3)["ijmaef"];
            GAIBC["aibc"]  =        -0.5*T(2)[  "aemn"]* L(3)["minbce"];

              DAI[  "ai"]  =        0.25*T(3)["aefimn"]* L(2)[  "mnef"];
              DAI[  "ai"] -=         0.5*T(2)[  "eamn"]*GIJAK[  "mnei"];
            /*
             **************************************************************************/

            /***************************************************************************
             *
             * Lambda-CCSDT iteration
             */
            Z(1)[    "ia"] +=       FME[  "ie"]*  DAB[    "ea"];
            Z(1)[    "ia"] -=       FME[  "ma"]*  DIJ[    "im"];
            Z(1)[    "ia"] -=     WMNEJ["inam"]*  DIJ[    "mn"];
            Z(1)[    "ia"] -=     WAMEF["fiea"]*  DAB[    "ef"];
            Z(1)[    "ia"] +=     WMNEF["miea"]*  DAI[    "em"];
            Z(1)[    "ia"] -= 0.5*WABEF["efga"]*GAIBC[  "gief"];
            Z(1)[    "ia"] +=     WAMEI["eifm"]*GAIBC[  "fmea"];
            Z(1)[    "ia"] -=     WAMEI["eman"]*GIJAK[  "inem"];
            Z(1)[    "ia"] += 0.5*WMNIJ["imno"]*GIJAK[  "noam"];
            Z(1)[    "ia"] -= 0.5*WAMEF["gief"]*GABCD[  "efga"];
            Z(1)[    "ia"] +=     WAMEF["fmea"]*GAIBJ[  "eifm"];
            Z(1)[    "ia"] -=     WMNEJ["inem"]*GAIBJ[  "eman"];
            Z(1)[    "ia"] += 0.5*WMNEJ["noam"]*GIJKL[  "imno"];

            Z(2)[  "ijab"] -=     WMNEF["mjab"]*  DIJ[    "im"];
            Z(2)[  "ijab"] +=     WMNEF["ijeb"]*  DAB[    "ea"];
            Z(2)[  "ijab"] += 0.5*WMNEF["ijef"]*GABCD[  "efab"];
            Z(2)[  "ijab"] +=     WMNEF["imea"]*GAIBJ[  "ejbm"];
            Z(2)[  "ijab"] += 0.5*WMNEF["mnab"]*GIJKL[  "ijmn"];
            Z(2)[  "ijab"] -=     WAMEF["fiae"]*GAIBC[  "ejbf"];
            Z(2)[  "ijab"] -=     WMNEJ["ijem"]*GAIBC[  "emab"];
            Z(2)[  "ijab"] -=     WAMEF["emab"]*GIJAK[  "ijem"];
            Z(2)[  "ijab"] -=     WMNEJ["niam"]*GIJAK[  "mjbn"];
            Z(2)[  "ijab"] += 0.5*WABEJ["efbm"]* L(3)["ijmaef"];
            Z(2)[  "ijab"] -= 0.5*WAMIJ["ejnm"]* L(3)["imnabe"];

            Z(3)["ijkabc"]  =     WMNEF["ijab"]* L(1)[    "kc"];
            Z(3)["ijkabc"] +=       FME[  "ia"]* L(2)[  "jkbc"];
            Z(3)["ijkabc"] +=     WAMEF["ekbc"]* L(2)[  "ijae"];
            Z(3)["ijkabc"] -=     WMNEJ["ijam"]* L(2)[  "mkbc"];
            Z(3)["ijkabc"] +=     WMNEF["ijae"]*GAIBC[  "ekbc"];
            Z(3)["ijkabc"] -=     WMNEF["mkbc"]*GIJAK[  "ijam"];
            Z(3)["ijkabc"] +=       FAE[  "ea"]* L(3)["ijkebc"];
            Z(3)["ijkabc"] -=       FMI[  "im"]* L(3)["mjkabc"];
            Z(3)["ijkabc"] += 0.5*WABEF["efab"]* L(3)["ijkefc"];
            Z(3)["ijkabc"] += 0.5*WMNIJ["ijmn"]* L(3)["mnkabc"];
            Z(3)["ijkabc"] +=     WAMEI["eiam"]* L(3)["mjkbec"];
            /*
             **************************************************************************/

            Z(1)[    "ia"] += Q(1)[    "ai"];
            Z(2)[  "ijab"] += Q(2)[  "abij"];
            Z(3)["ijkabc"] += Q(3)["abcijk"];

            Z.weight(D);
            L += Z;

            this->energy() = 0.25*real(scalar(conj(WMNEF)*L(2)));
            this->conv() = Z.norm(00);

            diis.extrapolate(L, Z);
        }
};

}
}

static const char* spec = R"!(

convergence?
    double 1e-9,
max_iterations?
    int 50,
conv_type?
    enum { MAXE, RMSE, MAE },
diis?
{
    damping?
        double 0.0,
    start?
        int 1,
    order?
        int 5,
    jacobi?
        bool false
}

)!";

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::LambdaCCSDT_Q);
REGISTER_TASK(aquarius::cc::LambdaCCSDT_Q<double>,"lambdaccsdt(q)",spec);
