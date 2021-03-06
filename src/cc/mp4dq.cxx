#include "util/global.hpp"

#include "time/time.hpp"
#include "task/task.hpp"
#include "operator/2eoperator.hpp"
#include "operator/excitationoperator.hpp"
#include "operator/denominator.hpp"

using namespace aquarius::op;
using namespace aquarius::input;
using namespace aquarius::tensor;
using namespace aquarius::time;
using namespace aquarius::task;

namespace aquarius
{
namespace cc
{

template <typename U>
class MP4DQ : public Task
{
    public:
        MP4DQ(const string& name, Config& config)
        : Task(name, config)
        {
            vector<Requirement> reqs;
            reqs.push_back(Requirement("moints", "H"));
            addProduct(Product("double", "mp2", reqs));
            addProduct(Product("double", "mp3", reqs));
            addProduct(Product("double", "mp4d", reqs));
            addProduct(Product("double", "mp4q", reqs));
            addProduct(Product("double", "energy", reqs));
            addProduct(Product("double", "S2", reqs));
            addProduct(Product("double", "multiplicity", reqs));
        }

        bool run(TaskDAG& dag, const Arena& arena)
        {
            const auto& H = get<TwoElectronOperator<U>>("H");

            const Space& occ = H.occ;
            const Space& vrt = H.vrt;

            auto& T = puttmp("T", new ExcitationOperator<U,2>("T", arena, occ, vrt));
            auto& D = puttmp("D", new Denominator       <U  >(H));
            auto& Z = puttmp("Z", new ExcitationOperator<U,2>("Z", arena, occ, vrt));

            Z(0) = (U)0.0;
            T(0) = (U)0.0;
            T(1) = (U)0.0;
            T(2) = H.getABIJ();

            T.weight(D);

            double energy = 0.25*real(scalar(H.getABIJ()*T(2)));
            double mp2energy = energy;

            Logger::log(arena) << "MP2 energy = " << setprecision(15) << energy << endl;
            put("mp2", new U(energy));

            auto& H_ = get<TwoElectronOperator<U>>("H");

            TwoElectronOperator<U> Hnew("W", H_, TwoElectronOperator<U>::AB|
                                         TwoElectronOperator<U>::IJ|
                                         TwoElectronOperator<U>::IJKL|
                                         TwoElectronOperator<U>::AIBJ);

            SpinorbitalTensor<U>& FAE = Hnew.getAB();
            SpinorbitalTensor<U>& FMI = Hnew.getIJ();
            SpinorbitalTensor<U>& WMNEF = Hnew.getIJAB();
            SpinorbitalTensor<U>& WABEF = Hnew.getABCD();
            SpinorbitalTensor<U>& WMNIJ = Hnew.getIJKL();
            SpinorbitalTensor<U>& WAMEI = Hnew.getAIBJ();

            Z(2)["abij"] = WMNEF["ijab"];
            Z(2)["abij"] += FAE["af"]*T(2)["fbij"];
            Z(2)["abij"] -= FMI["ni"]*T(2)["abnj"];
            Z(2)["abij"] += 0.5*WABEF["abef"]*T(2)["efij"];
            Z(2)["abij"] += 0.5*WMNIJ["mnij"]*T(2)["abmn"];
            Z(2)["abij"] -= WAMEI["amei"]*T(2)["ebmj"];

            Z.weight(D);
            T += Z;

            energy = 0.25*real(scalar(H.getABIJ()*T(2)));
            double mp3energy = energy;

            //Logger::log(arena) << "MP3 energy = " << setprecision(15) << energy << endl;
            Logger::log(arena) << "MP3 correlation energy = " << setprecision(15) << energy - mp2energy << endl;
            put("mp3", new U(energy));

            auto& Znew = gettmp<ExcitationOperator<U,2>>("Z");

            Znew(2)["abij"] = WMNEF["ijab"];
            Znew(2)["abij"] += FAE["af"]*T(2)["fbij"];
            Znew(2)["abij"] -= FMI["ni"]*T(2)["abnj"];
            Znew(2)["abij"] += 0.5*WABEF["abef"]*T(2)["efij"];
            Znew(2)["abij"] += 0.5*WMNIJ["mnij"]*T(2)["abmn"];
            Znew(2)["abij"] -= WAMEI["amei"]*T(2)["ebmj"];

            Znew.weight(D);
            T += Znew;

            energy = 0.25*real(scalar(H.getABIJ()*T(2)));
            double mp4denergy = energy - mp3energy;

            //Logger::log(arena) << "LCCD(2) energy = " << setprecision(15) << energy << endl;
            Logger::log(arena) << "MP4D correlation energy = " << setprecision(15) << energy - mp3energy << endl;
            put("mp4d", new U(energy));

            auto& Tccd = get   <ExcitationOperator<U,2>>("T");
            auto& Dccd = gettmp<Denominator       <U  >>("D");
            auto& Zccd = gettmp<ExcitationOperator<U,2>>("Z");

            Zccd(0) = (U)0.0;
            Tccd(0) = (U)0.0;
            Tccd(1) = (U)0.0;
            Tccd(2) = H.getABIJ();

            Tccd.weight(Dccd);

            FMI["mi"] += 0.5*WMNEF["mnef"]*T(2)["efin"];
            WMNIJ["mnij"] += 0.5*WMNEF["mnef"]*T(2)["efij"];
            FAE["ae"] -= 0.5*WMNEF["mnef"]*T(2)["afmn"];
            WAMEI["amei"] -= 0.5*WMNEF["mnef"]*T(2)["afin"];
            Z(2)["abij"] = WMNEF["ijab"];
            Z(2)["abij"] += FAE["af"]*T(2)["fbij"];
            Z(2)["abij"] -= FMI["ni"]*T(2)["abnj"];
            Z(2)["abij"] += 0.5*WABEF["abef"]*T(2)["efij"];
            Z(2)["abij"] += 0.5*WMNIJ["mnij"]*T(2)["abmn"];
            Z(2)["abij"] -= WAMEI["amei"]*T(2)["ebmj"];

            Z.weight(D);
            T += Z;

            energy = 0.25*real(scalar(H.getABIJ()*T(2)));
            double mp4qenergy = energy - mp3energy;

            //Logger::log(arena) << "CCD(1) energy = " << setprecision(15) << energy << endl;
            Logger::log(arena) << "MP4Q correlation energy = " << setprecision(15) << energy - mp3energy << endl;
            Logger::log(arena) << "MP4DQ correlation energy = " << setprecision(15) << mp4denergy + mp4qenergy << endl;
            put("mp4q", new U( energy));

            /*
            if (isUsed("S2") || isUsed("multiplicity"))
            {
                double s2 = getProjectedS2(occ, vrt, T(1), T(2));
                double mult = sqrt(4*s2+1);

                put("S2", new Scalar(arena, s2));
                put("multiplicity", new Scalar(arena, mult));
            }
            */

            put("energy", new U(energy));

            return true;
        }
};

}
}

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::MP4DQ);
REGISTER_TASK(aquarius::cc::MP4DQ<double>,"mp4dq");
