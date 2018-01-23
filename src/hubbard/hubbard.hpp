#ifndef _AQUARIUS_HUBBARD_HUBBARD_HPP_
#define _AQUARIUS_HUBBARD_HUBBARD_HPP_

#include "util/global.hpp"

#include "input/config.hpp"
#include "task/task.hpp"
#include "operator/2eoperator.hpp"
#include "tensor/symblocked_tensor.hpp"

namespace aquarius
{
namespace hubbard
{

template <typename U>
class Hubbard : public task::Task
{
    protected:
        int nelec;
        int norb;
        int dimension;
        double radius;
        vector<vec3> gvecs;
        int nocc;
        int multiplicity ;
        int nirreps = 0;
        double V;
        double L;
        double PotVm;
//20     vector<double> integral_diagonal = {-2.0,0.558819356316,-0.558819356316,4.45759206721,-4.45759206721,-1.47891491526,1.47891491526,-0.185401954358,0.185401954358,0.0317683411165,-0.0317683411165,0.0};
//20     vector<double> integral_offdiagonal = {0.553263286885,0.553263286885,0.541358378777,0.541358378777,0.488524003875,0.488524003875,0.383193040171,0.383193040171,0.23348635632,0.23348635632,0.000001};
        vector<double> integral_diagonal = {-2.0,1.43097547642473,-1.43097547642473,-0.64433186304585,0.64433186304585,-0.23866779703493,0.23866779703493,0.07888212711970,-0.07888212711970,0.02024015034907,-0.02024015034907,0.0};
        vector<double> integral_offdiagonal = {0.63010100439710,0.63010100439710,0.59907771548103,0.59907771548103,0.39793740954679,0.39793740954679,0.24333180010029,0.24333180010029,0.14708211245860,0.14708211245860,-0.08798636465276};

//100       vector<double> integral_diagonal = {-2.0,-1.63134856258178,1.63134856258178,0.96275471595923,-0.96275471595923,0.47397981641010,-0.47397981641010,0.20482077754233,-0.20482077754233,-0.06666351318360,0.06666351318360,-0.0};
//100       vector<double> integral_offdiagonal = {0.48459733938535,0.48459733938535,0.58515347988139,0.58515347988139,0.47524987987444,0.47524987987444,0.34681997613413,0.34681997613413,0.24687933044638,0.24687933044638,0.17689631990092};
        vector<double> v_onsite = {4.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000};   

    public:
        Hubbard(const string& name, input::Config& config);
        int getNumAlphaElectrons() const { return (nelec+multiplicity)/2; } ;

        int getNumBetaElectrons() const { return (nelec-multiplicity+1)/2; } ;
        int getNumOrbitals() const { return norb; } ;
        int getNumIrreps() const { return nirreps; } ;

        bool run(task::TaskDAG& dag, const Arena& arena);
};

}
}

#endif
