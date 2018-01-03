#ifndef _AQUARIUS_hubbard_hubbard_HPP_
#define _AQUARIUS_hubbard_hubbard_HPP_

#include "util/global.hpp"

#include "task/task.hpp"
#include "operator/2eoperator.hpp"
#include "tensor/symblocked_tensor.hpp"

namespace aquarius
{
namespace hubbard
{

template <typename U>
class hubbard : public task::Task
{
    protected:
        int nelec;
        int norb;
        double radius;
        vector<vec3> gvecs;
        int nocc;
        double V;
        double L;
        double PotVm;
        int multiplicity ;
        int nirreps = 0;
        vector<double> integral_diagonal = {0.0,0.558819356316,-0.558819356316,4.45759206721,-4.45759206721,-1.47891491526,1.47891491526,-0.185401954358,0.185401954358,0.0317683411165,-0.0317683411165};
        vector<double> integral_offdiagonal = {0.553263286885,0.553263286885,0.541358378777,0.541358378777,0.488524003875,0.488524003875,0.383193040171,0.383193040171,0.23348635632,0.23348635632,0.000001};
        vector<double> v_onsite = {4.0000,4.0000,4.0000,4.0000,4.0000,4.0000,4.0000,4.0000,4.0000,4.0000,4.0000,4.0000};   

        void writeIntegrals(bool pvirt, bool qvirt, bool rvirt, bool svirt,
                            tensor::SymmetryBlockedTensor<U>& tensor);

    public:
        hubbard(const string& name, input::Config& config);
        int getNumAlphaElectrons() const { return (nelec+multiplicity)/2; } ;

        int getNumBetaElectrons() const { return (nelec-multiplicity+1)/2; } ;
        const int getNumOrbitals() const { return norb; } ;
        int getNumIrreps() const { return nirreps; } ;

        bool run(task::TaskDAG& dag, const Arena& arena);
};

}
}

#endif
