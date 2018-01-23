#ifndef _AQUARIUS_HUBBARD_AOMOINTS_HPP_
#define _AQUARIUS_HUBBARD_AOMOINTS_HPP_

#include "util/global.hpp"
#include "integrals/2eints.hpp"
#include "tensor/symblocked_tensor.hpp"

#include "hubbard_moints.hpp"

namespace aquarius
{
namespace hubbard
{
template <typename T>
class Hubbard_AOMOints : public Hubbard_MOIntegrals<T>
{
    public:
        Hubbard_AOMOints(const string& name, input::Config& config);

    protected:
        bool run(task::TaskDAG& dag, const Arena& arena);
        void writeIntegrals(bool pvirt, bool qvirt, bool rvirt, bool svirt,
                                vector<T>& cfirst, vector<T>& csecond, vector<T>& cthird, vector<T>& cfourth,
                                tensor::SymmetryBlockedTensor<T>& tensor) ;
};

}
}


#endif
