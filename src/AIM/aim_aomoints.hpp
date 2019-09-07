#ifndef _AQUARIUS_AIM_AOMOINTS_HPP_
#define _AQUARIUS_AIM_AOMOINTS_HPP_

#include "util/global.hpp"
#include "integrals/2eints.hpp"
#include "tensor/symblocked_tensor.hpp"
#include "aim.hpp"

#include "aim_moints.hpp"

namespace aquarius
{
namespace aim
{
template <typename T>
class AIM_AOMOints : public AIM_MOIntegrals<T>
{
    public:
        AIM_AOMOints(const string& name, input::Config& config);

    protected:
        int nimporbs ; 
        string path ;
        bool run(task::TaskDAG& dag, const Arena& arena);
};

}
}


#endif
