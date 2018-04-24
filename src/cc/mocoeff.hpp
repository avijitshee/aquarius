#ifndef _AQUARIUS_CC_MOCOEFFS_HPP_
#define _AQUARIUS_CC_MOCOEFFS_HPP_

#include "util/global.hpp"

#include "task/task.hpp"

#include "operator/2eoperator.hpp"

namespace aquarius
{
namespace cc
{

template <typename T>
class MOCoeffs : public task::Task
{
    protected:
        MOCoeffs(const string& name, input::Config& config);
};

}
}

#endif
