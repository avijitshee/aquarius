#ifndef _AQUARIUS_aim_MOINTS_HPP_
#define _AQUARIUS_aim_MOINTS_HPP_

#include "util/global.hpp"

#include "task/task.hpp"

#include "operator/2eoperator.hpp"

namespace aquarius
{
namespace aim
{

template <typename T>
class AIM_MOIntegrals : public task::Task
{
    protected:
      AIM_MOIntegrals(const string& name, input::Config& config);
};

}
}

#endif
