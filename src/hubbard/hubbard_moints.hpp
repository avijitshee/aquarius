#ifndef _AQUARIUS_HUBBARD_MOINTS_HPP_
#define _AQUARIUS_HUBBARD_MOINTS_HPP_

#include "util/global.hpp"

#include "task/task.hpp"

#include "operator/2eoperator.hpp"

namespace aquarius
{
namespace hubbard
{

template <typename T>
class Hubbard_MOIntegrals : public task::Task
{
    protected:
      Hubbard_MOIntegrals(const string& name, input::Config& config);
};

}
}

#endif
