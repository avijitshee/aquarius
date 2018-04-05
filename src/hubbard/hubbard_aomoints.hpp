#ifndef _AQUARIUS_HUBBARD_AOMOINTS_HPP_
#define _AQUARIUS_HUBBARD_AOMOINTS_HPP_

#include "util/global.hpp"
#include "integrals/2eints.hpp"
#include "tensor/symblocked_tensor.hpp"
#include "hubbard.hpp"

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
        vector<T> v_onsite ;
        bool run(task::TaskDAG& dag, const Arena& arena);
        void writeIntegrals(vector<T>& cfirst, vector<T>& csecond, vector<T>& cthird, vector<T>& cfourth,
                                tensor::SymmetryBlockedTensor<T>& tensor, int np, int nq, int nr, int ns) ;
        void read_2e_integrals()
        {
          std::ifstream onsite("onsite.txt");
          std::istream_iterator<T> start(onsite), end;
          std::vector<T> int_onsite(start, end);
          std::copy(int_onsite.begin(), int_onsite.end(), std::back_inserter(v_onsite)) ;
        }


};

}
}


#endif
