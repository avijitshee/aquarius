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
        void writeIntegrals(bool pvirt, bool qvirt, bool rvirt, bool svirt,
                                vector<T>& cfirst, vector<T>& csecond, vector<T>& cthird, vector<T>& cfourth,
                                tensor::SymmetryBlockedTensor<T>& tensor) ;
        void read_2e_integrals()
        {
          std::ifstream onsite("onsite.txt");
          std::istream_iterator<T> start(onsite), end;
          std::vector<T> int_onsite(start, end);
          std::cout << "Read " << int_onsite.size() << " numbers" << std::endl;

  // print the numbers to stdout
         std::cout << "numbers read in:\n";
         std::copy(int_onsite.begin(), int_onsite.end(), 
         std::ostream_iterator<double>(std::cout, " "));
         std::cout << std::endl;
         v_onsite.insert(v_onsite.begin(),int_onsite.begin(),int_onsite.end());  
       }
};

}
}


#endif
