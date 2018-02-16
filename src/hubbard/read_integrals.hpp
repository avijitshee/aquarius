#ifndef _AQUARIUS_HUBBARD_READINTS_HPP_
#define _AQUARIUS_HUBBARD_READINTS_HPP_

#include "util/global.hpp"
#include <iostream>
#include <fstream>

#include "hubbard.hpp"
#include "input/config.hpp"
#include "task/task.hpp"
#include "operator/2eoperator.hpp"
#include "tensor/symblocked_tensor.hpp"

namespace aquarius
{
namespace hubbard
{

template <typename U>
class ReadInts : public task::Task  
{
    protected:
        vector<U> integral_diagonal ;
        vector<U> v_onsite ;

    public:
        ReadInts(const string& name, input::Config& config);
        void read_1e_integrals()
        {
          std::ifstream one_diag("one_diag.txt");
          std::istream_iterator<U> start(one_diag), end;
          std::vector<U> diagonal(start, end);
          std::cout << "Read " << diagonal.size() << " numbers" << std::endl;
          std::copy(diagonal.begin(), diagonal.end(),std::back_inserter(integral_diagonal)); 
        }

        void read_2e_integrals()
        {
          std::ifstream onsite("onsite.txt");
          std::istream_iterator<U> start(onsite), end;
          std::vector<U> int_onsite(start, end);
          std::cout << "Read " << int_onsite.size() << " numbers" << std::endl;

          std::copy(int_onsite.begin(), int_onsite.end(), std::back_inserter(v_onsite)) ;
        }

        bool run(task::TaskDAG& dag, const Arena& arena);                                                                            
};

}
}

#endif
