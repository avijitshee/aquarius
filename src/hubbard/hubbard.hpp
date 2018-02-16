#ifndef _AQUARIUS_HUBBARD_HUBBARD_HPP_
#define _AQUARIUS_HUBBARD_HUBBARD_HPP_

#include "util/global.hpp"
#include <iostream>
#include <fstream>

#include "input/config.hpp"
#include "task/task.hpp"
#include "operator/2eoperator.hpp"
#include "tensor/symblocked_tensor.hpp"

namespace aquarius
{
namespace hubbard
{

template <typename U>
class Hubbard 
{
    friend class HubbardTask;

    protected:
        int nelec;
        int norb;
        int dimension;
        double radius;
        vector<vec3> gvecs;
        int nocc;
        int nalpha;
        int nbeta;
        int multiplicity ;
        int nirreps = 1;
        vector<U> integral_diagonal ;
        vector<U> integral_offdiagonal ;

    public:
        vector<U> v_onsite ;
        Hubbard(const string& name, input::Config& config);
        int getNumAlphaElectrons() const { return (nelec+multiplicity)/2; } ;

        int getNumBetaElectrons() const { return (nelec-multiplicity+1)/2; } ;
        int getNumOrbitals() const { return norb; } ;
        int getNumIrreps() const { return nirreps; } ;
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

//        bool run(task::TaskDAG& dag, const Arena& arena);
};

class HubbardTask : public task::Task                                                                                               
{   
    public:
        HubbardTask(const string& name, input::Config& config);                                                                     
        
        bool run(task::TaskDAG& dag, const Arena& arena);                                                                            
};       

}
}

#endif
