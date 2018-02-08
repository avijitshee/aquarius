#ifndef _AQUARIUS_HUBBARD_UHF_MODELH_COMMON_HPP_
#define _AQUARIUS_HUBBARD_UHF_MODELH_COMMON_HPP_

#include "util/global.hpp"

#include "tensor/symblocked_tensor.hpp"
#include "operator/2eoperator.hpp"
#include "integrals/1eints.hpp"
#include "input/molecule.hpp"
#include "input/config.hpp"
#include "util/iterative.hpp"
#include "convergence/diis.hpp"
#include "task/task.hpp"
#include "operator/space.hpp"

namespace aquarius
{
namespace hubbard
{

template <typename T>
class uhf_modelh : public Iterative<T>
{
    protected:
        bool frozen_core;
        T damping;
        vector<int> occ_alpha, occ_beta;
        vector<vector<T>> E_alpha, E_beta;
//        vector<real_type_t<T>> E_alpha, E_beta;
        convergence::DIIS<tensor::SymmetryBlockedTensor<T>> diis;

    public:
        uhf_modelh(const string& name, input::Config& config);

        void iterate(const Arena& arena);

        bool run(task::TaskDAG& dag, const Arena& arena);

    protected:

        vector<T> v_onsite ;
//        virtual void calcSMinusHalf() = 0;
        void calcSMinusHalf() ;

        void calcS2();

//        virtual void diagonalizeFock() = 0;
        void diagonalizeFock() ;

//        virtual void buildFock() = 0;
        void buildFock() ;

        void calcEnergy();

        void calcDensity();

        void DIISExtrap() ;

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
