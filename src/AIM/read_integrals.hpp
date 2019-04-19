#ifndef _AQUARIUS_AIM_READINTS_HPP_
#define _AQUARIUS_AIM_READINTS_HPP_

#include "util/global.hpp"
#include <iostream>
#include <fstream>

#include "aim.hpp"
#include "input/config.hpp"
#include "task/task.hpp"
#include "operator/2eoperator.hpp"
#include "tensor/symblocked_tensor.hpp"

namespace aquarius
{
namespace aim
{

template <typename U>
class ReadInts : public task::Task  
{
    protected:
        vector<U> int_a ;
        vector<U> int_b ;
        vector<U> mo_coeff ;
        bool coeff_exists = false ;

    public:
        ReadInts(const string& name, input::Config& config);

        void read_coeff()
        {
         std::ifstream coeff("coeff.txt");
         if (coeff)
         {
          coeff_exists = true ; 
          std::istream_iterator<U> start(coeff), end;
          std::vector<U> coefficient(start, end);
          std::cout << "Read " << coefficient.size() << " numbers" << std::endl;
          std::copy(coefficient.begin(), coefficient.end(),std::back_inserter(mo_coeff)); 
         }
        }

        void read_1e_integrals(const int norb)
        {

           int_a.resize(norb*norb) ;
           int_b.resize(norb*norb) ;

           string path_fock = "fock_imp_0.txt";

           ifstream ifs(path_fock);
           string line;

           while (getline(ifs, line))
           {    
              U vala,valb;
              int p, q;
              istringstream(line) >> p >> q >> vala >> valb  ;

              int_a[p*norb+q]  = vala; 
              int_b[p*norb+q]  = valb; 

           }
        }

        bool run(task::TaskDAG& dag, const Arena& arena);                                                                            
};

}
}

#endif
