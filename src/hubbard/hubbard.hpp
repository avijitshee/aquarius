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
        int ndoc;
        int nalpha;
        int nbeta;
        int multiplicity ;
        int nirreps = 1;
        vector<double> integral_diagonal ;
        vector<double> integral_offdiagonal ;
        vector<int> alpha_array ;
        vector<int> beta_array ;
        string openshell_alpha ;
        string openshell_beta ; 

    public:
        vector<double> v_onsite ;
        Hubbard(const string& name, input::Config& config);
        int getNumAlphaElectrons()
        {
         alphastring_to_vector (alpha_array) ; 
         int countzero = std::count (alpha_array.begin(), alpha_array.end(), 0); 
         int nopenshell = alpha_array.size() ;
         return (nopenshell - countzero + ndoc);
        } ;
        int getNumBetaElectrons()
        {
         betastring_to_vector (beta_array) ; 
         int countzero = std::count (beta_array.begin(), beta_array.end(), 0); 
         int nopenshell = beta_array.size() ;
         return (nopenshell - countzero + ndoc);
        } ;

        int getNumOrbitals() const { return norb; } ;
        int getDoccOrbitals() const { return ndoc; } ;
        int getNumIrreps() const { return nirreps; } ;

        void alphastring_to_vector(vector<int>& alphavec )
        {
         alphavec.clear() ; 
        
         for (size_t i = 0; i < openshell_alpha.size(); ++i)
         {                                 
          alphavec.push_back(openshell_alpha[i] - '0'); 
         } 

        }

        void betastring_to_vector(vector<int>& betavec)
        {
         betavec.clear() ;
         for (size_t i = 0; i < openshell_beta.size(); ++i)
         {                                 
          betavec.push_back(openshell_beta[i] - '0'); 
         }
        }

        void read_1e_integrals()
        {
          std::ifstream one_diag("one_diag.txt");
          std::istream_iterator<double> start(one_diag), end;
          std::vector<double> diagonal(start, end);
          std::cout << "Read " << diagonal.size() << " numbers" << std::endl;
          std::copy(diagonal.begin(), diagonal.end(),std::back_inserter(integral_diagonal)); 
        }

        void read_2e_integrals()
        {
          std::ifstream onsite("onsite.txt");
          std::istream_iterator<double> start(onsite), end;
          std::vector<double> int_onsite(start, end);
          std::cout << "Read " << int_onsite.size() << " numbers" << std::endl;

          std::copy(int_onsite.begin(), int_onsite.end(), std::back_inserter(v_onsite)) ;
        }

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
