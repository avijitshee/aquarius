#ifndef _AQUARIUS_AIM_AIM_HPP_
#define _AQUARIUS_AIM_AIM_HPP_

#include "util/global.hpp"
#include <iostream>
#include <fstream>

#include "input/config.hpp"
#include "task/task.hpp"
#include "operator/2eoperator.hpp"
#include "tensor/symblocked_tensor.hpp"

namespace aquarius
{
namespace aim
{

template <typename U>
class AIM 
{
    friend class AIMTask;

    protected:
        int alpha_elec;
        int beta_elec;
        int norb;
        int nalpha;
        int nbeta;
        int nirreps = 1;

    public:
        AIM(const string& name, input::Config& config);
        int getNumAlphaElectrons() const {return alpha_elec; } ;
        int getNumBetaElectrons() const {return beta_elec; } ;

        int getNumOrbitals() const { return norb; } ;
        int getDoccOrbitals() {

         int unpaired ;
         int ndoc ;

         unpaired = abs(alpha_elec - beta_elec) ;
         ndoc = (unpaired == 0) ? alpha_elec : min(alpha_elec, beta_elec) ; 

         return ndoc; } ;
        int getNumIrreps() const { return nirreps; } ;

};

class AIMTask : public task::Task                                                                                               
{   
    public:
        AIMTask(const string& name, input::Config& config);                                                                     
        
        bool run(task::TaskDAG& dag, const Arena& arena);                                                                            
};       

}
}

#endif
