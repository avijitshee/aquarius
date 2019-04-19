#ifndef _AQUARIUS_SCF_PYSCF_IMPORT_COMMON_HPP_
#define _AQUARIUS_SCF_PYSCF_IMPORT_COMMON_HPP_

#include "util/global.hpp"

#include "tensor/symblocked_tensor.hpp"
#include "integrals/1eints.hpp"
#include "integrals/2eints.hpp"
#include "input/molecule.hpp"
#include "input/config.hpp"
#include "util/iterative.hpp"
#include "convergence/diis.hpp"
#include "task/task.hpp"
#include "operator/space.hpp"

namespace aquarius
{
namespace scf
{

template <typename T>
class pyscf_import : public Iterative<T>
{
    protected:
        bool frozen_core;
        T damping;
        T damp_density;
        string path_focka, path_fockb, path_overlap ;
        vector<int> occ_alpha, occ_beta;
        vector<vector<real_type_t<T>>> E_alpha, E_beta;
        convergence::DIIS<tensor::SymmetryBlockedTensor<T>> diis;

    public:
        pyscf_import(const string& name, input::Config& config);

        void iterate(const Arena& arena);

        bool run(task::TaskDAG& dag, const Arena& arena);

    protected:
//        virtual void calcSMinusHalf() = 0;
        void calcSMinusHalf();
        void get_overlap(const Arena& arena);

        void calcS2();

//        virtual void diagonalizeFock() = 0;
        void diagonalizeFock();
        void diagonalizeDensity();

//        virtual void buildFock() = 0;
        void buildFock();
        void buildFock_dalton();

        void calcEnergy();

        void calcDensity();

        void DIISExtrap();
};

}
}

#endif
