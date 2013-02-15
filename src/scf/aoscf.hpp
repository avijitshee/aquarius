/* Copyright (c) 2013, Devin Matthews
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL DEVIN MATTHEWS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE. */

#ifndef _AQUARIUS_SCF_AOSCF_HPP_
#define _AQUARIUS_SCF_AOSCF_HPP_

#include <vector>

#include "mpi.h"

#include "elemental.hpp"
#include "tensor.hpp"
#include "dist_tensor.hpp"

#include "slide/slide.hpp"
#include "input/molecule.hpp"
#include "input/config.hpp"
#include "diis/diis.hpp"
#include "util/util.h"
#include "util/blas.h"
#include "util/lapack.h"

#include "scf.hpp"
#include "cholesky.hpp"
#include "aoints.hpp"

namespace aquarius
{
namespace scf
{

class AOUHF : public UHF
{
    public:
        enum ConvergenceType {MAX_ABS, RMSD, MAD};

    protected:
        const AOIntegrals& ints;

        void buildFock();

    public:
        AOUHF(DistWorld* dw, const AOIntegrals& ints, const input::Config& config)
        : UHF(dw, ints.getMolecule(), config), ints(ints) {}
};

}
}

#endif
