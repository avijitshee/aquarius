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

#include "spinorbital.hpp"

using namespace aquarius::autocc;

namespace libtensor
{

template<>
double scalar(const IndexedTensor< SpinorbitalTensor<DistTensor> >& other)
{
    DistTensor dt(0, NULL, NULL, other.tensor_.getSpinCase(0).dw);
    SpinorbitalTensor<DistTensor> sodt(",");
    sodt.addSpinCase(dt, ",", "");
    int n;
    double ret, * val;
    sodt[""] = other;
    dt.getAllData(&n, &val);
    assert(n==1);
    ret = val[0];
    free(val);
    return ret;
}

template<>
double scalar(const IndexedTensorMult< SpinorbitalTensor<DistTensor> >& other)
{
    DistTensor dt(0, NULL, NULL, other.A_.tensor_.getSpinCase(0).dw);
    SpinorbitalTensor<DistTensor> sodt(",");
    sodt.addSpinCase(dt, ",", "");
    int n;
    double ret, * val;
    sodt[""] = other;
    dt.getAllData(&n, &val);
    assert(n==1);
    ret = val[0];
    free(val);
    return ret;
}

}
