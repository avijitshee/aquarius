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

#include <cmath>

#include "symmetry.hpp"

namespace aquarius
{
namespace tensor
{

#define PI 3.1415926535897932385

mat3x3 Rotation(vec3 axis, double degrees)
{
    axis.normalize();
    double c = cos(degrees*PI/180);
    double s = sin(degrees*PI/180);
    double x = axis[0];
    double y = axis[1];
    double z = axis[2];
    return mat3x3(x*x*(1-c)+c  , x*y*(1-c)-z*s, x*z*(1-c)+y*s,
                  x*y*(1-c)+z*s, y*y*(1-c)+c  , y*z*(1-c)-x*s,
                  x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c  );
}

mat3x3 Reflection(vec3 axis)
{
    axis.normalize();
    double x = axis[0];
    double y = axis[1];
    double z = axis[2];
    return mat3x3(1-2*x*x,  -2*x*y,  -2*x*z,
                   -2*x*y, 1-2*y*y,  -2*y*z,
                   -2*x*z,  -2*y*z, 1-2*z*z);
}

mat3x3 Identity()
{
    return mat3x3(1, 0, 0,
                  0, 1, 0,
                  0, 0, 1);
}

mat3x3 Inversion()
{
    return mat3x3(-1,  0,  0,
                   0, -1,  0,
                   0,  0, -1);
}

template <int n>
mat3x3 C(const vec3& axis)
{
    return Rotation(axis, 360/n);
}

template <int n>
mat3x3 S(const vec3& axis)
{
    return Rotation(axis, 360/n)*Reflection(axis);
}

/*
 * C1
 */
#define A Representation(PointGroup::C1, 0x1)

static const char *c1_name = "C1";
static const Representation c1_dirprd[] = {A};
static const Representation c1_irreps[] = {A};
static const char *c1_irrep_names[] = {"A"};
static const double c1_characters[] = {1};
static const int c1_irrep_degen[] = {1};
static const mat3x3 c1_ops[] = {Identity()};
static const char *c1_op_names[] = {"E"};

const PointGroup PointGroup::C1 = PointGroup(1, 1, c1_name, c1_dirprd, c1_irreps,
                                             c1_irrep_names, c1_characters, c1_irrep_degen,
                                             c1_ops, c1_op_names);

#undef A

/*
 * Cs
 */
#define Ap  Representation(PointGroup::Cs, 0x1)
#define App Representation(PointGroup::Cs, 0x2)

static const char *cs_name = "Cs";
static const Representation cs_dirprd[] = { Ap,App,
                                           App, Ap};
static const Representation cs_irreps[] = {Ap,App};
static const char *cs_irrep_names[] = {"Ap","App"};
static const double cs_characters[] = { 1, 1,
                                        1,-1};
static const int cs_irrep_degen[] = {1,1};
static const mat3x3 cs_ops[] = {Identity(),
                                Reflection(vec3(0,0,1))};
static const char *cs_op_names[] = {"E", "sxy"};

const PointGroup PointGroup::Cs = PointGroup(2, 2, cs_name, cs_dirprd, cs_irreps,
                                             cs_irrep_names, cs_characters, cs_irrep_degen,
                                             cs_ops, cs_op_names);

#undef Ap
#undef App

/*
 * Ci
 */
#define Ag Representation(PointGroup::Ci, 0x1)
#define Au Representation(PointGroup::Ci, 0x2)

static const char *ci_name = "Ci";
static const Representation ci_dirprd[] = {Ag,Au,
                                           Au,Ag};
static const Representation ci_irreps[] = {Ag,Au};
static const char *ci_irrep_names[] = {"Ag","Au"};
static const double ci_characters[] = { 1, 1,
                                        1,-1};
static const int ci_irrep_degen[] = {1,1};
static const mat3x3 ci_ops[] = {Identity(),
                                Inversion()};
static const char *ci_op_names[] = {"E", "i"};

const PointGroup PointGroup::Ci = PointGroup(2, 2, ci_name, ci_dirprd, ci_irreps,
                                             ci_irrep_names, ci_characters, ci_irrep_degen,
                                             ci_ops, ci_op_names);

#undef Ag
#undef Au

/*
 * Td
 */
#define A1 Representation(PointGroup::Td, 0x01)
#define A2 Representation(PointGroup::Td, 0x02)
#define E  Representation(PointGroup::Td, 0x04)
#define T1 Representation(PointGroup::Td, 0x08)
#define T2 Representation(PointGroup::Td, 0x10)

static const char *td_name = "Td";
static const Representation td_dirprd[] = {A1, A2,       E,         T1,         T2,
                                           A2, A1,       E,         T2,         T1,
                                            E,  E, A1+A2+E,      T1+T2,      T1+T2,
                                           T1, T2,   T1+T2, A1+E+T1+T2, A2+E+T1+T2,
                                           T2, T1,   T1+T2, A2+E+T1+T2, A1+E+T1+T2};
static const Representation td_irreps[] = {A1,A2,E,T1,T2};
static const char *td_irrep_names[] = {"A1","A2","E","T1","T2"};
static const double td_characters[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
                                        2,-1,-1,-1,-1,-1,-1,-1,-1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        3, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,
                                        3, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1};
static const int td_irrep_degen[] = {1,1,2,3,3};
static const mat3x3 td_ops[] = {Identity(),
                                C<3>(vec3( 1, 1, 1)),
                                C<3>(vec3(-1,-1,-1)),
                                C<3>(vec3( 1,-1,-1)),
                                C<3>(vec3(-1, 1, 1)),
                                C<3>(vec3(-1, 1,-1)),
                                C<3>(vec3( 1,-1, 1)),
                                C<3>(vec3(-1,-1, 1)),
                                C<3>(vec3( 1, 1,-1)),
                                C<2>(vec3( 1, 0, 0)),
                                C<2>(vec3( 0, 1, 0)),
                                C<2>(vec3( 0, 0, 1)),
                                S<4>(vec3( 1, 0, 0)),
                                S<4>(vec3(-1, 0, 0)),
                                S<4>(vec3( 0, 1, 0)),
                                S<4>(vec3( 0,-1, 0)),
                                S<4>(vec3( 0, 0, 1)),
                                S<4>(vec3( 0, 0,-1)),
                                Reflection(vec3( 1, 1, 0)),
                                Reflection(vec3( 1,-1, 0)),
                                Reflection(vec3( 1, 0, 1)),
                                Reflection(vec3( 1, 0,-1)),
                                Reflection(vec3( 0, 1, 1)),
                                Reflection(vec3( 0, 1,-1))};
static const char *td_op_names[] = {"E",
                                    "C3+++", "C3+++^2", "C3+--", "C3+--^2", "C3-+-", "C3-+-^2", "C3--+", "C3--+^2",
                                    "C2x", "C2y", "C2z",
                                    "S4x", "S4x^3", "S4y", "S4y^3", "S4z", "S4z^3",
                                    "sx+y", "sx-y", "sx+z", "sx-z", "sy+z", "sy-z"};

const PointGroup PointGroup::Td = PointGroup(24, 5, td_name, td_dirprd, td_irreps,
                                             td_irrep_names, td_characters, td_irrep_degen,
                                             td_ops, td_op_names);

#undef A1
#undef A2
#undef E
#undef T1
#undef T2

/*
 * Oh
 */

/*
 * Ih
 */

/*
 * C2
 */
#define A1 Representation(PointGroup::C2, 0x1)
#define A2 Representation(PointGroup::C2, 0x2)

static const char *c2_name = "C2";
static const Representation c2_dirprd[] = {A1,A1,
                                           A2,A2};
static const Representation c2_irreps[] = {A1,A2};
static const char *c2_irrep_names[] = {"Ag","Au"};
static const double c2_characters[] = { 1, 1,
                                        1,-1};
static const int c2_irrep_degen[] = {1,1};
static const mat3x3 c2_ops[] = {Identity(),
                                C<2>(vec3(0,0,1))};
static const char *c2_op_names[] = {"E", "C2"};

const PointGroup PointGroup::C2 = PointGroup(2, 2, c2_name, c2_dirprd, c2_irreps,
                                             c2_irrep_names, c2_characters, c2_irrep_degen,
                                             c2_ops, c2_op_names);

#undef A1
#undef A2

/*
 * C3
 */

/*
 * C4
 */

/*
 * C5
 */

/*
 * C6
 */

/*
 * C2v
 */

/*
 * C3v
 */

/*
 * C4v
 */

/*
 * C5v
 */

/*
 * C6v
 */

/*
 * C2h
 */

/*
 * C3h
 */

/*
 * C4h
 */

/*
 * C5h
 */

/*
 * C6h
 */

/*
 * D2
 */

/*
 * D3
 */

/*
 * D4
 */

/*
 * D5
 */

/*
 * D6
 */

/*
 * D2h
 */

/*
 * D3h
 */

/*
 * D4h
 */

/*
 * D5h
 */

/*
 * D6h
 */

/*
 * D2d
 */

/*
 * D3d
 */

/*
 * D4d
 */

/*
 * D5d
 */

/*
 * D6d
 */

/*
 * S4
 */

/*
 * S6
 */

/*
 * S8
 */

/*
 * S10
 */

/*
 * S12
 */

}
}
