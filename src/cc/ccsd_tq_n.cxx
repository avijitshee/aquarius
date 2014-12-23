/* Copyright (c) 2014, Devin Matthews
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

#include "ccsd_tq_n.hpp"

using namespace std;
using namespace aquarius::op;
using namespace aquarius::cc;
using namespace aquarius::input;
using namespace aquarius::tensor;
using namespace aquarius::task;
using namespace aquarius::time;

template <typename U>
CCSD_TQ_N<U>::CCSD_TQ_N(const string& name, const Config& config)
: Task("ccsd(tq-n)", name)
{
    vector<Requirement> reqs;
    reqs.push_back(Requirement("moints", "H"));
    reqs.push_back(Requirement("ccsd.Hbar", "Hbar"));
    reqs.push_back(Requirement("ccsd.T", "T"));
    reqs.push_back(Requirement("ccsd.L", "L"));
    this->addProduct(Product("double", "E(2)", reqs));
    this->addProduct(Product("double", "E(3)", reqs));
    this->addProduct(Product("double", "E(4)", reqs));
}

template <typename U>
void CCSD_TQ_N<U>::run(task::TaskDAG& dag, const Arena& arena)
{
    const TwoElectronOperator<U>& H = this->template get<TwoElectronOperator<U>>("H");
    const STTwoElectronOperator<U>& Hbar = this->template get<STTwoElectronOperator<U>>("Hbar");

    const Space& occ = H.occ;
    const Space& vrt = H.vrt;

    Denominator<U> D(H);
    const ExcitationOperator  <U,2>& T = this->template get<ExcitationOperator  <U,2>>("T");
    const DeexcitationOperator<U,2>& L = this->template get<DeexcitationOperator<U,2>>("L");

    SpinorbitalTensor<U> FME(Hbar.getIA());
    SpinorbitalTensor<U> FAE(Hbar.getAB());
    SpinorbitalTensor<U> FMI(Hbar.getIJ());
    FME -= H.getIA();
    FAE -= H.getAB();
    FMI -= H.getIJ();

    const SpinorbitalTensor<U>& WMNEF = Hbar.getIJAB();
    const SpinorbitalTensor<U>& WAMEF = Hbar.getAIBC();
    const SpinorbitalTensor<U>& WABEJ = Hbar.getABCI();
    const SpinorbitalTensor<U>& WABEF = Hbar.getABCD();
    const SpinorbitalTensor<U>& WMNIJ = Hbar.getIJKL();
    const SpinorbitalTensor<U>& WMNEJ = Hbar.getIJAK();
    const SpinorbitalTensor<U>& WAMIJ = Hbar.getAIJK();
    const SpinorbitalTensor<U>& WAMEI = Hbar.getAIBJ();

    ExcitationOperator<U,4> T_1("T^(1)", arena, occ, vrt);
    ExcitationOperator<U,4> T_2("T^(2)", arena, occ, vrt);
    ExcitationOperator<U,4> T_3("T^(3)", arena, occ, vrt);

    DeexcitationOperator<U,4> L_1("L^(1)", arena, occ, vrt);
    DeexcitationOperator<U,4> L_2("L^(2)", arena, occ, vrt);
    DeexcitationOperator<U,4> L_3("L^(3)", arena, occ, vrt);

    ExcitationOperator<U,4> DT_1("DT^(1)", arena, occ, vrt);
    ExcitationOperator<U,4> DT_2("DT^(2)", arena, occ, vrt);
    ExcitationOperator<U,4> DT_3("DT^(3)", arena, occ, vrt);

    DeexcitationOperator<U,4> DL_1("DL^(1)", arena, occ, vrt);
    DeexcitationOperator<U,4> DL_2("DL^(2)", arena, occ, vrt);
    DeexcitationOperator<U,4> DL_3("DL^(3)", arena, occ, vrt);

    SpinorbitalTensor<U> WTWABEJ(WABEJ);
    WTWABEJ["abej"] += FME["me"]*T(2)["abmj"];

    SpinorbitalTensor<U> WABEJ_1(WABEJ);
    SpinorbitalTensor<U> WAMIJ_1(WAMIJ);

    SpinorbitalTensor<U> FME_2(FME);
    SpinorbitalTensor<U> FMI_2(FMI);
    SpinorbitalTensor<U> FAE_2(FAE);
    SpinorbitalTensor<U> WAMEI_2(WAMEI);
    SpinorbitalTensor<U> WMNIJ_2(WMNIJ);
    SpinorbitalTensor<U> WABEF_2(WABEF);
    SpinorbitalTensor<U> WABEJ_2(WABEJ);
    SpinorbitalTensor<U> WAMIJ_2(WAMIJ);
    SpinorbitalTensor<U> WMNEJ_2(WMNEJ);
    SpinorbitalTensor<U> WAMEF_2(WAMEF);

    SpinorbitalTensor<U> DIJ_1(FMI);
    SpinorbitalTensor<U> DAB_1(FAE);
    SpinorbitalTensor<U> DAI_1(T(1));
    SpinorbitalTensor<U> GIJAK_1(WMNEJ);
    SpinorbitalTensor<U> GAIBC_1(WAMEF);

    SpinorbitalTensor<U> DIJ_2(FMI);
    SpinorbitalTensor<U> DAB_2(FAE);
    SpinorbitalTensor<U> DTWIJ_2(FMI);
    SpinorbitalTensor<U> DTWAB_2(FAE);
    SpinorbitalTensor<U> DAI_2(T(1));
    SpinorbitalTensor<U> GIJKL_2(WMNIJ);
    SpinorbitalTensor<U> GAIBJ_2(WAMEI);
    SpinorbitalTensor<U> GABCD_2(WABEF);
    SpinorbitalTensor<U> GIJAK_2(WMNEJ);
    SpinorbitalTensor<U> GAIBC_2(WAMEF);

    SpinorbitalTensor<U> Tau(T(2));
    SpinorbitalTensor<U> DIJ(FMI);
    SpinorbitalTensor<U> DAB(FAE);
    SpinorbitalTensor<U> FTWMI_2(FMI);
    SpinorbitalTensor<U> FTWAE_2(FAE);
    SpinorbitalTensor<U> WTWAMEI_2(WAMEI);
    SpinorbitalTensor<U> WTWABEJ_2(WABEJ);
    SpinorbitalTensor<U> WTWAMIJ_2(WAMIJ);
    ExcitationOperator<U,4> Z("Z", arena, occ, vrt);

    /***************************************************************************
     *
     * T^(1)
     *
     **************************************************************************/

    DT_1(3)["abcijk"]  = WTWABEJ["bcek"]*T(2)["aeij"];
    DT_1(3)["abcijk"] -=   WAMIJ["bmjk"]*T(2)["acim"];

    T_1 = DT_1;
    T_1.weight(D);

    /***************************************************************************
     *
     * T^(2)
     *
     **************************************************************************/

    WAMIJ_1[  "amij"]  =  0.5*WMNEF["mnef"]*T_1(3)["aefijn"];
    WABEJ_1[  "abej"]  = -0.5*WMNEF["mnef"]*T_1(3)["afbmnj"];

    DT_2(1)[    "ai"]  = 0.25*WMNEF["mnef"]*T_1(3)["aefimn"];

    DT_2(2)[  "abij"]  =  0.5*WAMEF["bmef"]*T_1(3)["aefijm"];
    DT_2(2)[  "abij"] -=  0.5*WMNEJ["mnej"]*T_1(3)["abeinm"];
    DT_2(2)[  "abij"] +=        FME[  "me"]*T_1(3)["abeijm"];

    DT_2(3)["abcijk"]  =    WABEJ_1["bcek"]*  T(2)[  "aeij"];
    DT_2(3)["abcijk"] -=    WAMIJ_1["bmjk"]*  T(2)[  "acim"];
    DT_2(3)["abcijk"] +=        FAE[  "ce"]*T_1(3)["abeijk"];
    DT_2(3)["abcijk"] -=        FMI[  "mk"]*T_1(3)["abcijm"];
    DT_2(3)["abcijk"] +=  0.5*WABEF["abef"]*T_1(3)["efcijk"];
    DT_2(3)["abcijk"] +=  0.5*WMNIJ["mnij"]*T_1(3)["abcmnk"];
    DT_2(3)["abcijk"] +=      WAMEI["amei"]*T_1(3)["ebcjmk"];

    T_2 = DT_2;
    T_2.weight(D);

    /***************************************************************************
     *
     * T^(3)
     *
     **************************************************************************/

      FME_2[    "me"]  =       WMNEF["mnef"]*T_2(1)[    "fn"];

      FMI_2[    "mi"]  =   0.5*WMNEF["mnef"]*T_2(2)[  "efin"];
      FMI_2[    "mi"] +=       WMNEJ["nmfi"]*T_2(1)[    "fn"];

      FAE_2[    "ae"]  =  -0.5*WMNEF["mnef"]*T_2(2)[  "afmn"];
      FAE_2[    "ae"] +=       WAMEF["amef"]*T_2(1)[    "fm"];

    WAMIJ_2[  "amij"]  =       WMNEJ["nmej"]*T_2(2)[  "aein"];
    WAMIJ_2[  "amij"] +=   0.5*WAMEF["amef"]*T_2(2)[  "efij"];
    WAMIJ_2[  "amij"] -=       WMNIJ["nmij"]*T_2(1)[    "an"];
    WAMIJ_2[  "amij"] +=       WAMEI["amej"]*T_2(1)[    "ei"];
    WAMIJ_2[  "amij"] +=   0.5*WMNEF["mnef"]*T_2(3)["aefijn"];
    WAMIJ_2[  "amij"] +=       FME_2[  "me"]*  T(2)[  "aeij"];

    WABEJ_2[  "abej"]  =       WAMEF["amef"]*T_2(2)[  "fbmj"];
    WABEJ_2[  "abej"] +=   0.5*WMNEJ["mnej"]*T_2(2)[  "abmn"];
    WABEJ_2[  "abej"] +=       WABEF["abef"]*T_2(1)[    "fj"];
    WABEJ_2[  "abej"] -=       WAMEI["amej"]*T_2(1)[    "bm"];
    WABEJ_2[  "abej"] -=   0.5*WMNEF["mnef"]*T_2(3)["afbmnj"];

    WAMEI_2[  "amei"]  =       WMNEF["mnef"]*T_2(2)[  "afni"];
    WAMEI_2[  "amei"] +=       WAMEF["amef"]*T_2(1)[    "fi"];
    WAMEI_2[  "amei"] -=       WMNEJ["nmei"]*T_2(1)[    "an"];

    WMNIJ_2[  "mnij"]  =   0.5*WMNEF["mnef"]*T_2(2)[  "efij"];
    WMNIJ_2[  "mnij"] +=       WMNEJ["mnej"]*T_2(1)[    "ei"];

    WABEF_2[  "abef"]  =   0.5*WMNEF["mnef"]*T_2(2)[  "abmn"];
    WABEF_2[  "abef"] -=       WAMEF["amef"]*T_2(1)[    "bm"];

    DT_3(1)[    "ai"]  =         FAE[  "ae"]*T_2(1)[    "ei"];
    DT_3(1)[    "ai"] -=         FMI[  "mi"]*T_2(1)[    "am"];
    DT_3(1)[    "ai"] -=       WAMEI["amei"]*T_2(1)[    "em"];
    DT_3(1)[    "ai"] +=         FME[  "me"]*T_2(2)[  "aeim"];
    DT_3(1)[    "ai"] +=   0.5*WAMEF["amef"]*T_2(2)[  "efim"];
    DT_3(1)[    "ai"] -=   0.5*WMNEJ["mnei"]*T_2(2)[  "eamn"];
    DT_3(1)[    "ai"] +=  0.25*WMNEF["mnef"]*T_2(3)["aefimn"];

    DT_3(2)[  "abij"]  =       FAE_2[  "af"]*  T(2)[  "fbij"];
    DT_3(2)[  "abij"] -=       FMI_2[  "ni"]*  T(2)[  "abnj"];
    DT_3(2)[  "abij"] +=       WABEJ["abej"]*T_2(1)[    "ei"];
    DT_3(2)[  "abij"] -=       WAMIJ["amij"]*T_2(1)[    "bm"];
    DT_3(2)[  "abij"] +=         FAE[  "af"]*T_2(2)[  "fbij"];
    DT_3(2)[  "abij"] -=         FMI[  "ni"]*T_2(2)[  "abnj"];
    DT_3(2)[  "abij"] +=   0.5*WABEF["abef"]*T_2(2)[  "efij"];
    DT_3(2)[  "abij"] +=   0.5*WMNIJ["mnij"]*T_2(2)[  "abmn"];
    DT_3(2)[  "abij"] +=       WAMEI["amei"]*T_2(2)[  "ebjm"];
    DT_3(2)[  "abij"] +=   0.5*WAMEF["bmef"]*T_2(3)["aefijm"];
    DT_3(2)[  "abij"] -=   0.5*WMNEJ["mnej"]*T_2(3)["abeinm"];
    DT_3(2)[  "abij"] +=         FME[  "me"]*T_2(3)["abeijm"];

    DT_3(3)["abcijk"]  =     WABEJ_2["bcek"]*  T(2)[  "aeij"];
    DT_3(3)["abcijk"] -=     WAMIJ_2["bmjk"]*  T(2)[  "acim"];
    DT_3(3)["abcijk"] +=       WABEJ["bcek"]*T_2(2)[  "aeij"];
    DT_3(3)["abcijk"] -=       WAMIJ["bmjk"]*T_2(2)[  "acim"];
    DT_3(3)["abcijk"] +=         FAE[  "ce"]*T_2(3)["abeijk"];
    DT_3(3)["abcijk"] -=         FMI[  "mk"]*T_2(3)["abcijm"];
    DT_3(3)["abcijk"] +=   0.5*WABEF["abef"]*T_2(3)["efcijk"];
    DT_3(3)["abcijk"] +=   0.5*WMNIJ["mnij"]*T_2(3)["abcmnk"];
    DT_3(3)["abcijk"] +=       WAMEI["amei"]*T_2(3)["ebcjmk"];

    T_3 = DT_3;
    T_3.weight(D);

    /***************************************************************************
     *
     * Complete W^(2)
     *
     **************************************************************************/

    WMNEJ_2["mnej"]  =  WMNEF["mnef"]*T_2(1)[  "fj"];
    WAMEF_2["amef"]  = -WMNEF["nmef"]*T_2(1)[  "an"];
    WABEJ_2["abej"] -=  FME_2[  "me"]*  T(2)["abmj"];

    /***************************************************************************
     *
     * L^(1)
     *
     **************************************************************************/

    DL_1(3)["ijkabc"]  = WMNEF["ijab"]*L(1)[  "kc"];
    DL_1(3)["ijkabc"] +=   FME[  "ia"]*L(2)["jkbc"];
    DL_1(3)["ijkabc"] += WAMEF["ekbc"]*L(2)["ijae"];
    DL_1(3)["ijkabc"] -= WMNEJ["ijam"]*L(2)["mkbc"];

    L_1 = DL_1;
    L_1.weight(D);

    /***************************************************************************
     *
     * L^(2)
     *
     **************************************************************************/

    GIJAK_1[  "ijak"]  =  (1.0/ 2.0)*   T(2)[  "efkm"]* L_1(3)["ijmaef"];
    GAIBC_1[  "aibc"]  = -(1.0/ 2.0)*   T(2)[  "aemn"]* L_1(3)["minbce"];

      DAI_1[    "ai"]  =  (1.0/ 4.0)* T_1(3)["efamni"]*   L(2)[  "mnef"];
      DAI_1[    "ai"] -=  (1.0/ 2.0)*   T(2)[  "eamn"]*GIJAK_1[  "mnei"];

    DL_2(1)[    "ia"]  =  (1.0/ 2.0)*WABEJ_1[  "efam"]*   L(2)[  "imef"];
    DL_2(1)[    "ia"] -=  (1.0/ 2.0)*WAMIJ_1[  "eimn"]*   L(2)[  "mnea"];
    DL_2(1)[    "ia"] +=               WMNEF[  "miea"]*  DAI_1[    "em"];
    DL_2(1)[    "ia"] -=  (1.0/ 2.0)*  WABEF[  "efga"]*GAIBC_1[  "gief"];
    DL_2(1)[    "ia"] +=               WAMEI[  "eifm"]*GAIBC_1[  "fmea"];
    DL_2(1)[    "ia"] -=               WAMEI[  "eman"]*GIJAK_1[  "inem"];
    DL_2(1)[    "ia"] +=  (1.0/ 2.0)*  WMNIJ[  "imno"]*GIJAK_1[  "noam"];

    DL_2(2)[  "ijab"]  = -             WAMEF[  "fiae"]*GAIBC_1[  "ejbf"];
    DL_2(2)[  "ijab"] -=               WMNEJ[  "ijem"]*GAIBC_1[  "emab"];
    DL_2(2)[  "ijab"] -=               WAMEF[  "emab"]*GIJAK_1[  "ijem"];
    DL_2(2)[  "ijab"] -=               WMNEJ[  "niam"]*GIJAK_1[  "mjbn"];
    DL_2(2)[  "ijab"] +=  (1.0/ 2.0)*  WABEJ[  "efbm"]* L_1(3)["ijmaef"];
    DL_2(2)[  "ijab"] -=  (1.0/ 2.0)*  WAMIJ[  "ejnm"]* L_1(3)["imnabe"];

    DL_2(3)["ijkabc"]  =               WMNEF[  "ijae"]*GAIBC_1[  "ekbc"];
    DL_2(3)["ijkabc"] -=               WMNEF[  "mkbc"]*GIJAK_1[  "ijam"];
    DL_2(3)["ijkabc"] +=                 FAE[    "ea"]* L_1(3)["ijkebc"];
    DL_2(3)["ijkabc"] -=                 FMI[    "im"]* L_1(3)["mjkabc"];
    DL_2(3)["ijkabc"] +=  (1.0/ 2.0)*  WABEF[  "efab"]* L_1(3)["ijkefc"];
    DL_2(3)["ijkabc"] +=  (1.0/ 2.0)*  WMNIJ[  "ijmn"]* L_1(3)["mnkabc"];
    DL_2(3)["ijkabc"] +=               WAMEI[  "eiam"]* L_1(3)["mjkbec"];

    L_2 = DL_2;
    L_2.weight(D);

    /***************************************************************************
     *
     * L^(3)
     *
     **************************************************************************/

        DIJ[    "ij"]  =  (1.0/ 2.0)*   T(2)[  "efjm"]*   L(2)[  "imef"];
        DAB[    "ab"]  = -(1.0/ 2.0)*   T(2)[  "aemn"]*   L(2)[  "mnbe"];

    DTWIJ_2[    "ij"]  =  (1.0/12.0)* T_1(3)["efgjmn"]* L_1(3)["imnefg"];
    DTWIJ_2[    "ij"] +=  (1.0/ 2.0)* T_2(2)[  "efjm"]*   L(2)[  "imef"];

    DTWAB_2[    "ab"]  = -(1.0/12.0)* T_1(3)["aefmno"]* L_1(3)["mnobef"];
    DTWAB_2[    "ab"] -=  (1.0/ 2.0)* T_2(2)[  "aemn"]*   L(2)[  "mnbe"];

      DIJ_2[    "ij"]  =             DTWIJ_2[    "ij"];
      DIJ_2[    "ij"] +=  (1.0/ 2.0)*   T(2)[  "efjm"]* L_2(2)[  "imef"];

      DAB_2[    "ab"]  =             DTWAB_2[    "ab"];
      DAB_2[    "ab"] -=  (1.0/ 2.0)*   T(2)[  "aemn"]* L_2(2)[  "mnbe"];

    GABCD_2[  "abcd"]  =  (1.0/ 6.0)* T_1(3)["abemno"]* L_1(3)["mnocde"];
    GAIBJ_2[  "aibj"]  = -(1.0/ 4.0)* T_1(3)["aefjmn"]* L_1(3)["imnbef"];
    GIJKL_2[  "ijkl"]  =  (1.0/ 6.0)* T_1(3)["efgklm"]* L_1(3)["ijmefg"];

    GIJAK_2[  "ijak"]  =  (1.0/ 2.0)*   T(2)[  "efkm"]* L_2(3)["ijmaef"];
    GAIBC_2[  "aibc"]  = -(1.0/ 2.0)*   T(2)[  "aemn"]* L_2(3)["minbce"];

      DAI_2[    "ai"]  =  (1.0/ 4.0)* T_2(3)["efamni"]*   L(2)[  "mnef"];
      DAI_2[    "ai"] +=              T_2(1)[    "ei"]*    DAB[    "ae"];
      DAI_2[    "ai"] -=              T_2(1)[    "am"]*    DIJ[    "mi"];
      DAI_2[    "ai"] -=  (1.0/ 2.0)*   T(2)[  "eamn"]*GIJAK_2[  "mnei"];

    DL_3(1)[    "ia"]  =               FME_2[    "ia"];
    DL_3(1)[    "ia"] +=               FAE_2[    "ea"]*   L(1)[    "ie"];
    DL_3(1)[    "ia"] -=               FMI_2[    "im"]*   L(1)[    "ma"];
    DL_3(1)[    "ia"] -=             WAMEI_2[  "eiam"]*   L(1)[    "me"];
    DL_3(1)[    "ia"] +=  (1.0/ 2.0)*WABEJ_2[  "efam"]*   L(2)[  "imef"];
    DL_3(1)[    "ia"] -=  (1.0/ 2.0)*WAMIJ_2[  "eimn"]*   L(2)[  "mnea"];
    DL_3(1)[    "ia"] +=                 FME[    "ie"]*DTWAB_2[    "ea"];
    DL_3(1)[    "ia"] -=                 FME[    "ma"]*DTWIJ_2[    "im"];
    DL_3(1)[    "ia"] -=               WMNEJ[  "inam"]*  DIJ_2[    "mn"];
    DL_3(1)[    "ia"] -=               WAMEF[  "fiea"]*  DAB_2[    "ef"];
    DL_3(1)[    "ia"] +=               WMNEF[  "miea"]*  DAI_2[    "em"];
    DL_3(1)[    "ia"] -=  (1.0/ 2.0)*  WABEF[  "efga"]*GAIBC_2[  "gief"];
    DL_3(1)[    "ia"] +=               WAMEI[  "eifm"]*GAIBC_2[  "fmea"];
    DL_3(1)[    "ia"] -=               WAMEI[  "eman"]*GIJAK_2[  "inem"];
    DL_3(1)[    "ia"] +=  (1.0/ 2.0)*  WMNIJ[  "imno"]*GIJAK_2[  "noam"];
    DL_3(1)[    "ia"] -=  (1.0/ 2.0)*  WAMEF[  "gief"]*GABCD_2[  "efga"];
    DL_3(1)[    "ia"] +=               WAMEF[  "fmea"]*GAIBJ_2[  "eifm"];
    DL_3(1)[    "ia"] -=               WMNEJ[  "inem"]*GAIBJ_2[  "eman"];
    DL_3(1)[    "ia"] +=  (1.0/ 2.0)*  WMNEJ[  "noam"]*GIJKL_2[  "imno"];
    DL_3(1)[    "ia"] +=                 FAE[    "ea"]* L_2(1)[    "ie"];
    DL_3(1)[    "ia"] -=                 FMI[    "im"]* L_2(1)[    "ma"];
    DL_3(1)[    "ia"] -=               WAMEI[  "eiam"]* L_2(1)[    "me"];
    DL_3(1)[    "ia"] +=  (1.0/ 2.0)*  WABEJ[  "efam"]* L_2(2)[  "imef"];
    DL_3(1)[    "ia"] -=  (1.0/ 2.0)*  WAMIJ[  "einm"]* L_2(2)[  "mnae"];

    DL_3(2)[  "ijab"]  =               FME_2[    "ia"]*   L(1)[    "jb"];
    DL_3(2)[  "ijab"] +=             WAMEF_2[  "ejab"]*   L(1)[    "ie"];
    DL_3(2)[  "ijab"] -=             WMNEJ_2[  "ijam"]*   L(1)[    "mb"];
    DL_3(2)[  "ijab"] +=               FAE_2[    "ea"]*   L(2)[  "ijeb"];
    DL_3(2)[  "ijab"] -=               FMI_2[    "im"]*   L(2)[  "mjab"];
    DL_3(2)[  "ijab"] +=  (1.0/ 2.0)*WABEF_2[  "efab"]*   L(2)[  "ijef"];
    DL_3(2)[  "ijab"] +=  (1.0/ 2.0)*WMNIJ_2[  "ijmn"]*   L(2)[  "mnab"];
    DL_3(2)[  "ijab"] +=             WAMEI_2[  "eiam"]*   L(2)[  "mjbe"];
    DL_3(2)[  "ijab"] +=  (1.0/ 2.0)*WABEJ_1[  "efbm"]* L_1(3)["ijmaef"];
    DL_3(2)[  "ijab"] -=  (1.0/ 2.0)*WAMIJ_1[  "ejnm"]* L_1(3)["imnabe"];
    DL_3(2)[  "ijab"] +=                 FME[    "ia"]* L_2(1)[    "jb"];
    DL_3(2)[  "ijab"] +=               WAMEF[  "ejab"]* L_2(1)[    "ie"];
    DL_3(2)[  "ijab"] -=               WMNEJ[  "ijam"]* L_2(1)[    "mb"];
    DL_3(2)[  "ijab"] -=               WMNEF[  "mjab"]*  DIJ_2[    "im"];
    DL_3(2)[  "ijab"] +=               WMNEF[  "ijeb"]*  DAB_2[    "ea"];
    DL_3(2)[  "ijab"] +=  (1.0/ 2.0)*  WMNEF[  "ijef"]*GABCD_2[  "efab"];
    DL_3(2)[  "ijab"] +=               WMNEF[  "imea"]*GAIBJ_2[  "ejbm"];
    DL_3(2)[  "ijab"] +=  (1.0/ 2.0)*  WMNEF[  "mnab"]*GIJKL_2[  "ijmn"];
    DL_3(2)[  "ijab"] -=               WAMEF[  "fiae"]*GAIBC_2[  "ejbf"];
    DL_3(2)[  "ijab"] -=               WMNEJ[  "ijem"]*GAIBC_2[  "emab"];
    DL_3(2)[  "ijab"] -=               WAMEF[  "emab"]*GIJAK_2[  "ijem"];
    DL_3(2)[  "ijab"] -=               WMNEJ[  "niam"]*GIJAK_2[  "mjbn"];
    DL_3(2)[  "ijab"] +=                 FAE[    "ea"]* L_2(2)[  "ijeb"];
    DL_3(2)[  "ijab"] -=                 FMI[    "im"]* L_2(2)[  "mjab"];
    DL_3(2)[  "ijab"] +=  (1.0/ 2.0)*  WABEF[  "efab"]* L_2(2)[  "ijef"];
    DL_3(2)[  "ijab"] +=  (1.0/ 2.0)*  WMNIJ[  "ijmn"]* L_2(2)[  "mnab"];
    DL_3(2)[  "ijab"] +=               WAMEI[  "eiam"]* L_2(2)[  "mjbe"];
    DL_3(2)[  "ijab"] +=  (1.0/ 2.0)*  WABEJ[  "efbm"]* L_2(3)["ijmaef"];
    DL_3(2)[  "ijab"] -=  (1.0/ 2.0)*  WAMIJ[  "ejnm"]* L_2(3)["imnabe"];

    DL_3(3)["ijkabc"]  =               FME_2[    "ia"]*   L(2)[  "jkbc"];
    DL_3(3)["ijkabc"] +=             WAMEF_2[  "ekbc"]*   L(2)[  "ijae"];
    DL_3(3)["ijkabc"] -=             WMNEJ_2[  "ijam"]*   L(2)[  "mkbc"];
    DL_3(3)["ijkabc"] +=               WMNEF[  "ijab"]* L_2(1)[    "kc"];
    DL_3(3)["ijkabc"] +=                 FME[    "ia"]* L_2(2)[  "jkbc"];
    DL_3(3)["ijkabc"] +=               WAMEF[  "ekbc"]* L_2(2)[  "ijae"];
    DL_3(3)["ijkabc"] -=               WMNEJ[  "ijam"]* L_2(2)[  "mkbc"];
    DL_3(3)["ijkabc"] +=               WMNEF[  "ijae"]*GAIBC_2[  "ekbc"];
    DL_3(3)["ijkabc"] -=               WMNEF[  "mkbc"]*GIJAK_2[  "ijam"];
    DL_3(3)["ijkabc"] +=                 FAE[    "ea"]* L_2(3)["ijkebc"];
    DL_3(3)["ijkabc"] -=                 FMI[    "im"]* L_2(3)["mjkabc"];
    DL_3(3)["ijkabc"] +=  (1.0/ 2.0)*  WABEF[  "efab"]* L_2(3)["ijkefc"];
    DL_3(3)["ijkabc"] +=  (1.0/ 2.0)*  WMNIJ[  "ijmn"]* L_2(3)["mnkabc"];
    DL_3(3)["ijkabc"] +=               WAMEI[  "eiam"]* L_2(3)["mjkbec"];

    L_3 = DL_3;
    L_3.weight(D);

    /***************************************************************************
     *           _
     * <0|(1+L)[[H,T^(1)],T^(2)]|0>
     *
     **************************************************************************/

    Z(2)["abij"]  =       FME_2[  "me"]*T_1(3)["eabmij"];
    Z(2)["abij"] += 0.5*WAMEF_2["bmef"]*T_1(3)["aefijm"];
    Z(2)["abij"] -= 0.5*WMNEJ_2["nmej"]*T_1(3)["abeimn"];

    U E0112 = 0.25*scalar(L(2)["mnef"]*Z(2)["efmn"]);

    /***************************************************************************
     *
     * CCSD(T-2)
     *
     **************************************************************************/

    U E101 = (1.0/36.0)*scalar(L_1(3)["mnoefg"]*DT_1(3)["efgmno"]);
    U E2 = E101;

    /***************************************************************************
     *
     * CCSD(T-3)
     *
     **************************************************************************/

    U E201 = (1.0/36.0)*scalar(L_2(3)["mnoefg"]*DT_1(3)["efgmno"]);
    U E102 = (1.0/36.0)*scalar(L_1(3)["mnoefg"]*DT_2(3)["efgmno"]);

    printf("E201: %18.15f\n", E201);
    printf("E102: %18.15f\n", E102);
    printf("\n");

    U E3 = E102;

    /***************************************************************************
     *
     * CCSD(T-4)
     *
     **************************************************************************/

    U E301 =  (1.0/36.0)*scalar(L_3(3)["mnoefg"]*DT_1(3)["efgmno"]);
    U E202 =  (1.0/ 1.0)*scalar(L_2(1)[    "me"]*DT_2(1)[    "em"])
             +(1.0/ 4.0)*scalar(L_2(2)[  "mnef"]*DT_2(2)[  "efmn"])
             +(1.0/36.0)*scalar(L_2(3)["mnoefg"]*DT_2(3)["efgmno"]);
    U E103 =  (1.0/36.0)*scalar(L_1(3)["mnoefg"]*DT_3(3)["efgmno"]);

    printf("E301: %18.15f\n", E301);
    printf("E202: %18.15f\n", E202);
    printf("E103: %18.15f\n", E103);
    printf("\n");

    printf("E0112: %18.15f\n", E0112);
    printf("\n");

    printf("E301:       %18.15f\n", E301);
    printf("E202+E0112: %18.15f\n", E202+E0112);
    printf("\n");

    printf("E202:       %18.15f\n", E202);
    printf("E103+E0112: %18.15f\n", E103+E0112);
    printf("\n");

    U E4 = E202;

    this->put("E(2)", new U(E2));
    this->put("E(3)", new U(E3));
    this->put("E(4)", new U(E4));
}

INSTANTIATE_SPECIALIZATIONS(CCSD_TQ_N);
REGISTER_TASK(CCSD_TQ_N<double>,"ccsd(tq-n)");
