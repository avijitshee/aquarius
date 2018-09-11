#include "aim_aomoints.hpp"

#include "time/time.hpp"

using namespace aquarius::tensor;
using namespace aquarius::input;
using namespace aquarius::integrals;
using namespace aquarius::task;
using namespace aquarius::op;
using namespace aquarius::symmetry;

namespace aquarius
{
namespace aim
{

template <typename U>
AIM_AOMOints<U>::AIM_AOMOints(const string& name, Config& config)
: AIM_MOIntegrals<U>(name, config), path(config.get<string>("filename"))
{
   int nirreps = 1 ;
}

template <typename U>
bool AIM_AOMOints<U>::run(TaskDAG& dag, const Arena& arena)
{
    int nirreps = 1 ;
    CTF_Timer_epoch ep("AIM_AOMOints");
    ep.begin();
    const auto& occ = this->template get<MOSpace<U>>("occ");
    const auto& vrt = this->template get<MOSpace<U>>("vrt");

    const SymmetryBlockedTensor<U>& cA_ = vrt.Calpha;
    const SymmetryBlockedTensor<U>& ca_ = vrt.Cbeta;
    const SymmetryBlockedTensor<U>& cI_ = occ.Calpha;
    const SymmetryBlockedTensor<U>& ci_ = occ.Cbeta;

    auto& Ea = this->template get<vector<vector<real_type_t<U>>>>("Ea");
    auto& Eb = this->template get<vector<vector<real_type_t<U>>>>("Eb");

    auto& Fa = this->template get<SymmetryBlockedTensor<U>>("Fa");
    auto& Fb = this->template get<SymmetryBlockedTensor<U>>("Fb");

    vector<vector<U>> cA(nirreps), ca(nirreps), cI(nirreps), ci(nirreps), FA(nirreps), FB(nirreps);

    const vector<int>& N = occ.nao;
    const vector<int>& nI = occ.nalpha;
    const vector<int>& ni = occ.nbeta;
    const vector<int>& nA = vrt.nalpha;
    const vector<int>& na = vrt.nbeta;

    norb = N[0]; 

    for (int i = 0;i < nirreps;i++)
    {
        vector<int> irreps = {i,i};
        cA_.getAllData(irreps, cA[i]);
        assert(cA[i].size() == N[i]*nA[i]);
        ca_.getAllData(irreps, ca[i]);
        assert(ca[i].size() == N[i]*na[i]);
        cI_.getAllData(irreps, cI[i]);
        assert(cI[i].size() == N[i]*nI[i]);
        ci_.getAllData(irreps, ci[i]);
        assert(ci[i].size() == N[i]*ni[i]);
        Fa.getAllData(irreps, FA[i]);
        Fb.getAllData(irreps, FB[i]);
    }

    vector<vector<U>> E(norb,vector<U>(norb));

   /*In the following we define the one-electronic AO/site integrals for aim model..
    */ 

    Space orb(PointGroup::C1(), N, N);


  /* Read two-electron integrals from an external file and put them in a symmetry-packed array 
   */

    SymmetryBlockedTensor<U> X("X", arena, PointGroup::C1(), 4, {N,N,N,N}, {NS,NS,NS,NS}, false);

    vector<tkv_pair<U>> ijklpairs;

    ifstream ifs(path);
    string line;

    int countline = 0 ;
    while (getline(ifs, line))
    {
       double val;
       int p, q, k, l;
       istringstream(line) >> p >> q >> k >> l >> val;

       ijklpairs.emplace_back(countline,val);
       countline++ ;
    }

    X.writeRemoteData({0,0,0,0},ijklpairs);

    auto& H = this->put("H", new TwoElectronOperator<U>("H", OneElectronOperator<U>("f", occ, vrt, Fa, Fb)));

    CTFTensor<U>& VIJKL = H.getIJKL()({0,1},{0,1})({0,0,0,0});
    CTFTensor<U>& VAIJK = H.getAIJK()({1,0},{0,1})({0,0,0,0});
    CTFTensor<U>& VABIJ = H.getABIJ()({1,0},{0,1})({0,0,0,0});
    CTFTensor<U>& VAIBJ = H.getAIBJ()({1,0},{1,0})({0,0,0,0});
    CTFTensor<U>& VABCI = H.getABCI()({1,0},{1,0})({0,0,0,0});
    CTFTensor<U>& VABCD = H.getABCD()({1,0},{1,0})({0,0,0,0});

     {
         CTFTensor<U> tmp("tmp", arena, 4, {nA[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["AQRS"] = X({0,0,0,0})["PQRS"]*cA_({0,0})["PA"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {nA[0],nA[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["ABRS"] =   tmp["AQRS"]*cA_({0,0})["QB"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {nA[0],nA[0],nA[0],N[0]}, {NS,NS,NS,NS});
         tmp2["ABCS"] = tmp1["ABRS"]*cA_({0,0})["RC"];
         VABCD["ABCD"] =   tmp2["ABCS"]*cA_({0,0})["SD"];
     }

     {
         CTFTensor<U> tmp("tmp", arena, 4, {nA[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
           tmp["AQRS"] = X({0,0,0,0})["PQRS"]*cA_({0,0})["PA"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {nA[0],nA[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["ABRS"] =   tmp["AQRS"]*cA_({0,0})["QB"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {nA[0],nA[0],nA[0],N[0]}, {NS,NS,NS,NS});
         tmp2["ABCS"] = tmp1["ABRS"]*cA_({0,0})["RC"];
         VABCI["ABCI"] =   tmp2["ABCS"]*cI_({0,0})["SI"];
     }

     {
         CTFTensor<U> tmp("tmp", arena, 4, {nA[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
           tmp["AQRS"] = X({0,0,0,0})["PQRS"]*cA_({0,0})["PA"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {nA[0],nA[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["ABRS"] =   tmp["AQRS"]*cA_({0,0})["QB"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {nA[0],nA[0],nI[0],N[0]}, {NS,NS,NS,NS});
           tmp2["ABIS"] = tmp1["ABRS"]*cI_({0,0})["RI"];
         VABIJ["ABIJ"] =   tmp2["ABIS"]*cI_({0,0})["SJ"];
     }

     {
         CTFTensor<U> tmp("tmp", arena, 4, {nA[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
           tmp["AQRS"] = X({0,0,0,0})["PQRS"]*cA_({0,0})["PA"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {nA[0],nI[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["AIRS"] =   tmp["AQRS"]*cI_({0,0})["QI"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {nA[0],nI[0],nA[0],N[0]}, {NS,NS,NS,NS});
         tmp2["AIBS"] = tmp1["AIRS"]*cA_({0,0})["RB"];
         VAIBJ["AIBJ"] =   tmp2["AIBS"]*cI_({0,0})["SJ"];
     }

     {
         CTFTensor<U> tmp("tmp", arena, 4, {nA[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
           tmp["AQRS"] = X({0,0,0,0})["PQRS"]*cA_({0,0})["PA"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {nA[0],nI[0],N[0],N[0]}, {NS,NS,NS,NS});
           tmp1["AIRS"] = tmp["AQRS"]*cI_({0,0})["QI"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {nA[0],nI[0],nI[0],N[0]}, {NS,NS,NS,NS});
           tmp2["AIJS"] = tmp1["AIRS"]*cI_({0,0})["RJ"];
         VAIJK["AIJK"] =   tmp2["AIJS"]*cI_({0,0})["SK"];
     }

     {
         CTFTensor<U> tmp("tmp", arena, 4, {nI[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
           tmp["IQRS"] = X({0,0,0,0})["PQRS"]*cI_({0,0})["PI"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {nI[0],nI[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["IJRS"] =   tmp["IQRS"]*cI_({0,0})["QJ"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {nI[0],nI[0],nI[0],N[0]}, {NS,NS,NS,NS});
           tmp2["IJKS"] = tmp1["IJRS"]*cI_({0,0})["RK"];
         VIJKL["IJKL"] =   tmp2["IJKS"]*cI_({0,0})["SL"];
     }

    H.getABCD()({2,0},{2,0})["ABCD"]  = 0.5*H.getABCD()({1,0},{1,0})["ABCD"];
    H.getABCD()({0,0},{0,0})["abcd"]  = 0.5*H.getABCD()({1,0},{1,0})["abcd"];

    H.getABCI()({2,0},{1,1})["ABCI"]  =     H.getABCI()({1,0},{1,0})["ABCI"];
    H.getABCI()({1,0},{0,1})["AbcI"]  =    -H.getABCI()({1,0},{1,0})["bAcI"];
    H.getABCI()({0,0},{0,0})["abci"]  =     H.getABCI()({1,0},{1,0})["abci"];

    H.getABIJ()({2,0},{0,2})["ABIJ"]  = 0.5*H.getABIJ()({1,0},{0,1})["ABIJ"];
    H.getABIJ()({0,0},{0,0})["abij"]  = 0.5*H.getABIJ()({1,0},{0,1})["abij"];

    H.getAIBJ()({1,1},{1,1})["AIBJ"]  =     H.getAIBJ()({1,0},{1,0})["AIBJ"];
    H.getAIBJ()({1,1},{1,1})["AIBJ"] -=     H.getABIJ()({1,0},{0,1})["ABJI"];
    H.getAIBJ()({0,1},{0,1})["aIbJ"]  =     H.getAIBJ()({1,0},{1,0})["aIbJ"];
    H.getAIBJ()({1,0},{0,1})["AibJ"]  =    -H.getABIJ()({1,0},{0,1})["AbJi"];
    H.getAIBJ()({0,1},{1,0})["aIBj"]  =    -H.getABIJ()({1,0},{0,1})["aBjI"];
    H.getAIBJ()({0,0},{0,0})["aibj"]  =     H.getAIBJ()({1,0},{1,0})["aibj"];
    H.getAIBJ()({0,0},{0,0})["aibj"] -=     H.getABIJ()({1,0},{0,1})["abji"];

    H.getAIJK()({1,1},{0,2})["AIJK"]  =     H.getAIJK()({1,0},{0,1})["AIJK"];
    H.getAIJK()({0,1},{0,1})["aIJk"]  =    -H.getAIJK()({1,0},{0,1})["aIkJ"];
    H.getAIJK()({0,0},{0,0})["aijk"]  =     H.getAIJK()({1,0},{0,1})["aijk"];

    H.getIJKL()({0,2},{0,2})["IJKL"]  = 0.5*H.getIJKL()({0,1},{0,1})["IJKL"];
    H.getIJKL()({0,0},{0,0})["ijkl"]  = 0.5*H.getIJKL()({0,1},{0,1})["ijkl"];

    H.getIJAK()({0,2},{1,1})["JKAI"]  =     H.getAIJK()({1,1},{0,2})["AIJK"];
    H.getIJAK()({0,1},{1,0})["JkAi"]  =     H.getAIJK()({1,0},{0,1})["AiJk"];
    H.getIJAK()({0,1},{0,1})["JkaI"]  =     H.getAIJK()({0,1},{0,1})["aIJk"];
    H.getIJAK()({0,0},{0,0})["jkai"]  =     H.getAIJK()({0,0},{0,0})["aijk"];

    H.getAIBC()({1,1},{2,0})["AIBC"]  =     H.getABCI()({2,0},{1,1})["BCAI"];
    H.getAIBC()({1,0},{1,0})["AiBc"]  =     H.getABCI()({1,0},{1,0})["BcAi"];
    H.getAIBC()({0,1},{1,0})["aIBc"]  =     H.getABCI()({1,0},{0,1})["BcaI"];
    H.getAIBC()({0,0},{0,0})["aibc"]  =     H.getABCI()({0,0},{0,0})["bcai"];

    H.getIJAB()({0,2},{2,0})["IJAB"]  =     H.getABIJ()({2,0},{0,2})["ABIJ"];
    H.getIJAB()({0,1},{1,0})["IjAb"]  =     H.getABIJ()({1,0},{0,1})["AbIj"];
    H.getIJAB()({0,0},{0,0})["ijab"]  =     H.getABIJ()({0,0},{0,0})["abij"];

    return true;
}

}
}

static const char* spec = R"!(

filename?
    string v_no_sub.txt
)!";

INSTANTIATE_SPECIALIZATIONS(aquarius::aim::AIM_AOMOints);
REGISTER_TASK(CONCAT(aquarius::aim::AIM_AOMOints<double>),"aim_aomoints",spec);
