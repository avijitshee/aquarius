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

    const vector<int>& N = occ.nao; // this must be reduced to the number of system orbitals..
    const vector<int>& nI = occ.nalpha;
    const vector<int>& ni = occ.nbeta;
    const vector<int>& nA = vrt.nalpha;
    const vector<int>& na = vrt.nbeta;

    Space orb(PointGroup::C1(), N, N);

  /* Read two-electron integrals from an external file and put them in a symmetry-packed array 
   */

    SymmetryBlockedTensor<U> X("X", arena, PointGroup::C1(), 4, {N,N,N,N}, {NS,NS,NS,NS}, false);

   if (arena.rank == 0){
    vector<tkv_pair<U>> ijklpairs;

    ifstream ifs(path);
    string line;

    int countline = 0 ;
    while (getline(ifs, line)){
       U val;
       int p, q, k, l;
       istringstream(line) >> p >> q >> k >> l >> val;
       countline = ((p*N[0]+k)*N[0]+q)*N[0]+l ;  //mulliken to direc ordered
       ijklpairs.emplace_back(countline,val);
    }
    X.writeRemoteData({0,0,0,0},ijklpairs);
   }
   else
   {
    X.writeRemoteData({0,0,0,0});
   } 

    auto& H = this->put("H", new TwoElectronOperator<U>("H", OneElectronOperator<U>("f", occ, vrt, Fa, Fb)));

    CTFTensor<U>& VIJKL = H.getIJKL()({0,1},{0,1})({0,0,0,0});
    CTFTensor<U>& VAIJK = H.getAIJK()({1,0},{0,1})({0,0,0,0});
    CTFTensor<U>& VABIJ = H.getABIJ()({1,0},{0,1})({0,0,0,0});
    CTFTensor<U>& VAIBJ = H.getAIBJ()({1,0},{1,0})({0,0,0,0});
    CTFTensor<U>& VABCI = H.getABCI()({1,0},{1,0})({0,0,0,0});
    CTFTensor<U>& VABCD = H.getABCD()({1,0},{1,0})({0,0,0,0});

/* transform VABCD
 */

     {
         CTFTensor<U> tmp("tmp", arena, 4, {na[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["aQRS"] = X({0,0,0,0})["PQRS"]*ca_({0,0})["Pa"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {na[0],na[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["abRS"] =   tmp["aQRS"]*ca_({0,0})["Qb"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {na[0],na[0],na[0],N[0]}, {NS,NS,NS,NS});
         tmp2["abcS"] = tmp1["abRS"]*ca_({0,0})["Rc"];
         H.getABCD()({0,0},{0,0})({0,0,0,0})["abcd"] =  0.5*tmp2["abcS"]*ca_({0,0})["Sd"];
     }


     {
         CTFTensor<U> tmp("tmp", arena, 4, {nA[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["AQRS"] = X({0,0,0,0})["PQRS"]*cA_({0,0})["PA"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {nA[0],na[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["AbRS"] =   tmp["AQRS"]*ca_({0,0})["Qb"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {nA[0],na[0],nA[0],N[0]}, {NS,NS,NS,NS});
         tmp2["AbCS"] = tmp1["AbRS"]*cA_({0,0})["RC"];
         VABCD["AbCd"] =   tmp2["AbCS"]*ca_({0,0})["Sd"];
     }

     {
         CTFTensor<U> tmp("tmp", arena, 4, {nA[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["AQRS"] = X({0,0,0,0})["PQRS"]*cA_({0,0})["PA"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {nA[0],nA[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["ABRS"] =   tmp["AQRS"]*cA_({0,0})["QB"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {nA[0],nA[0],nA[0],N[0]}, {NS,NS,NS,NS});
         tmp2["ABCS"] = tmp1["ABRS"]*cA_({0,0})["RC"];
         H.getABCD()({2,0},{2,0})({0,0,0,0})["ABCD"] =  0.5*tmp2["ABCS"]*cA_({0,0})["SD"];
     }

/* transform ABCI
 */

     {
         CTFTensor<U> tmp("tmp", arena, 4, {nA[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["AQRS"] = X({0,0,0,0})["PQRS"]*cA_({0,0})["PA"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {nA[0],na[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["AbRS"] =   tmp["AQRS"]*ca_({0,0})["Qb"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {nA[0],na[0],nA[0],N[0]}, {NS,NS,NS,NS});
         tmp2["AbCS"] = tmp1["AbRS"]*cA_({0,0})["RC"];
         VABCI["AbCi"] =   tmp2["AbCS"]*ci_({0,0})["Si"];
     }

     {
         CTFTensor<U> tmp("tmp", arena, 4, {nA[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["AQRS"] = X({0,0,0,0})["PQRS"]*cA_({0,0})["PA"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {nA[0],nA[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["ABRS"] =   tmp["AQRS"]*cA_({0,0})["QB"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {nA[0],nA[0],nA[0],N[0]}, {NS,NS,NS,NS});
         tmp2["ABCS"] = tmp1["ABRS"]*cA_({0,0})["RC"];
         H.getABCI()({2,0},{1,1})({0,0,0,0})["ABCI"] = tmp2["ABCS"]*cI_({0,0})["SI"];
     }

     {
         CTFTensor<U> tmp("tmp", arena, 4, {nA[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["AQRS"] = X({0,0,0,0})["PQRS"]*cA_({0,0})["PA"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {nA[0],na[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["AbRS"] =   tmp["AQRS"]*ca_({0,0})["Qb"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {nA[0],na[0],na[0],N[0]}, {NS,NS,NS,NS});
         tmp2["AbcS"] = tmp1["AbRS"]*ca_({0,0})["Rc"];
         H.getABCI()({1,0},{0,1})({0,0,0,0})["AbcI"] =   tmp2["AbcS"]*cI_({0,0})["SI"];
     }

     {
         CTFTensor<U> tmp("tmp", arena, 4, {na[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["aQRS"] = X({0,0,0,0})["PQRS"]*ca_({0,0})["Pa"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {na[0],na[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["abRS"] =   tmp["aQRS"]*ca_({0,0})["Qb"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {na[0],na[0],na[0],N[0]}, {NS,NS,NS,NS});
         tmp2["abcS"] = tmp1["abRS"]*ca_({0,0})["Rc"];
         H.getABCI()({0,0},{0,0})({0,0,0,0})["abci"] =   tmp2["abcS"]*ci_({0,0})["Si"];
     }

/* ABIJ
 */

     {
         CTFTensor<U> tmp("tmp", arena, 4, {nA[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["AQRS"] = X({0,0,0,0})["PQRS"]*cA_({0,0})["PA"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {nA[0],na[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["AbRS"] =   tmp["AQRS"]*ca_({0,0})["Qb"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {nA[0],na[0],nI[0],N[0]}, {NS,NS,NS,NS});
         tmp2["AbIS"] = tmp1["AbRS"]*cI_({0,0})["RI"];
         VABIJ["AbIj"] =   tmp2["AbIS"]*ci_({0,0})["Sj"];
     }

/*testing
 */

         SymmetryBlockedTensor<U> ABIJ("ABIJ", arena, PointGroup::C1(), 4, {{nA[0]},{nA[0]},{nI[0]},{nI[0]}}, {NS,NS,NS,NS}, false);
         SymmetryBlockedTensor<U> abij("abij", arena, PointGroup::C1(), 4, {{na[0]},{na[0]},{ni[0]},{ni[0]}}, {NS,NS,NS,NS}, false);

     {
         CTFTensor<U> tmp("tmp", arena, 4, {nA[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["AQRS"] = X({0,0,0,0})["PQRS"]*cA_({0,0})["PA"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {nA[0],nA[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["ABRS"] =   tmp["AQRS"]*cA_({0,0})["QB"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {nA[0],nA[0],nI[0],N[0]}, {NS,NS,NS,NS});
         tmp2["ABIS"] = tmp1["ABRS"]*cI_({0,0})["RI"];
         ABIJ({0,0,0,0})["ABIJ"] =   tmp2["ABIS"]*cI_({0,0})["SJ"];
     }

     {
         CTFTensor<U> tmp("tmp", arena, 4, {na[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["aQRS"] = X({0,0,0,0})["PQRS"]*ca_({0,0})["Pa"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {na[0],na[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["abRS"] =   tmp["aQRS"]*ca_({0,0})["Qb"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {na[0],na[0],ni[0],N[0]}, {NS,NS,NS,NS});
         tmp2["abiS"] = tmp1["abRS"]*ci_({0,0})["Ri"];
         abij({0,0,0,0})["abij"] =   tmp2["abiS"]*ci_({0,0})["Sj"];
     }

/* AIBJ
 */
     {
         CTFTensor<U> tmp("tmp", arena, 4, {nA[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["AQRS"] = X({0,0,0,0})["PQRS"]*cA_({0,0})["PA"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {nA[0],ni[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["AiRS"] =   tmp["AQRS"]*ci_({0,0})["Qi"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {nA[0],ni[0],nA[0],N[0]}, {NS,NS,NS,NS});
         tmp2["AiBS"] = tmp1["AiRS"]*cA_({0,0})["RB"];
         VAIBJ["AiBj"] = tmp2["AiBS"]*ci_({0,0})["Sj"];
     }

     {
         CTFTensor<U> tmp("tmp", arena, 4, {nA[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["AQRS"] = X({0,0,0,0})["PQRS"]*cA_({0,0})["PA"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {nA[0],nI[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["AIRS"] =   tmp["AQRS"]*cI_({0,0})["QI"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {nA[0],nI[0],nA[0],N[0]}, {NS,NS,NS,NS});
         tmp2["AIBS"] = tmp1["AIRS"]*cA_({0,0})["RB"];
         H.getAIBJ()({1,1},{1,1})({0,0,0,0})["AIBJ"] = tmp2["AIBS"]*cI_({0,0})["SJ"];
     }

     {
         CTFTensor<U> tmp("tmp", arena, 4, {na[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["aQRS"] = X({0,0,0,0})["PQRS"]*ca_({0,0})["Pa"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {na[0],ni[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["aiRS"] =   tmp["aQRS"]*ci_({0,0})["Qi"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {na[0],ni[0],na[0],N[0]}, {NS,NS,NS,NS});
         tmp2["aibS"] = tmp1["aiRS"]*ca_({0,0})["Rb"];
         H.getAIBJ()({0,0},{0,0})({0,0,0,0})["aibj"] = tmp2["aibS"]*ci_({0,0})["Sj"];
     }


     {
         CTFTensor<U> tmp("tmp", arena, 4, {na[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["aQRS"] = X({0,0,0,0})["PQRS"]*ca_({0,0})["Pa"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {na[0],nI[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["aIRS"] =   tmp["aQRS"]*cI_({0,0})["QI"];
         CTFTensor<U> tmp2("tmp2", arena, 4, {na[0],nI[0],na[0],N[0]}, {NS,NS,NS,NS});
         tmp2["aIbS"] = tmp1["aIRS"]*ca_({0,0})["Rb"];
         H.getAIBJ()({0,1},{0,1})({0,0,0,0})["aIbJ"] = tmp2["aIbS"]*cI_({0,0})["SJ"];
     }

/* AIJK
 */

     {
        CTFTensor<U> tmp("tmp", arena, 4, {nA[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
        tmp["AQRS"] = X({0,0,0,0})["PQRS"]*cA_({0,0})["PA"];
 
        CTFTensor<U> tmp1("tmp1", arena, 4, {nA[0],ni[0],N[0],N[0]}, {NS,NS,NS,NS});
        tmp1["AiRS"] = tmp["AQRS"]*ci_({0,0})["Qi"];

        CTFTensor<U> tmp2("tmp2", arena, 4, {nA[0],ni[0],nI[0],N[0]}, {NS,NS,NS,NS});
        tmp2["AiJS"] = tmp1["AiRS"]*cI_({0,0})["RJ"];
        VAIJK["AiJk"] =   tmp2["AiJS"]*ci_({0,0})["Sk"];
     }

     {
        CTFTensor<U> tmp("tmp", arena, 4, {nA[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
        tmp["AQRS"] = X({0,0,0,0})["PQRS"]*cA_({0,0})["PA"];
        CTFTensor<U> tmp1("tmp1", arena, 4, {nA[0],nI[0],N[0],N[0]}, {NS,NS,NS,NS});
        tmp1["AIRS"] = tmp["AQRS"]*cI_({0,0})["QI"];
 
        CTFTensor<U> tmp2("tmp2", arena, 4, {nA[0],nI[0],nI[0],N[0]}, {NS,NS,NS,NS});
        tmp2["AIJS"] = tmp1["AIRS"]*cI_({0,0})["RJ"];
        H.getAIJK()({1,1},{0,2})({0,0,0,0})["AIJK"] =   tmp2["AIJS"]*cI_({0,0})["SK"];
     }

     {
        CTFTensor<U> tmp("tmp", arena, 4, {na[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
        tmp["aQRS"] = X({0,0,0,0})["PQRS"]*ca_({0,0})["Pa"];
        CTFTensor<U> tmp1("tmp1", arena, 4, {na[0],ni[0],N[0],N[0]}, {NS,NS,NS,NS});
        tmp1["aiRS"] = tmp["aQRS"]*ci_({0,0})["Qi"];
 
        CTFTensor<U> tmp2("tmp2", arena, 4, {na[0],ni[0],ni[0],N[0]}, {NS,NS,NS,NS});
        tmp2["aijS"] = tmp1["aiRS"]*ci_({0,0})["Rj"];
        H.getAIJK()({0,0},{0,0})({0,0,0,0})["aijk"] =   tmp2["aijS"]*ci_({0,0})["Sk"];
     }


     {
        CTFTensor<U> tmp("tmp", arena, 4, {na[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
        tmp["aQRS"] = X({0,0,0,0})["PQRS"]*ca_({0,0})["Pa"];
        CTFTensor<U> tmp1("tmp1", arena, 4, {na[0],nI[0],N[0],N[0]}, {NS,NS,NS,NS});
        tmp1["aIRS"] = tmp["aQRS"]*cI_({0,0})["QI"];
 
        CTFTensor<U> tmp2("tmp2", arena, 4, {na[0],nI[0],nI[0],N[0]}, {NS,NS,NS,NS});
        tmp2["aIJS"] = tmp1["aIRS"]*cI_({0,0})["RJ"];
        H.getAIJK()({0,1},{0,1})({0,0,0,0})["aIJk"] =   tmp2["aIJS"]*ci_({0,0})["Sk"];
     }

/* IJKL
 */

     {
         CTFTensor<U> tmp("tmp", arena, 4, {nI[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["IQRS"] = X({0,0,0,0})["PQRS"]*cI_({0,0})["PI"];

         CTFTensor<U> tmp1("tmp1", arena, 4, {nI[0],ni[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["IjRS"] =   tmp["IQRS"]*ci_({0,0})["Qj"];

         CTFTensor<U> tmp2("tmp2", arena, 4, {nI[0],ni[0],nI[0],N[0]}, {NS,NS,NS,NS});
         tmp2["IjKS"] = tmp1["IjRS"]*cI_({0,0})["RK"];
         VIJKL["IjKl"] =   tmp2["IjKS"]*ci_({0,0})["Sl"];
     }

     {
         CTFTensor<U> tmp("tmp", arena, 4, {nI[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["IQRS"] = X({0,0,0,0})["PQRS"]*cI_({0,0})["PI"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {nI[0],nI[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["IJRS"] =   tmp["IQRS"]*cI_({0,0})["QJ"];

         CTFTensor<U> tmp2("tmp2", arena, 4, {nI[0],nI[0],nI[0],N[0]}, {NS,NS,NS,NS});
         tmp2["IJKS"] = tmp1["IJRS"]*cI_({0,0})["RK"];
         H.getIJKL()({0,2},{0,2})({0,0,0,0})["IJKL"] =   0.5*tmp2["IJKS"]*cI_({0,0})["SL"];
     }

     {
         CTFTensor<U> tmp("tmp", arena, 4, {ni[0],N[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp["iQRS"] = X({0,0,0,0})["PQRS"]*ci_({0,0})["Pi"];
         CTFTensor<U> tmp1("tmp1", arena, 4, {ni[0],ni[0],N[0],N[0]}, {NS,NS,NS,NS});
         tmp1["ijRS"] =   tmp["iQRS"]*ci_({0,0})["Qj"];

         CTFTensor<U> tmp2("tmp2", arena, 4, {ni[0],ni[0],ni[0],N[0]}, {NS,NS,NS,NS});
         tmp2["ijkS"] = tmp1["ijRS"]*ci_({0,0})["Rk"];
         H.getIJKL()({0,0},{0,0})({0,0,0,0})["ijkl"] =   0.5*tmp2["ijkS"]*ci_({0,0})["Sl"];
     }

        H.getAIJK()({0,1},{0,1})["aIjK"]  =    -H.getAIJK()({0,1},{0,1})["aIKj"];
        H.getABCI()({1,0},{0,1})["bAcI"]  =    -H.getABCI()({1,0},{0,1})["AbcI"];

        H.getAIBJ()({1,1},{1,1})["AIBJ"] -= ABIJ["ABJI"];
        H.getAIBJ()({0,0},{0,0})["aibj"] -= abij["abji"];

        H.getABIJ()({2,0},{0,2})["ABIJ"]  = 0.5*ABIJ["ABIJ"];
        H.getABIJ()({0,0},{0,0})["abij"]  = 0.5*abij["abij"];

        H.getAIBJ()({1,0},{0,1})["AibJ"] = -H.getABIJ()({1,0},{0,1})["AbJi"];
        H.getAIBJ()({0,1},{1,0})["aIBj"] = -H.getABIJ()({1,0},{0,1})["BaIj"];

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

    this->log(arena) << "ABCD: " << setprecision(15) << H.getABCD()({2,0},{2,0}).norm(2) << endl;
    this->log(arena) << "AbCd: " << setprecision(15) << H.getABCD()({1,0},{1,0}).norm(2) << endl;
    this->log(arena) << "abcd: " << setprecision(15) << H.getABCD()({0,0},{0,0}).norm(2) << endl;
    this->log(arena) << "ABCI: " << setprecision(15) << H.getABCI()({2,0},{1,1}).norm(2) << endl;
    this->log(arena) << "AbCi: " << setprecision(15) << H.getABCI()({1,0},{1,0}).norm(2) << endl;
    this->log(arena) << "AbcI: " << setprecision(15) << H.getABCI()({1,0},{0,1}).norm(2) << endl;
    this->log(arena) << "abci: " << setprecision(15) << H.getABCI()({0,0},{0,0}).norm(2) << endl;
    this->log(arena) << "AIBC: " << setprecision(15) << H.getAIBC()({1,1},{2,0}).norm(2) << endl;
    this->log(arena) << "AiBc: " << setprecision(15) << H.getAIBC()({1,0},{1,0}).norm(2) << endl;
    this->log(arena) << "aIBc: " << setprecision(15) << H.getAIBC()({0,1},{1,0}).norm(2) << endl;
    this->log(arena) << "aibc: " << setprecision(15) << H.getAIBC()({0,0},{0,0}).norm(2) << endl;
    this->log(arena) << "ABIJ: " << setprecision(15) << H.getABIJ()({2,0},{0,2}).norm(2) << endl;
    this->log(arena) << "AbIj: " << setprecision(15) << H.getABIJ()({1,0},{0,1}).norm(2) << endl;
    this->log(arena) << "abij: " << setprecision(15) << H.getABIJ()({0,0},{0,0}).norm(2) << endl;
    this->log(arena) << "AIBJ: " << setprecision(15) << H.getAIBJ()({1,1},{1,1}).norm(2) << endl;
    this->log(arena) << "AiBj: " << setprecision(15) << H.getAIBJ()({1,0},{1,0}).norm(2) << endl;
    this->log(arena) << "aIbJ: " << setprecision(15) << H.getAIBJ()({0,1},{0,1}).norm(2) << endl;
    this->log(arena) << "AibJ: " << setprecision(15) << H.getAIBJ()({1,0},{0,1}).norm(2) << endl;
    this->log(arena) << "aIBj: " << setprecision(15) << H.getAIBJ()({0,1},{1,0}).norm(2) << endl;
    this->log(arena) << "aibj: " << setprecision(15) << H.getAIBJ()({0,0},{0,0}).norm(2) << endl;
    this->log(arena) << "IJAB: " << setprecision(15) << H.getIJAB()({0,2},{2,0}).norm(2) << endl;
    this->log(arena) << "IjAb: " << setprecision(15) << H.getIJAB()({0,1},{1,0}).norm(2) << endl;
    this->log(arena) << "ijab: " << setprecision(15) << H.getIJAB()({0,0},{0,0}).norm(2) << endl;
    this->log(arena) << "AIJK: " << setprecision(15) << H.getAIJK()({1,1},{0,2}).norm(2) << endl;
    this->log(arena) << "AiJk: " << setprecision(15) << H.getAIJK()({1,0},{0,1}).norm(2) << endl;
    this->log(arena) << "aIJk: " << setprecision(15) << H.getAIJK()({0,1},{0,1}).norm(2) << endl;
    this->log(arena) << "aijk: " << setprecision(15) << H.getAIJK()({0,0},{0,0}).norm(2) << endl;
    this->log(arena) << "IJAK: " << setprecision(15) << H.getIJAK()({0,2},{1,1}).norm(2) << endl;
    this->log(arena) << "IjAk: " << setprecision(15) << H.getIJAK()({0,1},{1,0}).norm(2) << endl;
    this->log(arena) << "IjaK: " << setprecision(15) << H.getIJAK()({0,1},{0,1}).norm(2) << endl;
    this->log(arena) << "ijak: " << setprecision(15) << H.getIJAK()({0,0},{0,0}).norm(2) << endl;
    this->log(arena) << "IJKL: " << setprecision(15) << H.getIJKL()({0,2},{0,2}).norm(2) << endl;
    this->log(arena) << "IjKl: " << setprecision(15) << H.getIJKL()({0,1},{0,1}).norm(2) << endl;
    this->log(arena) << "ijkl: " << setprecision(15) << H.getIJKL()({0,0},{0,0}).norm(2) << endl;


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
