#include "hubbard_aomoints.hpp"

#include "time/time.hpp"

using namespace aquarius::tensor;
using namespace aquarius::input;
using namespace aquarius::integrals;
using namespace aquarius::task;
using namespace aquarius::op;
using namespace aquarius::symmetry;

namespace aquarius
{
namespace hubbard
{

template <typename U>
Hubbard_AOMOints<U>::Hubbard_AOMOints(const string& name, Config& config)
: Hubbard_MOIntegrals<U>(name, config)
{
//    this->getProduct("H").addRequirement("eri", "I"); //this line has to be modified
  int nelec = 12 ;
}

template <typename U>
bool Hubbard_AOMOints<U>::run(TaskDAG& dag, const Arena& arena)
{
    int nirreps = 1 ;
    CTF_Timer_epoch ep("Hubbard_AOMOints");
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

    vector<vector<U>> cA(nirreps), ca(nirreps), cI(nirreps), ci(nirreps);

    const vector<int>& N = occ.nao;
    const vector<int>& nI = occ.nalpha;
    const vector<int>& ni = occ.nbeta;
    const vector<int>& nA = vrt.nalpha;
    const vector<int>& na = vrt.nbeta;

    int norb = N[0]; 
    int nocc = nI[0]; 

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
    }

    vector<vector<U>> E(norb,vector<U>(norb));

        int nvrt = norb-nocc;

   /*In the following we define the one-electronic AO/site integrals for Hubbard model..
    */ 

    Space orb(PointGroup::C1(), N, N);

    auto& H = this->put("H", new TwoElectronOperator<U>("V", OneElectronOperator<U>("f", occ, vrt, Fa, Fb)));

    vector<tkv_pair<U>> ijklpairs;

//  int k ; 
//  for (int i = 0;i < norb;i++)  
//  {
//      k = i*norb+i ;
//      ijklpairs.emplace_back(k*norb*norb+k, v_onsite[i]);
//  }

    for (int i = 0; i < nocc ; i++)
    {
    for (int j = 0; j < 1 ; j++)
    {
     printf("Print mo, ao and coefficient: %d %d %.15f\n ",i,j,ci[0][i*norb+j]);
    }
    }

    /*
     * <ab||ij>
     */
    writeIntegrals(true, true, false, false,cA[0],ca[0],cI[0],ci[0],H.getABIJ()({1,0},{0,1}));
//    writeIntegrals(true, true, false, false,cAtrans,catrans,cItrans,citrans,H.getABIJ()({1,0},{0,1}));
    H.getABIJ()({2,0},{0,2})["ABIJ"]  = 0.5*H.getABIJ()({1,0},{0,1})["ABIJ"];
    H.getABIJ()({0,0},{0,0})["abij"]  = 0.5*H.getABIJ()({1,0},{0,1})["abij"];
    /*
     * <ab||ci>
     */
    writeIntegrals(true, true, true, false,cA[0],ca[0],cA[0],ci[0],H.getABCI()({1,0},{1,0}));
 //   writeIntegrals(true, true, true, false,cAtrans,catrans,cAtrans,citrans,H.getABCI()({1,0},{1,0}));
    H.getABCI()({2,0},{1,1})["ABCI"]  =     H.getABCI()({1,0},{1,0})["ABCI"];
    H.getABCI()({1,0},{0,1})["AbcI"]  =    -H.getABCI()({1,0},{1,0})["bAcI"];
    H.getABCI()({0,0},{0,0})["abci"]  =     H.getABCI()({1,0},{1,0})["abci"];

    /*
     * <ai||jk>
     */
    writeIntegrals(true, false, false, false,cA[0],ci[0],cI[0],ci[0],H.getAIJK()({1,0},{0,1}));
 //   writeIntegrals(true, false, false, false,cAtrans,citrans,cItrans,citrans,H.getAIJK()({1,0},{0,1}));
    SymmetryBlockedTensor<U> tmp(H.getAIJK()({1,0},{0,1}));
    H.getAIJK()({1,1},{0,2})["AIJK"]  =     H.getAIJK()({1,0},{0,1})["AIJK"];
    H.getAIJK()({0,1},{0,1})["aIJk"]  =    -H.getAIJK()({1,0},{0,1})["aIkJ"];
    H.getAIJK()({0,0},{0,0})["aijk"]  =     H.getAIJK()({1,0},{0,1})["aijk"];

    /*
     * <ij||kl>
     */
    writeIntegrals(false, false, false, false,cI[0],ci[0],cI[0],ci[0],H.getIJKL()({0,1},{0,1}));
//    writeIntegrals(false, false, false, false,cItrans,citrans,cItrans,citrans,H.getIJKL()({0,1},{0,1}));
    H.getIJKL()({0,2},{0,2})["IJKL"]  = 0.25*H.getIJKL()({0,1},{0,1})["IJKL"];
    H.getIJKL()({0,0},{0,0})["ijkl"]  = 0.25*H.getIJKL()({0,1},{0,1})["ijkl"];

    /*
     * <ab||cd>
     */
    writeIntegrals(true, true, true, true,cA[0],ca[0],cA[0],ca[0],H.getABCD()({1,0},{1,0}));
 // writeIntegrals(true, true, true, true,cAtrans,catrans,cAtrans,catrans,H.getABCD()({1,0},{1,0}));
    H.getABCD()({2,0},{2,0})["ABCD"]  = 0.25*H.getABCD()({1,0},{1,0})["ABCD"];
    H.getABCD()({0,0},{0,0})["abcd"]  = 0.25*H.getABCD()({1,0},{1,0})["abcd"];

    /*
     * <ai||bj>
     */
    writeIntegrals(true, false, true, false,cA[0],ci[0],cA[0],ci[0],H.getAIBJ()({1,0},{1,0}));

    H.getAIBJ()({1,1},{1,1})["AIBJ"]  =     H.getAIBJ()({1,0},{1,0})["AIBJ"];
    H.getAIBJ()({1,1},{1,1})["AIBJ"] -=     H.getABIJ()({1,0},{0,1})["ABJI"];
    H.getAIBJ()({0,1},{0,1})["aIbJ"]  =     H.getAIBJ()({1,0},{1,0})["aIbJ"];
    H.getAIBJ()({1,0},{0,1})["AibJ"]  =    -H.getABIJ()({1,0},{0,1})["AbJi"];
    H.getAIBJ()({0,1},{1,0})["aIBj"]  =    -H.getABIJ()({1,0},{0,1})["aBjI"];
    H.getAIBJ()({0,0},{0,0})["aibj"]  =     H.getAIBJ()({1,0},{1,0})["aibj"];
    H.getAIBJ()({0,0},{0,0})["aibj"] -=     H.getABIJ()({1,0},{0,1})["abji"];

    /*
     * Fill in pieces which are equal by Hermiticity
     */

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

    printf("print scalar prod IJKL: %.15f\n",real(scalar(H.getIJKL()*H.getIJKL()))) ;
    printf("print scalar prod ABCD: %.15f\n",real(scalar(H.getABCD()*H.getABCD()))) ;
    printf("print scalar prod AIBJ: %.15f\n",real(scalar(H.getAIBJ()*H.getAIBJ()))) ;
    printf("print scalar prod IJAK: %.15f\n",real(scalar(H.getIJAK()*H.getIJAK()))) ;
    printf("print scalar prod ABCI: %.15f\n",real(scalar(H.getABCI()*H.getABCI()))) ;
    printf("print scalar prod IJAB: %.15f\n",real(scalar(H.getABIJ()*H.getABIJ()))) ;

    return true;
}

template <typename U>
void Hubbard_AOMOints<U>::writeIntegrals(bool pvirt, bool qvirt, bool rvirt, bool svirt,
                                vector<U>& cfirst, vector<U>& csecond, vector<U>& cthird, vector<U>& cfourth,
                                SymmetryBlockedTensor<U>& tensor)
{
    vector<tkv_pair<U>> pairs;
    tensor.getLocalData({0,0,0,0}, pairs);

    int norb = 12;
    int nocc = 6 ;

    int np = (pvirt ? norb-nocc : nocc);
    int nq = (qvirt ? norb-nocc : nocc);
    int nr = (rvirt ? norb-nocc : nocc);
    int ns = (svirt ? norb-nocc : nocc);

   /* H(Pjkl) = V(ijkl)*C(Pi)
    */

//   vector<U> buf1(np*nq,0.0) ;
//   vector<U> buf2(nr*ns,0.0) ;
//   vector<U> buf3(np*nq*nr*ns,0.0) ;

     vector<double> v_onsite = {4.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000};   

//  for (int i = 0; i < norb ; i++)
//  {
//     double value = v_onsite[i] ;
//     scal(np, value, &cfirst[i*np], 1);
//     gemm('T','N',np, nq, 1, 1.0, &cfirst[i*np], np, &csecond[i*nq], nq, 0.0, &buf1[0], np);
//     gemm('T','N',nr, ns, 1, 1.0, &cthird[i*nr], nr, &cfourth[i*ns], ns, 0.0, &buf2[0], nr);
//     gemm('N','T',np*nq, nr*ns, 1, 1.0, &buf1[0],np*nq, &buf2[0], nr*ns, 1.0, &buf3[0], np*nq);
//  }   

    int k ;
    k = 0 ;
    double value ;

   for (int s = 0; s < ns ; s++)
   {
   for (int r = 0; r < nr ; r++)
   {
   for (int q = 0; q < nq ; q++)
   {
   for (int p = 0; p < np ; p++)
   {
        value = 0.0 ; 
        for (int i = 0; i < norb ; i++)
        {
         value += v_onsite[i]*cfirst[p*norb+i]*csecond[q*norb+i]*cthird[r*norb+i]*cfourth[s*norb+i] ; 
        }   
        pairs.emplace_back(k,value) ;
        k += 1 ;
   }   
   }   
   }   
   }   

//    printf("I came here: %.15f\n", buf3[1]);

    tensor.writeRemoteData({0,0,0,0}, pairs);
}

}
}

INSTANTIATE_SPECIALIZATIONS(aquarius::hubbard::Hubbard_AOMOints);
REGISTER_TASK(aquarius::hubbard::Hubbard_AOMOints<double>,"hubbard_aomoints");
