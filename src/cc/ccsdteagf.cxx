#include "util/global.hpp"

#include "convergence/complex_linear_krylov.hpp"
#include "util/iterative.hpp"
#include "operator/2eoperator.hpp"
#include "operator/st2eoperator.hpp"
#include "operator/excitationoperator.hpp"
#include "operator/denominator.hpp"

using namespace aquarius::tensor;
using namespace aquarius::task;
using namespace aquarius::input;
using namespace aquarius::op;
using namespace aquarius::convergence;
using namespace aquarius::symmetry;

namespace aquarius
{
namespace cc
{

template <typename U>
class CCSDTEAGF : public Iterative<complex_type_t<U>>
{
    protected:
        typedef complex_type_t<U> CU;

        Config krylov_config;
        int orbital;
        vector<CU> omegas;
        CU omega;
        CU value ;
        int orbstart;
        int orbend;
        int element_start ;
        int element_end ;

    public:
        CCSDTEAGF(const string& name, Config& config)
        : Iterative<CU>(name, config), krylov_config(config.get("krylov"))
        {
            vector<Requirement> reqs;
            reqs.emplace_back("ccsdt.T", "T");
            reqs.emplace_back("ccsdt.L", "L");
            reqs.emplace_back("ccsdt.Hbar", "Hbar");
            this->addProduct("ccsdt.eagf", "gf_ea", reqs);

            double from = config.get<double>("omega_min");
            double to = config.get<double>("omega_max");
            int n = config.get<double>("npoint");
            double eta = config.get<double>("eta");
            element_start = config.get<int>("element_start");
            element_end = config.get<int>("element_end");
            double beta = config.get<double>("beta");
            string grid_type = config.get<string>("grid");

            double delta = (to-from)/max(1,n-1);
            for (int i = 0;i < n;i++)
            {
             if (grid_type == "real") omegas.emplace_back(from+delta*i, eta);
             if (grid_type == "imaginary") omegas.emplace_back(0.,(2.0*i+1)*M_PI/beta);
            }


            ifstream ifsa("wlist_sub.txt");
            if (ifsa){
            omegas.clear() ; 
            string line;
            while (getline(ifsa, line)){
             U val;
             istringstream(line) >> val;
             omegas.emplace_back(0.,val);
            }
           }

        }

        bool run(TaskDAG& dag, const Arena& arena)
        {
            auto& H = this->template get<STTwoElectronOperator<U>>("Hbar");

            const PointGroup& group = H.getABIJ().getGroup();
            int nirrep = group.getNumIrreps();

            const Space& occ = H.occ;
            const Space& vrt = H.vrt;

            int nI = occ.nalpha[0];
            int ni = occ.nbeta[0];
            int nA = vrt.nalpha[0];
            int na = vrt.nbeta[0];
            int orbleft ;
            int orbright ;

//            int maxspin = (nI == ni) ? 1 : 2 ;
            int maxspin = 2 ;

            auto& T = this->template get<ExcitationOperator  <U,3>>("T");
            auto& L = this->template get<DeexcitationOperator<U,3>>("L");

//          auto& gf_ea = this-> put("gf_ea", new vector<vector<vector<CU>>>) ;
            auto& gf_ea = this-> put("gf_ea", new vector<CU>) ;

            SpinorbitalTensor<U> Dab("D(ab)", arena, group, {vrt,occ}, {1,0}, {1,0});
            SpinorbitalTensor<U> Gieab("G(am,ef)", arena, group, {vrt,occ}, {1,1}, {2,0});
            SpinorbitalTensor<U> Gamoefg("G(amo,efg)", arena, group, {vrt,occ}, {1,2}, {3,0});

            Dab["ab"]  =     -L(1)["mb"  ]*T(1)["am"  ];
            Dab["ab"] -= 0.5*L(2)["kmbe"]*T(2)["aekm"];
            Dab["ab"] -= (1.0/12.0)*L(3)["mnobcd"]*T(3)["acdmno"];

            Gieab["amef"]  = -L(2)["nmef"]*T(1)[  "an"];
            Gieab["amef"] -=  L(3)["nmoefg"]*T(2)[  "agno"];
            Gamoefg["amoefg"] -=  L(3)["nmoefg"]*T(1)[  "an"];
//         {
//           gf_ea.resize(maxspin);

//          for (int nspin = 0;nspin < maxspin;nspin++){
//            gf_ea[nspin].resize(omegas.size());
//          }  

//          for (int nspin = 0;nspin < maxspin;nspin++){
//            for (int i = 0;i < omegas.size();i++){
//              gf_ea[nspin][i].resize(element_end-element_start+1);
//            }
//          }
//         } 

           vector<int> array1(pow((nI+nA),2));
           vector<int> array2(pow((nI+nA),2));
           vector< pair <int,int> > get_index ; 

           int x = 0 ; 
           for (int orbleft = 0; orbleft < (nI+nA) ; orbleft++){   
             for (int orbright = 0; orbright < (nI+nA) ; orbright++){   
                array1[x] = orbleft ;
                array2[x] = orbright ;
                x += 1 ;
             }
           }

           for (int i = 0; i < (nI+nA)*(nI+nA) ; i++){   
              get_index.push_back( make_pair(array1[i],array2[i]) );
           }

    /* start calculating all GF elements..
     */  

       int uppertriangle ;

       for (int nspin = 0; nspin < maxspin ; nspin++){
           uppertriangle = 0 ;
        for (int orbs = element_start; orbs < element_end ; orbs++){   

           orbleft = get_index[orbs].first ;
           orbright = get_index[orbs].second ;

           this->log(arena) << "Computing Green's function element: " << orbleft << " "<< orbright << endl ;

            bool isalpha_right = false;
            bool isvrt_right = true;
            bool isalpha_left = false;
            bool isvrt_left = true;

            int orbleft_dummy = orbleft ;
            int orbright_dummy = orbright ;


            if (nspin == 0)
            {
                isalpha_left = true;
                if ((orbleft ) >= nI)
                {
                    isvrt_left = false;
                    orbleft_dummy = orbleft - nI;
                }
            }
            else
            {
                if (orbleft >= ni)
                {
                    isvrt_left = false;
                    orbleft_dummy = orbleft - ni;
                }
            }

            if (nspin == 0)
            {
                isalpha_right = true;
                if ((orbright ) >= nI)
                {
                    isvrt_right = false;
                    orbright_dummy = orbright - nI;
                }
            }
            else
            {
                if (orbright >= ni)
                {
                    isvrt_right = false;
                    orbright_dummy = orbright - ni;
                }
            }


            auto& Rr = this->puttmp("Rr", new ExcitationOperator  <U,3,2>("Rr", arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& Ri = this->puttmp("Ri", new ExcitationOperator  <U,3,2>("Ri", arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& Zr = this->puttmp("Zr", new ExcitationOperator  <U,3,2>("Zr", arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& Zi = this->puttmp("Zi", new ExcitationOperator  <U,3,2>("Zi", arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& b  = this->puttmp("b",  new ExcitationOperator  <U,3,2>("b",  arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& e  = this->puttmp("e",  new DeexcitationOperator<U,3,2>("e",  arena, occ, vrt, isalpha_left ? -1 : 1));


            auto& XMI = this->puttmp("XMI", new SpinorbitalTensor<U>("X(mi)", arena, group, {vrt,occ}, {0,1}, {0,0}, isalpha_left ? 1 : -1));


            auto& XMCI = this->puttmp("XMCI", new SpinorbitalTensor<U>("X(mc,i)", arena, group, {vrt,occ}, {1,1}, {0,1}, isalpha_right ? -1 : 1));
            auto& XACE = this->puttmp("XACE", new SpinorbitalTensor<U>("X(ac,e)", arena, group, {vrt,occ}, {2,0}, {1,0}, isalpha_right ? -1 : 1));


            auto& XANE = this->puttmp("XANE", new SpinorbitalTensor<U>("X(an,e)", arena, group, {vrt,occ}, {1,1}, {1,0}, isalpha_right ? -1 : 1));
            auto& XMNI = this->puttmp("XMNI", new SpinorbitalTensor<U>("X(mn,i)", arena, group, {vrt,occ}, {0,2}, {0,1}, isalpha_right ? -1 : 1));


            SpinorbitalTensor<U> ap ("ap"  , arena, group, {vrt,occ}, {!isvrt_right, isvrt_right}, {0,0}, isalpha_right ? 1 : -1);
            SpinorbitalTensor<U> apt("ap^t", arena, group, {vrt,occ}, {0,0}, {!isvrt_left, isvrt_left}, isalpha_left ? -1 : 1);

            vector<tkv_pair<U>> pair_left{{orbleft_dummy, 1}};
            vector<tkv_pair<U>> pair_right{{orbright_dummy, 1}};

            CTFTensor<U>& tensor1 = ap({!isvrt_right && isalpha_right, isvrt_right && isalpha_right}, {0,0})({0});
            if (arena.rank == 0)
                tensor1.writeRemoteData(pair_right);
            else
                tensor1.writeRemoteData();

            CTFTensor<U>& tensor2 = apt({0,0}, {!isvrt_left && isalpha_left, isvrt_left && isalpha_left})({0});
            if (arena.rank == 0)
                tensor2.writeRemoteData(pair_left);
            else
                tensor2.writeRemoteData();


            if ((isvrt_right) && (isvrt_left))
            {

                /*
                 *  ab...    abe...
                 * b  (e) = t
                 *  ijk...   ijk...
                 */
                b(1)[  "a"] = -T(1)[  "ak"]*ap["k"];
                b(2)["abi"] = -T(2)["abik"]*ap["k"];
                b(3)["abcij"] = -T(3)["abcijk"]*ap["k"];

                /*
                 *  ijk...   ijk...
                 * e  (e) = l
                 *  ab...    abe...
                 */
                e(1)[  "a"]  = -L(1)[  "ka"]*apt["k"];
                e(2)["iab"]  = -L(2)["ikab"]*apt["k"];
                e(3)["ijabc"]  = -L(3)["ijkabc"]*apt["k"];
            }
            else if((isvrt_right) && (!isvrt_left))
            {
                b(1)[  "a"] = -T(1)[  "ak"]*ap["k"];
                b(2)["abi"] = -T(2)["abik"]*ap["k"];
                b(3)["abcij"] = -T(3)["abcijk"]*ap["k"];

                e(1)[  "a"]  =               apt["a"];
                e(1)[  "a"]  +=   Dab[  "ea"]*apt["e"];
                e(2)["iab"]  =  L(1)[  "ia"]*apt["b"];
                e(2)["iab"]  += Gieab["eiba"]*apt["e"];
                e(3)["ijabc"]  =  L(2)[  "ijab"]*apt["c"];
                e(3)["ijabc"]  +=  Gamoefg["djicba"]*apt["d"];
            }
            else if((!isvrt_right) && (isvrt_left))
            {
                b(1)["a"] = ap["a"];

                e(1)[  "a"]  = -L(1)[  "ka"]*apt["k"];
                e(2)["iab"]  = -L(2)["ikab"]*apt["k"];
                e(3)["ijabc"]  = -L(3)["ijkabc"]*apt["k"];
            } 
            else
            {
                b(1)["a"] = ap["a"];

                e(1)[  "a"]  =               apt["a"];
                e(1)[  "a"]  +=   Dab[  "ea"]*apt["e"];
                e(2)["iab"]  =  L(1)[  "ia"]*apt["b"];
                e(2)["iab"]  += Gieab["eiba"]*apt["e"];
                e(3)["ijabc"]  =  L(2)[  "ijab"]*apt["c"];

                e(3)["ijabc"]  +=  Gamoefg["djicba"]*apt["d"];
            }


            auto& D = this->puttmp("D", new ComplexDenominator<U>(H));

            int omega_counter = 0 ;
            for (auto& o : omegas)
            {
                this->puttmp("krylov", new ComplexLinearKrylov<ExcitationOperator<U,3,2>>(krylov_config, b));
                omega.real( o.real());
                omega.imag( o.imag());

                this->log(arena) << "Computing Green's function at " << fixed << setprecision(6) << o << endl;

                Rr = b;
                Ri = 0;
                D.weight(Rr, Ri, omega);
                U norm = sqrt(aquarius::abs(scalar(Rr*Rr)) +
                              aquarius::abs(scalar(Ri*Ri)));
                Rr /= norm;
                Ri /= norm;

                Iterative<CU>::run(dag, arena);
//                gf_ea[nspin][omega_counter][uppertriangle] = value ;
                gf_ea.emplace_back(value) ;
                omega_counter += 1 ;
                uppertriangle +=1 ;
            }
           }
          }

         if (arena.rank == 0)
         {
             int counter = 0 ;
             for (int nspin = 0 ; nspin < maxspin ; nspin++){
             stringstream stream1;
             stream1 << "gf_ea_"<<nspin<<"_"<<element_start<<"_"<<element_end<< ".txt";
             string fileName1 = stream1.str();
             std::ofstream gffile;
             gffile.open (fileName1, ofstream::out);


            for (int j=element_start ; j < element_end ; j++){
             for (int i=0 ; i < omegas.size() ; i++){
//                 gffile << nspin << " " << i << " " << j << " " << setprecision(12) <<  gf_ea[nspin][i][j-element_start] << endl ;
                   gffile << nspin << " " << j << " " << i << " " << setprecision(12) <<  gf_ea[counter].real() << " " << gf_ea[counter].imag() << endl ;
                   counter += 1 ;
             }}  

             gffile.close();
             }
         }

            return true;
        }

        void iterate(const Arena& arena)
        {
            const auto& H = this->template get<STTwoElectronOperator<U>>("Hbar");

            const SpinorbitalTensor<U>&   FME =   H.getIA();
            const SpinorbitalTensor<U>&   FAE =   H.getAB();
            const SpinorbitalTensor<U>&   FMI =   H.getIJ();
            const SpinorbitalTensor<U>& WMNEF = H.getIJAB();
            const SpinorbitalTensor<U>& WAMEF = H.getAIBC();
            const SpinorbitalTensor<U>& WABEJ = H.getABCI();
            const SpinorbitalTensor<U>& WABEF = H.getABCD();
            const SpinorbitalTensor<U>& WMNIJ = H.getIJKL();
            const SpinorbitalTensor<U>& WMNEJ = H.getIJAK();
            const SpinorbitalTensor<U>& WAMIJ = H.getAIJK();
            const SpinorbitalTensor<U>& WAMEI = H.getAIBJ();

            auto& T = this->template get<ExcitationOperator<U,3>>("T");

            auto& XMCI = this->template gettmp<SpinorbitalTensor<U>>("XMCI");
            auto& XACE = this->template gettmp<SpinorbitalTensor<U>>("XACE");
            auto& XANE = this->template gettmp<SpinorbitalTensor<U>>("XANE");
            auto& XMNI = this->template gettmp<SpinorbitalTensor<U>>("XMNI");
            auto& XMI = this->template gettmp<SpinorbitalTensor<U>>("XMI");


            auto& D = this->template gettmp<ComplexDenominator<U>>("D");
            auto& krylov = this->template gettmp<ComplexLinearKrylov<ExcitationOperator<U,3,2>>>("krylov");

            auto& Rr = this->template gettmp<  ExcitationOperator<U,3,2>>("Rr");
            auto& Ri = this->template gettmp<  ExcitationOperator<U,3,2>>("Ri");
            auto& Zr = this->template gettmp<  ExcitationOperator<U,3,2>>("Zr");
            auto& Zi = this->template gettmp<  ExcitationOperator<U,3,2>>("Zi");
            auto& b  = this->template gettmp<  ExcitationOperator<U,3,2>>("b");
            auto& e  = this->template gettmp<DeexcitationOperator<U,3,2>>("e");

            //printf("<Rr|Rr>: %.15f\n", scalar(Rr*Rr));
            //printf("<Ri|Ri>: %.15f\n", scalar(Ri*Ri));
            //printf("<Rr|Ri>: %.15f\n", scalar(Rr*Ri));

            //printf("<B|Rr>: %.15f\n", scalar(b*Rr));
            //printf("<B|Ri>: %.15f\n", scalar(b*Ri));

            for (int ri: {0,1})
            {
                ExcitationOperator<U,3,2>& R = (ri == 0 ? Rr : Ri);
                ExcitationOperator<U,3,2>& Z = (ri == 0 ? Zr : Zi);

                  XMI[  "m"] = -0.5*WMNEF["mnef"]*R(2)["efn"];

                XMCI[  "mci"]  =      WAMEI["cmei"]*R(1)[     "e"];
                XMCI[  "mci"] +=      WMNEJ["nmei"]*R(2)[   "ecn"];
                XMCI[  "mci"] +=  0.5*WMNEF["mnef"]*R(3)[ "efcin"];
                XMCI[  "mci"] +=  0.5*WAMEF["cmef"]*R(2)[  "efi"];

                XACE[  "ace"]  =      WABEF["acef"]*R(1)[     "f"];
                XACE[  "ace"] +=      WAMEF["amef"]*R(2)[   "fcm"];
                XACE[  "ace"] -=  0.5*WMNEF["mnef"]*R(3)[ "afcmn"];

                XANE[  "ane"]  =     WAMEF["anef"]*R(1)[     "f"];
                XANE[  "ane"] -=     WMNEF["mnef"]*R(2)[   "afm"];

                XMNI[  "mni"]  =  0.5*WMNEF["mnef"]*R(2)[  "efi"];
                XMNI[  "mni"] +=      WMNEJ["nmei"]*R(1)[    "e"];

                Z(1)[  "a"]  =       FAE[  "ae"]*R(1)[  "e"];
                Z(1)[  "a"] -=       FME[  "me"]*R(2)["aem"];
                Z(1)[  "a"] -= 0.5*WAMEF["amef"]*R(2)["efm"];
                Z(1)[  "a"] += 0.25*WMNEF["mnef"]*R(3)[ "efamn"];

                Z(2)["abi"]   =     WABEJ["baei"]*R(1)[  "e"];
                Z(2)["abi"]  +=       FAE[  "ae"]*R(2)["ebi"];
                Z(2)["abi"]  -=       FMI[  "mi"]*R(2)["abm"];
                Z(2)["abi"]  -=       XMI[  "m"]*T(2)["abim"];
                Z(2)["abi"]  += 0.5*WABEF["abef"]*R(2)["efi"];
                Z(2)["abi"]  -=     WAMEI["amei"]*R(2)["ebm"];
                Z(2)["abi"]  +=        FME[  "me"]*R(3)[ "eabmi"];
                Z(2)["abi"] +=  0.5*WAMEF["amef"]*R(3)[ "efbim"];
                Z(2)["abi"] -=  0.5*WMNEJ["mnei"]*R(3)[ "eabmn"];


                Z(3)["abcij"]  =      WABEJ["abej"]*R(2)[   "eci"];
                Z(3)["abcij"] -=      WAMIJ["amij"]*R(2)[   "bcm"];
                Z(3)["abcij"] +=        FAE[  "ae"]*R(3)[ "ebcij"];
                Z(3)["abcij"] -=        FMI[  "mi"]*R(3)[ "abcmj"];
                Z(3)["abcij"] -=      WAMEI["amei"]*R(3)[ "ebcmj"];
                Z(3)["abcij"] +=  0.5*WABEF["abef"]*R(3)[ "efcij"];
                Z(3)["abcij"] +=  0.5*WMNIJ["mnij"]*R(3)[ "abcmn"];
                Z(3)["abcij"] -=       XMCI[ "mci"]*T(2)[  "abmj"];
                Z(3)["abcij"] -=       XACE[ "ace"]*T(2)[  "beji"];
                Z(3)["abcij"] +=        XMI[   "n"]*T(3)["abcijn"];
                Z(3)["abcij"] -=       XANE[ "ane"]*T(3)["ebcijn"];
                Z(3)["abcij"] +=  0.5*XMNI["mni"]*T(3)["abcmjn"];
            }

            //printf("<Z1|Z1>: %.15f\n", scalar(Zr(1)*Zr(1)));
            //printf("<Z2|Z2>: %.15f\n", 0.5*scalar(Zr(2)*Zr(2)));
            //printf("<Zr|Zr>: %.15f\n", scalar(Zr*Zr));
            //printf("<Zi|Zi>: %.15f\n", scalar(Zi*Zi));

            /*
             * Convert H*r to (H-w)*r
             */


              Zr *= -1;
              Zi *= -1;

            Zr += omega.real()*Rr;
            Zr -= omega.imag()*Ri;
            Zi += omega.real()*Ri;
            Zi += omega.imag()*Rr;
            //Zr *= -1;
            //Zi *= -1;

            //printf("<Ur|Ur>: %.15f\n", scalar(Zr*Zr));
            //printf("<Ui|Ui>: %.15f\n", scalar(Zi*Zi));

            krylov.extrapolate(Rr, Ri, Zr, Zi, D, omega);

            this->conv() = max(Zr.norm(00), Zi.norm(00));

            krylov.getSolution(Zr, Zi);


            this->energy() = CU(    scalar(e(1)[  "a"]*Zr(1)[  "a"]) +
                                0.5*scalar(e(2)["iab"]*Zr(2)["abi"]) +
                                (1.0/12.0)*scalar(e(3)["ijabc"]*Zr(3)["abcij"]),
                                    scalar(e(1)[  "a"]*Zi(1)[  "a"]) +
                                0.5*scalar(e(2)["iab"]*Zi(2)["abi"]) +
                                (1.0/12.0)*scalar(e(3)["ijabc"]*Zi(3)["abcij"]));

            value = this->energy() ;
     

        }
};

}
}

static const char* spec = R"(

element_start ?
   int 0,
element_end int,
npoint int,
omega_min double,
omega_max double,
eta double,
beta?
   double 100.0 , 
grid?
  enum{ real, imaginary },
convergence?
    double 1e-9,
max_iterations?
    int 150,
conv_type?
    enum { MAXE, RMSE, MAE },
krylov?
{
    order?
            int 10,
    compaction?
            enum { discrete, continuous },
}

)";

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::CCSDTEAGF);
REGISTER_TASK(aquarius::cc::CCSDTEAGF<double>, "ccsdteagf",spec);
