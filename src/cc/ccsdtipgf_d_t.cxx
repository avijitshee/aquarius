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
class CCSDTIPGF_D_T : public Iterative<complex_type_t<U>>
{
    protected:
        typedef complex_type_t<U> CU;

        Config krylov_config;
        vector<CU> omegas;
        CU omega;
        CU value ;
        int orbstart;
        int orbend;
        int element_start ;
        int element_end ;

    public:
        CCSDTIPGF_D_T(const string& name, Config& config)
        : Iterative<CU>(name, config), krylov_config(config.get("krylov"))
        {
            vector<Requirement> reqs;
            reqs.emplace_back("ccsd.T", "T");
            reqs.emplace_back("ccsd.L", "L");
            reqs.emplace_back("ccsd.Hbar", "Hbar");
            this->addProduct("ccsd_t.ipgf", "gf_ip", reqs);

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
             istringstream(line) >> setprecision(12) >> val;
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

//          int maxspin = (nI == ni) ? 1 : 2 ;
            int maxspin = 2 ;

            auto& T = this->template get<ExcitationOperator  <U,2>>("T");
            auto& L = this->template get<DeexcitationOperator<U,2>>("L");

            auto& gf_ip = this-> put("gf_ip", new vector<CU>) ;

            SpinorbitalTensor<U> Dij    ("D(ij)",      arena, group, {vrt,occ}, {0,1}, {0,1});
            SpinorbitalTensor<U> Gijak  ("G(ij,ak)",   arena, group, {vrt,occ}, {0,2}, {1,1});

                Dij[    "ij"]  =            L(1)[    "ie"]*T(1)[    "ej"];
                Dij[    "ij"] += (1.0/ 2.0)*L(2)[  "imef"]*T(2)[  "efjm"];

              Gijak[  "ijak"]  =            L(2)[  "ijae"]*T(1)[    "ek"];

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

       for (int nspin = 0; nspin < maxspin ; nspin++){
        for (int orbs = element_start; orbs < element_end ; orbs++){   

           orbleft = get_index[orbs].first ;
           orbright = get_index[orbs].second ;

           this->log(arena) << "Computing Green's function element: " << orbleft << " "<< orbright << endl ;


           bool isalpha_right = false;
           bool isvrt_right = false;
           bool isalpha_left = false;
           bool isvrt_left = false;

           int orbleft_dummy = orbleft ;
           int orbright_dummy = orbright ;

           if (nspin == 0)
           {
               isalpha_left = true;
               if ((orbleft ) >= nI)
               {
                   isvrt_left = true;
                   orbleft_dummy = orbleft - nI;
               }
           }
           else
           {
               if (orbleft >= ni)
               {
                   isvrt_left = true;
                   orbleft_dummy = orbleft - ni;
               }
           }

           if (nspin == 0)
           {
               isalpha_right = true;
               if ((orbright ) >= nI)
               {
                   isvrt_right = true;
                   orbright_dummy = orbright - nI;
               }
           }
           else
           {
               if (orbright >= ni)
               {
                   isvrt_right = true;
                   orbright_dummy = orbright - ni;
               }
           }

            auto& Rr = this->puttmp("Rr", new ExcitationOperator  <U,2,3>("Rr", arena, occ, vrt, isalpha_right ? -1 : 1));
            auto& Ri = this->puttmp("Ri", new ExcitationOperator  <U,2,3>("Ri", arena, occ, vrt, isalpha_right ? -1 : 1));
            auto& Zr = this->puttmp("Zr", new ExcitationOperator  <U,2,3>("Zr", arena, occ, vrt, isalpha_right ? -1 : 1));
            auto& Zi = this->puttmp("Zi", new ExcitationOperator  <U,2,3>("Zi", arena, occ, vrt, isalpha_right ? -1 : 1));
            auto& b  = this->puttmp("b",  new ExcitationOperator  <U,2,3>("b",  arena, occ, vrt, isalpha_right ? -1 : 1));
            auto& e  = this->puttmp("e",  new DeexcitationOperator<U,2,3>("e",  arena, occ, vrt, isalpha_left ? 1 : -1));

            auto& XE   = this->puttmp("XE",   new SpinorbitalTensor<U>("X(e)",    arena, group, {vrt,occ}, {0,0}, {1,0}, isalpha_right ? -1 : 1));
            auto& XMIJ = this->puttmp("XMIJ", new SpinorbitalTensor<U>("X(m,ij)", arena, group, {vrt,occ}, {0,1}, {0,2}, isalpha_right ? -1 : 1));
            auto& XAEI = this->puttmp("XAEI", new SpinorbitalTensor<U>("X(a,ei)", arena, group, {vrt,occ}, {1,0}, {1,1}, isalpha_right ? -1 : 1));

            SpinorbitalTensor<U> ap ("ap"  , arena, group, {vrt,occ}, {0,0}, {isvrt_right, !isvrt_right}, isalpha_right ? -1 : 1);
            SpinorbitalTensor<U> apt("ap^t", arena, group, {vrt,occ}, {isvrt_left, !isvrt_left}, {0,0}, isalpha_left ? 1 : -1);


            vector<tkv_pair<U>> pair_left{{orbleft_dummy, 1}};
            vector<tkv_pair<U>> pair_right{{orbright_dummy, 1}};


            CTFTensor<U>& tensor1 = ap({0,0}, {isvrt_right && isalpha_right, !isvrt_right && isalpha_right})({0});
                tensor1.writeRemoteData(pair_right);
            if (arena.rank == 0)
                tensor1.writeRemoteData(pair_right);
            else
                tensor1.writeRemoteData();

            CTFTensor<U>& tensor2 = apt({isvrt_left && isalpha_left, !isvrt_left && isalpha_left}, {0,0})({0});
                tensor2.writeRemoteData(pair_left);
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

                b(1)[  "i"] = T(1)[  "ei"]*ap["e"];
                b(2)["aij"] = T(2)["aeij"]*ap["e"];

                /*
                 *  ijk...   ijk...
                 * e  (e) = l
                 *  ab...    abe...
                 */

                e(1)[  "i"] = L(1)[  "ie"]*apt["e"];
                e(2)["ija"] = L(2)["ijae"]*apt["e"];

            }
            else if((isvrt_right) && (!isvrt_left))
            {
                /*
                 *  ab...    abe...
                 * b  (e) = t
                 *  q
                 *  ijk...   ijk...
                 */
                b(1)[  "i"] = T(1)[  "ei"]*ap["e"];
                b(2)["aij"] = T(2)["aeij"]*ap["e"];

                /*
                 *  ijk...           ij...     ijk...
                 * e  (m) = d  (1 + l     ) + G
                 *  ab...    km      ab...     abm...
                 */
                e(1)[  "i"]  =               apt["i"];

                e(1)[  "i"] -=   Dij[  "im"]*apt["m"];
                e(2)["ija"]  =  L(1)[  "ia"]*apt["j"];
                e(2)["ija"] -= Gijak["ijam"]*apt["m"];
                e(3)["ijkab"]  =    L(2)[  "ijab"]*apt["k"];

            }
            else if((!isvrt_right) && (isvrt_left))
            {
                /*
                 * b (m) = d
                 *  i       im
                 */
                b(1)["i"] = ap["i"];
                /*
                 *  ijk...   ijk...
                 * e  (e) = l
                 *  ab...    abe...
                 */
                e(1)[  "i"] = L(1)[  "ie"]*apt["e"];
                e(2)["ija"] = L(2)["ijae"]*apt["e"];

            } 
            else
            {
                /*
                 * b (m) = d
                 *  i       im
                 */
                b(1)["i"] = ap["i"];

                /*
                 *  ijk...           ij...     ijk...
                 * e  (m) = d  (1 + l     ) + G
                 *  ab...    km      ab...     abm...
                 */
                e(1)[  "i"]   =               apt["i"];

                e(1)[  "i"] -=   Dij[  "im"]*apt["m"];
                e(2)["ija"]  =  L(1)[  "ia"]*apt["j"];
                e(2)["ija"] -= Gijak["ijam"]*apt["m"];
                e(3)["ijkab"]  =    L(2)[  "ijab"]*apt["k"];

            }


            auto& D = this->puttmp("D", new ComplexDenominator<U>(H));

            int omega_counter = 0 ;
            for (auto& o : omegas)
            {

                this->puttmp("krylov", new ComplexLinearKrylov<ExcitationOperator<U,2,3>>(krylov_config, b));
                omega.real( o.real());
                omega.imag( o.imag());

                this->log(arena) << "Computing Green's function at " << fixed << setprecision(12) << o << endl;

                Rr = b;
                Ri = 0;
                D.weight(Rr, Ri, omega);
                U norm = sqrt(aquarius::abs(scalar(Rr*Rr)) +
                              aquarius::abs(scalar(Ri*Ri)));
                Rr /= norm;
                Ri /= norm;

                Iterative<CU>::run(dag, arena);
                gf_ip.emplace_back(value) ;
                omega_counter += 1 ;

               }
              }
            }

         if (arena.rank == 0)
         {
             int counter = 0 ;
             for (int nspin = 0 ; nspin < maxspin ; nspin++){
             stringstream stream1;
             stream1 << "gf_ip_"<<nspin<<"_"<<element_start<<"_"<<element_end<< ".txt";
             string fileName1 = stream1.str();
             std::ofstream gffile;
             gffile.open (fileName1, ofstream::out);

             for (int j=element_start ; j < element_end ; j++){
               for (int i=0 ; i < omegas.size() ; i++){
                   gffile << nspin << " " << j << " " << i << " " << setprecision(12) <<  gf_ip[counter].real() <<  " " << gf_ip[counter].imag() << endl ;
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

            auto& XE   = this->template gettmp<SpinorbitalTensor<U>>("XE");
            auto& XMIJ = this->template gettmp<SpinorbitalTensor<U>>("XMIJ");
            auto& XAEI = this->template gettmp<SpinorbitalTensor<U>>("XAEI");

            auto& D = this->template gettmp<ComplexDenominator<U>>("D");
            auto& krylov = this->template gettmp<ComplexLinearKrylov<ExcitationOperator<U,2,3>>>("krylov");

            auto& Rr = this->template gettmp<  ExcitationOperator<U,2,3>>("Rr");
            auto& Ri = this->template gettmp<  ExcitationOperator<U,2,3>>("Ri");
            auto& Zr = this->template gettmp<  ExcitationOperator<U,2,3>>("Zr");
            auto& Zi = this->template gettmp<  ExcitationOperator<U,2,3>>("Zi");
            auto& b  = this->template gettmp<  ExcitationOperator<U,2,3>>("b");
            auto& e  = this->template gettmp<DeexcitationOperator<U,2,3>>("e");

            //printf("<Rr|Rr>: %.15f\n", scalar(Rr*Rr));
            //printf("<Ri|Ri>: %.15f\n", scalar(Ri*Ri));
            //printf("<Rr|Ri>: %.15f\n", scalar(Rr*Ri));

            //printf("<B|Rr>: %.15f\n", scalar(b*Rr));
            //printf("<B|Ri>: %.15f\n", scalar(b*Ri));

            for (int ri: {0,1})
            {
                ExcitationOperator<U,2,3>& R = (ri == 0 ? Rr : Ri);
                ExcitationOperator<U,2,3>& Z = (ri == 0 ? Zr : Zi);

                  XE[    "e"]  = -0.5*WMNEF["mnfe"]*R(2)[   "fmn"];

                XMIJ[  "mij"]  =     -WMNIJ["mnij"]*R(1)[     "n"];
                XMIJ[  "mij"] +=      WMNEJ["nmei"]*R(2)[   "enj"];
                XMIJ[  "mij"] +=  0.5*WMNEF["mnef"]*R(3)[ "efinj"];

                XAEI[  "aei"]  =     -WAMEI["amei"]*R(1)[     "m"];
                XAEI[  "aei"] +=      WAMEF["amef"]*R(2)[   "fmi"];
                XAEI[  "aei"] +=  0.5*WMNEJ["mnei"]*R(2)[   "amn"];
                XAEI[  "aei"] -=  0.5*WMNEF["mnef"]*R(3)[ "afmni"];

                Z(1)[    "i"]  =       -FMI[  "mi"]*R(1)[     "m"];
                Z(1)[    "i"] +=        FME[  "me"]*R(2)[   "emi"];
                Z(1)[    "i"] -=  0.5*WMNEJ["mnei"]*R(2)[   "emn"];
                Z(1)[    "i"] += 0.25*WMNEF["mnef"]*R(3)[ "efmni"];

                Z(2)[  "aij"]  =     -WAMIJ["amij"]*R(1)[     "m"];
                Z(2)[  "aij"] +=        FAE[  "ae"]*R(2)[   "eij"];
                Z(2)[  "aij"] -=        FMI[  "mi"]*R(2)[   "amj"];
                Z(2)[  "aij"] +=  0.5*WMNIJ["mnij"]*R(2)[   "amn"];
                Z(2)[  "aij"] -=      WAMEI["amei"]*R(2)[   "emj"];
                Z(2)[  "aij"] +=         XE[   "e"]*T(2)[  "aeij"];
                Z(2)[  "aij"] +=        FME[  "me"]*R(3)[ "eamij"];
                Z(2)[  "aij"] +=  0.5*WAMEF["amef"]*R(3)[ "efimj"];
                Z(2)[  "aij"] -=  0.5*WMNEJ["mnej"]*R(3)[ "aeimn"];

                Z(3)["abijk"]  =      WABEJ["abej"]*R(2)[   "eik"];
                Z(3)["abijk"] -=      WAMIJ["amij"]*R(2)[   "bmk"];
                Z(3)["abijk"] -=       XMIJ[ "mik"]*T(2)[  "abmj"];
                Z(3)["abijk"] -=       XAEI[ "aei"]*T(2)[  "bejk"];
                Z(3)["abijk"] +=        FAE[  "ae"]*R(3)[ "ebijk"];
                Z(3)["abijk"] -=        FMI[  "mi"]*R(3)[ "abmjk"];
                Z(3)["abijk"] -=      WAMEI["amei"]*R(3)[ "ebmjk"];
                Z(3)["abijk"] +=  0.5*WABEF["abef"]*R(3)[ "efijk"];
                Z(3)["abijk"] +=  0.5*WMNIJ["mnij"]*R(3)[ "abmnk"];
            }

            //printf("<Z1|Z1>: %.15f\n", scalar(Zr(1)*Zr(1)));
            //printf("<Z2|Z2>: %.15f\n", 0.5*scalar(Zr(2)*Zr(2)));
            //printf("<Zr|Zr>: %.15f\n", scalar(Zr*Zr));
            //printf("<Zi|Zi>: %.15f\n", scalar(Zi*Zi));

            /*
             * Convert H*r to (H-w)*r
             */
//          Zr -= omega.real()*Rr;
//          Zr += omega.imag()*Ri;
//          Zi -= omega.real()*Ri;
//          Zi -= omega.imag()*Rr;

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

            this->energy() = CU(           scalar(e(1)[    "m"]*Zr(1)[    "m"]) +
                                (1.0/ 2.0)*scalar(e(2)[  "mne"]*Zr(2)[  "emn"]) +
                                (1.0/12.0)*scalar(e(3)["mnoef"]*Zr(3)["efmno"]),
                                           scalar(e(1)[    "m"]*Zi(1)[    "m"]) +
                                (1.0/ 2.0)*scalar(e(2)[  "mne"]*Zi(2)[  "emn"]) +
                                (1.0/12.0)*scalar(e(3)["mnoef"]*Zi(3)["efmno"]));

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

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::CCSDTIPGF_D_T);
REGISTER_TASK(aquarius::cc::CCSDTIPGF_D_T<double>, "ccsdtipgf_d_t",spec);
