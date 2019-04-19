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
class CCSDIPGF : public Iterative<complex_type_t<U>>
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
        string orb_range ;

    public:
        CCSDIPGF(const string& name, Config& config)
        : Iterative<CU>(name, config), krylov_config(config.get("krylov"))
        {
            vector<Requirement> reqs;
            reqs.emplace_back("ccsd.T", "T");
            reqs.emplace_back("ccsd.L", "L");
            reqs.emplace_back("ccsd.Hbar", "Hbar");
            this->addProduct("ccsd.ipgf", "gf_ip", reqs);

            orbital = config.get<int>("orbital");
            double from = config.get<double>("omega_min");
            double to = config.get<double>("omega_max");
            int n = config.get<double>("npoint");
            double eta = config.get<double>("eta");
            double beta = config.get<double>("beta");
            string grid_type = config.get<string>("grid");
            orb_range = config.get<string>("orbital_range");

            double delta = (to-from)/max(1,n-1);
            for (int i = 0;i < n;i++)
            {
             if (grid_type == "real") omegas.emplace_back(from+delta*i, eta);
             if (grid_type == "imaginary") omegas.emplace_back(0.,(2.0*i+1)*M_PI/beta);
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

            int maxspin = (nI == ni) ? 1 : 2 ;

            auto& T = this->template get<ExcitationOperator  <U,2>>("T");
            auto& L = this->template get<DeexcitationOperator<U,2>>("L");

            auto& gf_ip = this-> put("gf_ip", new vector<vector<vector<vector<CU>>>>) ;
            
            SpinorbitalTensor<U> Dij("D(ij)", arena, group, {vrt,occ}, {0,1}, {0,1});
            SpinorbitalTensor<U> Gijak("G(ij,ak)", arena, group, {vrt,occ}, {0,2}, {1,1});

            SpinorbitalTensor<U> Nij("N(ij)", arena, group, {vrt,occ}, {0,1}, {0,1});

            Dij["ij"]  =     L(1)["ie"  ]*T(1)["ej"  ];
            Dij["ij"] += 0.5*L(2)["imef"]*T(2)["efjm"];

            Gijak["ijak"] = L(2)["ijae"]*T(1)["ek"];

          if (orb_range == "full") 
           { orbstart = 0 ;
             orbend = nI + nA ;  
             gf_ip.resize(maxspin);

            for (int nspin = 0;nspin < maxspin;nspin++)
             {
              gf_ip[nspin].resize(omegas.size());
             }  

            for (int nspin = 0;nspin < maxspin;nspin++)
             {
              for (int i = 0;i < omegas.size();i++)
               {
                gf_ip[nspin][i].resize(orbend);
               }
             }
             for (int nspin = 0;nspin < maxspin;nspin++)
             for (int i = 0;i < omegas.size();i++)
             for (int j = 0;j < orbend;j++)
             {
               gf_ip[nspin][i][j].resize(orbend);
             }
           } 

           if (orb_range == "diagonal") 
           { orbstart = orbital-1 ;
             orbend = orbital;  
             gf_ip.resize(maxspin);
            for (int nspin = 0;nspin < maxspin;nspin++)
             {
              gf_ip[nspin].resize(omegas.size());
             }  
            for (int nspin = 0;nspin < maxspin;nspin++)
             {
              for (int i = 0;i < omegas.size();i++)
               {
                gf_ip[nspin][i].resize(1);
               }
             }
             for (int nspin = 0;nspin < maxspin;nspin++)
             for (int i = 0;i < omegas.size();i++)
             for (int j = 0;j < 1;j++)
             {
               gf_ip[nspin][i][j].resize(1);
             }
           }

        for (int nspin = 0; nspin < maxspin ; nspin++)   {
          for (int orbleft = orbstart; orbleft < orbend ; orbleft++)   {
           for (int orbright = orbstart; orbright < orbend ; orbright++) {  

             printf("Computing Green's function element:  %d %d\n", orbleft, orbright ) ;

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

            auto& Rr = this->puttmp("Rr", new ExcitationOperator  <U,1,2>("Rr", arena, occ, vrt, isalpha_right ? -1 : 1));
            auto& Ri = this->puttmp("Ri", new ExcitationOperator  <U,1,2>("Ri", arena, occ, vrt, isalpha_right ? -1 : 1));
            auto& Zr = this->puttmp("Zr", new ExcitationOperator  <U,1,2>("Zr", arena, occ, vrt, isalpha_right ? -1 : 1));
            auto& Zi = this->puttmp("Zi", new ExcitationOperator  <U,1,2>("Zi", arena, occ, vrt, isalpha_right ? -1 : 1));
            auto& b  = this->puttmp("b",  new ExcitationOperator  <U,1,2>("b",  arena, occ, vrt, isalpha_right ? -1 : 1));
            auto& e  = this->puttmp("e",  new DeexcitationOperator<U,1,2>("e",  arena, occ, vrt, isalpha_right ? 1 : -1));

            auto& XE = this->puttmp("XE", new SpinorbitalTensor<U>("X(e)", arena, group, {vrt,occ}, {0,0}, {1,0}, isalpha_right ? -1 : 1));

            SpinorbitalTensor<U> ap ("ap"  , arena, group, {vrt,occ}, {0,0}, {isvrt_right, !isvrt_right}, isalpha_right ? -1 : 1);
            SpinorbitalTensor<U> apt("ap^t", arena, group, {vrt,occ}, {isvrt_left, !isvrt_left}, {0,0}, isalpha_left ? 1 : -1);

            vector<tkv_pair<U>> pair_left{{orbleft_dummy, 1.0}};
            vector<tkv_pair<U>> pair_right{{orbright_dummy, 1.0}};

            CTFTensor<U>& tensor1 = ap({0,0}, {isvrt_right && isalpha_right, !isvrt_right && isalpha_right})({0});
            if (arena.rank == 0)
                tensor1.writeRemoteData(pair_right);
            else
                tensor1.writeRemoteData();

            CTFTensor<U>& tensor2 = apt({isvrt_left && isalpha_left, !isvrt_left && isalpha_left}, {0,0})({0});
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
                e(1)[  "i"]  =               apt["i"];

                e(1)[  "i"] -=   Dij[  "im"]*apt["m"];
                e(2)["ija"]  =  L(1)[  "ia"]*apt["j"];
                e(2)["ija"] -= Gijak["ijam"]*apt["m"];
            }

            auto& D = this->puttmp("D", new ComplexDenominator<U>(H));

            int omega_counter = 0 ;

            for (auto& o : omegas)
            {
                this->puttmp("krylov", new ComplexLinearKrylov<ExcitationOperator<U,1,2>>(krylov_config, b));
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

                if (orb_range == "full") gf_ip[nspin][omega_counter][orbleft][orbright] = value ;
                if (orb_range == "diagonal") gf_ip[nspin][omega_counter][0][0] = value ;

              printf("real value : %.15f\n", value.real());
              printf("imaginary value : %.15f\n", value.imag());

              omega_counter += 1 ;
            }
           }
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
            const SpinorbitalTensor<U>& WMNIJ = H.getIJKL();
            const SpinorbitalTensor<U>& WMNEJ = H.getIJAK();
            const SpinorbitalTensor<U>& WAMIJ = H.getAIJK();
            const SpinorbitalTensor<U>& WAMEI = H.getAIBJ();

            auto& T = this->template get<ExcitationOperator<U,2>>("T");

            auto& XE = this->template gettmp<SpinorbitalTensor<U>>("XE");

            auto& D = this->template gettmp<ComplexDenominator<U>>("D");
            auto& krylov = this->template gettmp<ComplexLinearKrylov<ExcitationOperator<U,1,2>>>("krylov");

            auto& Rr = this->template gettmp<  ExcitationOperator<U,1,2>>("Rr");
            auto& Ri = this->template gettmp<  ExcitationOperator<U,1,2>>("Ri");
            auto& Zr = this->template gettmp<  ExcitationOperator<U,1,2>>("Zr");
            auto& Zi = this->template gettmp<  ExcitationOperator<U,1,2>>("Zi");
            auto& b  = this->template gettmp<  ExcitationOperator<U,1,2>>("b");
            auto& e  = this->template gettmp<DeexcitationOperator<U,1,2>>("e");

            //printf("<Rr|Rr>: %.15f\n", scalar(Rr*Rr));
            //printf("<Ri|Ri>: %.15f\n", scalar(Ri*Ri));
            //printf("<Rr|Ri>: %.15f\n", scalar(Rr*Ri));

            //printf("<B|Rr>: %.15f\n", scalar(b*Rr));
            //printf("<B|Ri>: %.15f\n", scalar(b*Ri));

            for (int ri: {0,1})
            {
                ExcitationOperator<U,1,2>& R = (ri == 0 ? Rr : Ri);
                ExcitationOperator<U,1,2>& Z = (ri == 0 ? Zr : Zi);

                  XE[  "e"]  = -0.5*WMNEF["mnfe"]*R(2)[ "fmn"];

                Z(1)[  "i"]  =       -FMI[  "mi"]*R(1)[   "m"];
                Z(1)[  "i"] +=        FME[  "me"]*R(2)[ "emi"];
                Z(1)[  "i"] -=  0.5*WMNEJ["mnei"]*R(2)[ "emn"];

                Z(2)["aij"]  =     -WAMIJ["amij"]*R(1)[   "m"];
                Z(2)["aij"] +=        FAE[  "ae"]*R(2)[ "eij"];
                Z(2)["aij"] -=        FMI[  "mi"]*R(2)[ "amj"];
                Z(2)["aij"] +=         XE[   "e"]*T(2)["aeij"];
                Z(2)["aij"] +=  0.5*WMNIJ["mnij"]*R(2)[ "amn"];
                Z(2)["aij"] -=      WAMEI["amei"]*R(2)[ "emj"];
            }

            //printf("<Z1|Z1>: %.15f\n", scalar(Zr(1)*Zr(1)));
            //printf("<Z2|Z2>: %.15f\n", 0.5*scalar(Zr(2)*Zr(2)));
            //printf("<Zr|Zr>: %.15f\n", scalar(Zr*Zr));
            //printf("<Zi|Zi>: %.15f\n", scalar(Zi*Zi));

            /*
             * Convert H*r to (H-w)*r
             */
              Zr += omega.real()*Rr;
              Zr -= omega.imag()*Ri;
              Zi += omega.real()*Ri;
              Zi += omega.imag()*Rr;

//           Zr += omega.real()*Rr;
//           Zr -= omega.imag()*Ri;
//           Zi += omega.real()*Ri;
//           Zi += omega.imag()*Rr;

//            Zr *= -1;
//            Zi *= -1;

            //printf("<Ur|Ur>: %.15f\n", scalar(Zr*Zr));
            //printf("<Ui|Ui>: %.15f\n", scalar(Zi*Zi));

            krylov.extrapolate(Rr, Ri, Zr, Zi, D, omega);

            this->conv() = max(Zr.norm(00), Zi.norm(00));

            krylov.getSolution(Zr, Zi);

            this->energy() = CU(    scalar(e(1)[  "m"]*Zr(1)[  "m"]) +
                                0.5*scalar(e(2)["mne"]*Zr(2)["emn"]),
                                    scalar(e(1)[  "m"]*Zi(1)[  "m"]) +
                                0.5*scalar(e(2)["mne"]*Zi(2)["emn"]));

              value = {scalar(e(1)[  "m"]*Zr(1)[  "m"]) + 0.5*scalar(e(2)["mne"]*Zr(2)["emn"]), scalar(e(1)[  "m"]*Zi(1)[  "m"]) + 0.5*scalar(e(2)["mne"]*Zi(2)["emn"])} ;

        }
};

}
}

static const char* spec = R"(

orbital int,
npoint int,
omega_min double,
omega_max double,
eta double,
grid?
  enum{ real, imaginary },
orbital_range?
  enum{ diagonal, full},
beta?
   double 100.0 , 
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

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::CCSDIPGF);
REGISTER_TASK(aquarius::cc::CCSDIPGF<double>, "ccsdipgf",spec);
