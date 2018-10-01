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
class CCSDEAGF : public Iterative<complex_type_t<U>>
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
        CCSDEAGF(const string& name, Config& config)
        : Iterative<CU>(name, config), krylov_config(config.get("krylov"))
        {
            vector<Requirement> reqs;
            reqs.emplace_back("ccsd.T", "T");
            reqs.emplace_back("ccsd.L", "L");
            reqs.emplace_back("ccsd.Hbar", "Hbar");
            this->addProduct("ccsd.eagf", "gf_ea", reqs);

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

            auto& gf_ea = this-> put("gf_ea", new vector<vector<vector<vector<CU>>>>) ;
            
            SpinorbitalTensor<U> Dab("D(ab)", arena, group, {vrt,occ}, {1,0}, {1,0});
            SpinorbitalTensor<U> Gieab("G(am,ef)", arena, group, {vrt,occ}, {1,1}, {2,0});

            Dab["ab"]  =     -L(1)["mb"  ]*T(1)["am"  ];
            Dab["ab"] -= 0.5*L(2)["kmbe"]*T(2)["aekm"];

            Gieab["amef"]  = -L(2)["nmef"]*T(1)[  "an"];

          if (orb_range == "full") 
           { orbstart = 0 ;
             orbend = nI + nA ;  
             gf_ea.resize(maxspin);

            for (int nspin = 0;nspin < maxspin;nspin++)
             {
              gf_ea[nspin].resize(omegas.size());
             }  

            for (int nspin = 0;nspin < maxspin;nspin++)
             {
              for (int i = 0;i < omegas.size();i++)
               {
                gf_ea[nspin][i].resize(orbend);
               }
             }
             for (int nspin = 0;nspin < maxspin;nspin++)
             for (int i = 0;i < omegas.size();i++)
             for (int j = 0;j < orbend;j++)
             {
               gf_ea[nspin][i][j].resize(orbend);
             }
           } 

           if (orb_range == "diagonal") 
           { orbstart = orbital-1 ;
             orbend = orbital;  
             gf_ea.resize(maxspin);
            for (int nspin = 0;nspin < maxspin;nspin++)
             {
              gf_ea[nspin].resize(omegas.size());
             }  
            for (int nspin = 0;nspin < maxspin;nspin++)
             {
              for (int i = 0;i < omegas.size();i++)
               {
                gf_ea[nspin][i].resize(1);
               }
             }
             for (int nspin = 0;nspin < maxspin;nspin++)
             for (int i = 0;i < omegas.size();i++)
             for (int j = 0;j < 1;j++)
             {
               gf_ea[nspin][i][j].resize(1);
             }
           }

        for (int nspin = 0; nspin < maxspin ; nspin++)   
         {
          for (int orbleft = orbstart; orbleft < orbend ; orbleft++)   
          {
           for (int orbright = orbstart; orbright < orbend ; orbright++)   
            {

             printf("Computing Green's function element:  %d %d\n", orbleft, orbright ) ;

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

            auto& Rr = this->puttmp("Rr", new ExcitationOperator  <U,2,1>("Rr", arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& Ri = this->puttmp("Ri", new ExcitationOperator  <U,2,1>("Ri", arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& Zr = this->puttmp("Zr", new ExcitationOperator  <U,2,1>("Zr", arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& Zi = this->puttmp("Zi", new ExcitationOperator  <U,2,1>("Zi", arena, occ, vrt, isalpha_right ? 1 : -1));

            auto& b  = this->puttmp("b",  new ExcitationOperator  <U,2,1>("b",  arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& e  = this->puttmp("e",  new DeexcitationOperator<U,2,1>("e",  arena, occ, vrt, isalpha_right ? -1 : 1));

            auto& XMI = this->puttmp("XMI", new SpinorbitalTensor<U>("X(mi)", arena, group, {vrt,occ}, {0,1}, {0,0}, isalpha_left ? 1 : -1));

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

                /*
                 *  ijk...   ijk...
                 * e  (e) = l
                 *  ab...    abe...
                 */
                e(1)[  "a"]  = -L(1)[  "ka"]*apt["k"];
                e(2)["iab"]  = -L(2)["ikab"]*apt["k"];
            }
            else if((isvrt_right) && (!isvrt_left))
            {
                b(1)[  "a"] = -T(1)[  "ak"]*ap["k"];
                b(2)["abi"] = -T(2)["abik"]*ap["k"];

                e(1)[  "a"]  =               apt["a"];
                e(1)[  "a"]  +=   Dab[  "ea"]*apt["e"];
                e(2)["iab"]  =  L(1)[  "ia"]*apt["b"];
                e(2)["iab"]  += Gieab["eiba"]*apt["e"];
            }
            else if((!isvrt_right) && (isvrt_left))
            {
                b(1)["a"] = ap["a"];

                e(1)[  "a"]  = -L(1)[  "ka"]*apt["k"];
                e(2)["iab"]  = -L(2)["ikab"]*apt["k"];
            } 
            else
            {
                b(1)["a"] = ap["a"];

                e(1)[  "a"]  =               apt["a"];
                e(1)[  "a"]  +=   Dab[  "ea"]*apt["e"];
                e(2)["iab"]  =  L(1)[  "ia"]*apt["b"];
                e(2)["iab"]  += Gieab["eiba"]*apt["e"];
            }

            auto& D = this->puttmp("D", new ComplexDenominator<U>(H));

            int omega_counter = 0 ;
            for (auto& o : omegas)
            {
                this->puttmp("krylov", new ComplexLinearKrylov<ExcitationOperator<U,2,1>>(krylov_config, b));
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
                if (orb_range == "full") gf_ea[nspin][omega_counter][orbleft][orbright] = value ;
                if (orb_range == "diagonal") gf_ea[nspin][omega_counter][0][0] = value ;
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
            const SpinorbitalTensor<U>& WAMEF = H.getAIBC();
            const SpinorbitalTensor<U>& WABEF = H.getABCD();
            const SpinorbitalTensor<U>& WABEJ = H.getABCI();

            auto& T = this->template get<ExcitationOperator<U,2>>("T");

            auto& XMI = this->template gettmp<SpinorbitalTensor<U>>("XMI");

            auto& D = this->template gettmp<ComplexDenominator<U>>("D");
            auto& krylov = this->template gettmp<ComplexLinearKrylov<ExcitationOperator<U,2,1>>>("krylov");

            auto& Rr = this->template gettmp< ExcitationOperator<U,2,1>>("Rr");
            auto& Ri = this->template gettmp< ExcitationOperator<U,2,1>>("Ri");
            auto& Zr = this->template gettmp< ExcitationOperator<U,2,1>>("Zr");
            auto& Zi = this->template gettmp< ExcitationOperator<U,2,1>>("Zi");
            auto& b  = this->template gettmp<  ExcitationOperator<U,2,1>>("b");
            auto& e  = this->template gettmp<DeexcitationOperator<U,2,1>>("e");


            for (int ri: {0,1})
            {
                ExcitationOperator<U,2,1>& R = (ri == 0 ? Rr : Ri);
                ExcitationOperator<U,2,1>& Z = (ri == 0 ? Zr : Zi);

                XMI[  "m"] = -0.5*WMNEF["mnef"]*R(2)["efn"];

                Z(1)[  "a"]  =       FAE[  "ae"]*R(1)[  "e"];
                Z(1)[  "a"] -=       FME[  "me"]*R(2)["aem"];
                Z(1)[  "a"] -= 0.5*WAMEF["amef"]*R(2)["efm"];

                Z(2)["abi"]   =     WABEJ["baei"]*R(1)[  "e"];
                Z(2)["abi"]  +=       FAE[  "ae"]*R(2)["ebi"];
                Z(2)["abi"]  -=       FMI[  "mi"]*R(2)["abm"];
                Z(2)["abi"]  -=       XMI[  "m"]*T(2)["abim"];
                Z(2)["abi"]  += 0.5*WABEF["abef"]*R(2)["efi"];
                Z(2)["abi"]  -=     WAMEI["amei"]*R(2)["ebm"];
            }

 /*
  * Convert H*r to (w-H)*r
  */

              Zr *= -1;
              Zi *= -1;

              Zr += omega.real()*Rr;
              Zr -= omega.imag()*Ri;
              Zi += omega.real()*Ri;
              Zi += omega.imag()*Rr;

            //printf("<Ur|Ur>: %.15f\n", scalar(Zr*Zr));
            //printf("<Ui|Ui>: %.15f\n", scalar(Zi*Zi));

            krylov.extrapolate(Rr, Ri, Zr, Zi, D, omega);

            this->conv() = max(Zr.norm(00), Zi.norm(00));

            krylov.getSolution(Zr, Zi);

            this->energy() = CU(    scalar(e(1)[  "a"]*Zr(1)[  "a"]) +
                                0.5*scalar(e(2)["iab"]*Zr(2)["abi"]),
                                    scalar(e(1)[  "a"]*Zi(1)[  "a"]) +
                                0.5*scalar(e(2)["iab"]*Zi(2)["abi"]));
 
            value = {scalar(e(1)[  "a"]*Zr(1)[  "a"]) + 0.5*scalar(e(2)["iab"]*Zr(2)["abi"]), scalar(e(1)[  "a"]*Zi(1)[  "a"])+0.5*scalar(e(2)["iab"]*Zi(2)["abi"])};
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

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::CCSDEAGF);
REGISTER_TASK(aquarius::cc::CCSDEAGF<double>, "ccsdeagf",spec);
