#include "util/global.hpp"

#include "time/time.hpp"
#include "task/task.hpp"
#include "convergence/lanczos.hpp"
#include "util/iterative.hpp"
#include "operator/2eoperator.hpp"
#include "operator/st2eoperator.hpp"
#include "operator/space.hpp"
#include "operator/excitationoperator.hpp"
#include "operator/deexcitationoperator.hpp"
#include "operator/denominator.hpp"
#include "hubbard/uhf_modelH.hpp"

using namespace aquarius::tensor;
using namespace aquarius::task;
using namespace aquarius::input;
using namespace aquarius::op;
using namespace aquarius::hubbard;
using namespace aquarius::convergence;
using namespace aquarius::symmetry;

namespace aquarius
{
namespace cc
{

template <typename U>
class CCSDEAGF_LANCZOS : public Iterative<U>
{
    protected:
        typedef U X ; 
        typedef complex_type_t<U> CU;
        Config lanczos_config;
        int orbital;
        int orbstart;
        int orbend;
        vector<CU> omegas;
        CU omega;
        vector<U> old_value ;
        vector<U> integral_diagonal ;
        vector<U> v_onsite ;
        string orb_range ;

    public:
        CCSDEAGF_LANCZOS(const string& name, Config& config)
        : Iterative<U>(name, config), lanczos_config(config.get("lanczos"))
        {
            vector<Requirement> reqs;
            reqs.emplace_back("ccsd.T", "T");
            reqs.emplace_back("ccsd.L", "L");
            reqs.emplace_back("ccsd.Hbar", "Hbar");
            this->addProduct(Product("ccsd.eagf", "gf_ea", reqs));
            this->addProduct(Product("ccsd.eaalpha", "alpha_ea", reqs));
            this->addProduct(Product("ccsd.eabeta", "beta_ea", reqs));
            this->addProduct(Product("ccsd.eagamma", "gamma_ea", reqs));
            this->addProduct(Product("ccsd.eanorm", "norm_ea", reqs));

            orbital = config.get<int>("orbital");
            double from = config.get<double>("omega_min");
            double to = config.get<double>("omega_max");
            int n = config.get<int>("npoint");
            double eta = config.get<double>("eta");
            double beta = config.get<double>("beta");
            string grid_type = config.get<string>("grid");
            orb_range = config.get<string>("orbital_range");

            double delta = (to-from)/max(1,n-1);
            for (int i = 0;i < n;i++){
             if (grid_type == "real") omegas.emplace_back(from+delta*i, eta);
             if (grid_type == "imaginary") omegas.emplace_back(0.,(2.0*i+1)*M_PI/beta);
            }

            ifstream ifs("wlist_sub.txt");
            if (ifs){
            omegas.clear() ; 
            string line;
            while (getline(ifs, line))
            {
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
            int nvec_lanczos; 
            CU value ;
            CU value1 ;
            int nr_tasks ;
            int nsize ;
            int element_start ;
            int element_end ;
            int orbleft ;
            int orbright ;

            int maxspin = (nI == ni) ? 1 : 2 ;

            auto& T = this->template get<ExcitationOperator  <U,2>>("T");
            auto& L = this->template get<DeexcitationOperator<U,2>>("L");

            auto& gf_ea = this-> put("gf_ea", new vector<vector<vector<CU>>>) ;
            auto& alpha_ea = this-> put("alpha_ea", new vector<vector<U>>) ;
            auto& beta_ea = this-> put("beta_ea", new vector<vector<U>>) ;
            auto& gamma_ea = this-> put("gamma_ea", new vector<vector<U>>) ;
            auto& norm_ea = this-> put("norm_ea", new vector<U>) ;

            SpinorbitalTensor<U> Dab("D(ab)", arena, group, {vrt,occ}, {1,0}, {1,0});
            SpinorbitalTensor<U> Gieab("G(am,ef)", arena, group, {vrt,occ}, {1,1}, {2,0});

            SpinorbitalTensor<U> Nij("N(ij)", arena, group, {vrt,occ}, {0,1}, {0,1});

            Dab["ab"]  =     -L(1)["mb"  ]*T(1)["am"  ];
            Dab["ab"] -= 0.5*L(2)["kmbe"]*T(2)["aekm"];

            Gieab["amef"]  = -L(2)["nmef"]*T(1)[  "an"];

           if (orb_range == "full") 
           {
             nr_tasks =  (nI+nA)*((nI+nA)+1)/2 ;
             nsize = int(floor(nr_tasks/arena.size)) ; 

             element_start = arena.rank*nsize ;
             if (arena.rank < (arena.size-1)) {
               element_end = element_start + nsize ;
             } else {
               element_end = element_start+nr_tasks - (nsize*(arena.size-1)) ;
             }

             alpha_ea.resize(element_end-element_start) ;
             beta_ea.resize(element_end-element_start) ;
             gamma_ea.resize(element_end-element_start) ;

             gf_ea.resize(maxspin);

            for (int nspin = 0;nspin < maxspin;nspin++){
              gf_ea[nspin].resize(omegas.size());
             }  

            for (int nspin = 0;nspin < maxspin;nspin++){
              for (int i = 0;i < omegas.size();i++){
                gf_ea[nspin][i].resize(element_end-element_start);
               }
             }
           } 

           if (orb_range == "diagonal") 
           { orbstart = orbital-1 ;
             orbend = orbital;  
             alpha_ea.resize(1) ;
             beta_ea.resize(1) ;
             gamma_ea.resize(1) ;

            gf_ea.resize(maxspin);

            for (int nspin = 0;nspin < maxspin;nspin++){
              gf_ea[nspin].resize(omegas.size());
            }  

            for (int nspin = 0;nspin < maxspin;nspin++){
              for (int i = 0;i < omegas.size();i++){
                gf_ea[nspin][i].resize(1);
               }
             }
           }

          vector<CU> spec_func(omegas.size()) ;

        vector<int> array1(nr_tasks);
        vector<int> array2(nr_tasks);
        vector< pair <int,int> > get_index ; 

         int x = 0 ; 
         for (int orbleft = 0; orbleft < (nI+nA) ; orbleft++){   
           for (int orbright = orbleft; orbright < (nI+nA) ; orbright++){   
             array1[x] = orbleft ;
             array2[x] = orbright ;
             x += 1 ;
          }
         }

         for (int i = 0; i < (nI+nA)*((nI+nA)+1)/2 ; i++){   
            get_index.push_back( make_pair(array1[i],array2[i]) );
         }

         if (arena.rank ==0){
           std::ifstream iffile("gomega_ea.dat");
           if (iffile) remove("gomega_ea.dat");
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

  /* vector LL means Left Lanczos and RL means Right Lanczos, in case you are wondering!  
   */
            auto& RL = this->puttmp("RL", new ExcitationOperator  <U,2,1>("RL", arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& LL = this->puttmp("LL", new DeexcitationOperator  <U,2,1>("LL", arena, occ, vrt, isalpha_left ? -1 : 1));
            auto& Z = this->puttmp("Z", new ExcitationOperator  <U,2,1>("Z", arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& Y = this->puttmp("Y", new DeexcitationOperator  <U,2,1>("Y", arena, occ, vrt, isalpha_left ? -1 : 1));
            auto& b  = this->puttmp("b",  new ExcitationOperator  <U,2,1>("b",  arena, occ, vrt, isalpha_right ? 1 : -1));
            auto& e  = this->puttmp("e",  new DeexcitationOperator<U,2,1>("e",  arena, occ, vrt, isalpha_left ? -1 : 1));

            auto& XMI = this->puttmp("XMI", new SpinorbitalTensor<U>("X(mi)", arena, group, {vrt,occ}, {0,1}, {0,0}, isalpha_left ? 1 : -1));
            auto& GIM = this->puttmp("GIM", new SpinorbitalTensor<U>("G(im)", arena, group, {vrt,occ}, {0,0}, {0,1}, isalpha_right ? -1 : 1));
            auto& alpha = this-> puttmp("alpha", new unique_vector<U>()) ;
            auto& beta  = this-> puttmp("beta", new unique_vector<U>()) ;
            auto& gamma = this-> puttmp("gamma", new unique_vector<U>());

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

                if (orbright != orbleft)
                {
                 b(1)[  "a"] -= T(1)[  "ak"]*apt["k"];
                 b(2)["abi"] -= T(2)["abik"]*apt["k"];

                 e(1)[  "a"] -= L(1)[  "ka"]*ap["k"];
                 e(2)["iab"] -= L(2)["ikab"]*ap["k"];
                }
            }

            else if((isvrt_right) && (!isvrt_left))
            {
                /*
                 *  ab...    abe...
                 * b  (e) = t
                 *  ijk...   ijk...
                 */

                b(1)[  "a"]  =              apt["a"]; //new
                b(1)[  "a"] -= T(1)[  "ak"]*ap["k"];
                b(2)["abi"] = -T(2)["abik"]*ap["k"];
                /*
                 *  ijk...           ij...     ijk...
                 * e  (m) = d  (1 + l     ) + G
                 *  ab...    km      ab...     abm...
                 */
                e(1)[  "a"]  =               apt["a"];

                e(1)[  "a"]  +=   Dab[  "ea"]*apt["e"];
                e(2)["iab"]  =  L(1)[  "ia"]*apt["b"];
                e(2)["iab"]  += Gieab["eiba"]*apt["e"];
                e(1)[  "a"]  -= L(1)[  "ka"]*ap["k"];  //new
                e(2)["iab"]  -= L(2)["ikab"]*ap["k"];  //new

            }
            else if((!isvrt_right) && (isvrt_left))
            {
                /*
                 * b (m) = d
                 *  i       im
                 */
                b(1)["a"] = ap["a"];
                b(1)[  "a"] -= T(1)[  "ak"]*apt["k"]; //new
                b(2)["abi"] = -T(2)["abik"]*apt["k"]; //new

                /*
                 *  ijk...   ijk...
                 * e  (e) = l
                 *  ab...    abe...
                 */

                e(1)[  "a"]   =               ap["a"];
                e(1)[  "a"]  -= L(1)[  "ka"]*apt["k"];
                e(2)["iab"]  = -L(2)["ikab"]*apt["k"];

                e(1)[  "a"]  +=   Dab[  "ea"]*ap["e"];
                e(2)["iab"]  +=  L(1)[  "ia"]*ap["b"];
                e(2)["iab"]  += Gieab["eiba"]*ap["e"];
            }
            else
            {
                /*
                 * b (m) = d
                 *  i       im
                 */
                b(1)["a"] = ap["a"];

                /*
                 *  ijk...           ij...     ijk...
                 * e  (m) = d  (1 + l     ) + G
                 *  ab...    km      ab...     abm...
                 */
                e(1)[  "a"]  =               apt["a"];
                e(1)[  "a"]  +=   Dab[  "ea"]*apt["e"];
                e(2)["iab"]  =  L(1)[  "ia"]*apt["b"];
                e(2)["iab"]  += Gieab["eiba"]*apt["e"];

               if (orbright != orbleft)
              {
                b(1)["a"] += apt["a"];  //new
                e(1)[  "a"]  +=               ap["a"]; //extra
                e(1)[  "a"]  +=   Dab[  "ea"]*ap["e"];
                e(2)["iab"]  +=  L(1)[  "ia"]*ap["b"];
                e(2)["iab"]  += Gieab["eiba"]*ap["e"];
              }
            }

              auto& D = this->puttmp("D", new Denominator<U>(H));
           
              int number_of_vectors = nI*nA*nA + nA ; 
              this->puttmp("lanczos", new Lanczos<U,X>(lanczos_config,number_of_vectors));

              RL = b ;
              LL = e ;
            
             /* Evaluate norm 
              */ 

              U norm = sqrt(aquarius::abs(scalar(RL*LL))); 

              norm_ea.emplace_back(norm*norm) ;

              RL /= norm;
              LL /= norm;

              this->log(arena) << "print norm: " << norm << endl ;

              Iterative<U>::run(dag, arena);

              nvec_lanczos = alpha.size() ; 

              alpha_ea[uppertriangle].resize(nvec_lanczos) ;
              beta_ea[uppertriangle].resize(nvec_lanczos) ;
              gamma_ea[uppertriangle].resize(nvec_lanczos) ;

  /*Define full trdiagonal matrix 
   */  
              vector<U> Tdiag(nvec_lanczos*nvec_lanczos);

              for (int i=0 ; i < nvec_lanczos ; i++){
                for (int j=0 ; j < nvec_lanczos ; j++){
                 if (j==i) Tdiag[i*nvec_lanczos + j] = alpha[i] ; 
                 if (j==(i-1))Tdiag[i*nvec_lanczos + j] = beta[j];
                 if (j==(i+1))Tdiag[i*nvec_lanczos + j] = gamma[i];
               }
              }    

         for (int i=0 ; i < nvec_lanczos ; i++){
            alpha_ea[uppertriangle][i] = alpha[i] ;
            beta_ea[uppertriangle][i] = beta[i] ;
            gamma_ea[uppertriangle][i] = gamma[i] ;
         }

  /*
   * Diagonalize the tridiagonal matrix to see if that produces EOM-EA values..
   */
            vector<U>  l(nvec_lanczos*nvec_lanczos);
            vector<CU> s_tmp(nvec_lanczos);
            vector<U>  vr_tmp(nvec_lanczos*nvec_lanczos);

            int info = geev('N', 'V', nvec_lanczos, Tdiag.data(), nvec_lanczos,
                        s_tmp.data(), l.data(), nvec_lanczos,
                        vr_tmp.data(), nvec_lanczos);
            if (info != 0) throw runtime_error(str("check diagonalization: Info in geev: %d", info));

            for (int i=0 ; i < nvec_lanczos ; i++){
//            printf("real eigenvalues: %.15f\n", s_tmp[i].real());
//            printf("imaginary eigenvalues: %.15f\n", s_tmp[i].imag());
            }


           if (arena.rank ==0)
           {
            std::ifstream iffile("gomega_ea.dat");
            if (iffile) remove("gomega_ea.dat");
           }

            int omega_counter = 0 ;
            for (auto& o : omegas)
            {

            /*Evaluate continued fraction 
             */

              value  = {0.,0.} ;
              value1 = {0.,0.} ;

              CU alpha_temp ;
              CU beta_temp ;
              CU gamma_temp ;
              CU com_one(1.,0.) ;
              omega = {o.real(),o.imag()} ;

//            this->log(arena) << "Computing Green's function at " << fixed << setprecision(6) << o << endl ;

             for(int i=(nvec_lanczos-1);i >= 0;i--){  
              alpha_temp = {alpha[i],0.} ;
              beta_temp  = {beta[i],0.} ;
              gamma_temp = {gamma[i],0.} ;

              value = (com_one)/(omega - alpha_temp - beta_temp*gamma_temp*value1) ;                 
              value1 = value ;
             }
              if (orb_range == "full") gf_ea[nspin][omega_counter][uppertriangle] = value*norm*norm ;
              if (orb_range == "diagonal") gf_ea[nspin][omega_counter][0] = value*norm*norm ;
              if(orbright==orbleft) spec_func[omega_counter] += value*norm*norm ;
              omega_counter += 1 ;

             }
              uppertriangle +=1 ;
            }
          } 

         if (arena.rank == 0 )
         {
             std::ofstream gomega;
             gomega.open ("gomega_ea.dat", ofstream::out|std::ios::app);

//           for (int i=0 ; i < omegas.size() ; i++){
//             gomega << omegas[i].real() << " " << -1/M_PI*spec_func[i].imag() << std::endl ;
//           }

            int  uppertriangle = 0 ;
            for (int i=0 ; i < (nI+nA) ; i++){
             for (int j=i ; j < (nI+nA) ; j++){
               gomega << i << " " << j << " " << gf_ea[0][0][uppertriangle] << std::endl ;
               uppertriangle += 1 ;
             }
            }
             gomega.close();
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
            const SpinorbitalTensor<U>& WABEJ = H.getABCI();
            const SpinorbitalTensor<U>& WABEF = H.getABCD();

            auto& T = this->template get<ExcitationOperator<U,2>>("T");

            auto& XMI = this->template gettmp<SpinorbitalTensor<U>>("XMI");
            auto& GIM = this->template gettmp<SpinorbitalTensor<U>>("GIM");

            auto& D = this->template gettmp<Denominator<U>>("D");
            auto& lanczos = this->template gettmp<Lanczos<U,X>>("lanczos");

            auto& RL = this->template gettmp< ExcitationOperator<U,2,1>>("RL");
            auto& LL = this->template gettmp< DeexcitationOperator<U,2,1>>("LL");
            auto& Z  = this->template gettmp<  ExcitationOperator<U,2,1>>("Z");
            auto& Y  = this->template gettmp<  DeexcitationOperator<U,2,1>>("Y");
            auto& b  = this->template gettmp<  ExcitationOperator<U,2,1>>("b");
            auto& e  = this->template gettmp<DeexcitationOperator<U,2,1>>("e");

            auto& alpha = this->template gettmp<unique_vector<U>> ("alpha");
            auto& beta  = this->template gettmp<unique_vector<U>> ("beta");
            auto& gamma = this->template gettmp<unique_vector<U>> ("gamma");

                XMI[  "m"] = -0.5*WMNEF["mnef"]*RL(2)["efn"];

                Z(1)[  "a"]  =       FAE[  "ae"]*RL(1)[  "e"];
                Z(1)[  "a"] -=       FME[  "me"]*RL(2)["aem"];
                Z(1)[  "a"] -= 0.5*WAMEF["amef"]*RL(2)["efm"];

                Z(2)["abi"]   =     WABEJ["baei"]*RL(1)[  "e"];
                Z(2)["abi"]  +=       FAE[  "ae"]*RL(2)["ebi"];
                Z(2)["abi"]  -=       FMI[  "mi"]*RL(2)["abm"];
                Z(2)["abi"]  -=       XMI[  "m"]*T(2)["abim"];
                Z(2)["abi"]  += 0.5*WABEF["abef"]*RL(2)["efi"];
                Z(2)["abi"]  -=     WAMEI["amei"]*RL(2)["ebm"];

  /*Left hand matrix-vector product : Q^T Hbar
   *We will use Y array for the left hand residual..   
   */ 
                GIM[ "m"]   =  -0.5*T(2)["efmo"]*LL(2)["oef"];     
                                                              
                Y(1)[ "a"]  =       FAE[  "ea"]*LL(1)[  "e"];
                Y(1)[ "a"] -= 0.5*WABEJ["efam"]*LL(2)["mef"];
                                                              
                Y(2)["iab"]  =       FME[  "ia"]*LL(1)[  "b"];
                Y(2)["iab"] +=     WAMEF["eiba"]*LL(1)[  "e"];
                Y(2)["iab"] +=       FAE[  "ea"]*LL(2)["ieb"];
                Y(2)["iab"] -=       FMI[  "im"]*LL(2)["mab"];
                Y(2)["iab"] += 0.5*WABEF["efab"]*LL(2)["ief"];
                Y(2)["iab"] -=     WAMEI["eibm"]*LL(2)["mae"];
                Y(2)["iab"] -=     WMNEF["miba"]* GIM[  "m"];
 
            lanczos.extrapolate_tridiagonal(RL, LL, Z, Y, D, alpha, beta, gamma);

           this->conv() = max(pow(beta[beta.size()-1],2), pow(gamma[gamma.size()-1],2));
        }
};

}
}

static const char* spec = R"(

orbital ?
int 1,
npoint ?
int 100,
omega_min ?
double -10.0,
omega_max ?
double 10.0,
eta ?
double .001,
grid?
  enum{ real, imaginary },
orbital_range?
  enum{ diagonal, full},
beta?
   double 100.0 , 
convergence?
    double 1e-12,
max_iterations?
    int 150,
conv_type?
    enum { MAXE, RMSE, MAE },
lanczos?
{
    order?
            int 10,
    compaction?
            enum { discrete, continuous },
}

)";

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::CCSDEAGF_LANCZOS);
REGISTER_TASK(aquarius::cc::CCSDEAGF_LANCZOS<double>, "ccsdeagf_lanczos",spec);
