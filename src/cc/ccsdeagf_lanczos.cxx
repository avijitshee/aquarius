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
        int nr_impurities; 
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
            reqs.emplace_back("ccsd.ipgflanczos", "gf_ip");
            reqs.emplace_back("occspace", "occ");
            reqs.emplace_back("vrtspace", "vrt");
            reqs.emplace_back("Ea", "Ea");
            reqs.emplace_back("Eb", "Eb");
            this->addProduct(Product("ccsd.eagflanczos", "gf_ea", reqs));

            orbital = config.get<int>("orbital");
            double from = config.get<double>("omega_min");
            double to = config.get<double>("omega_max");
            int n = config.get<int>("npoint");
            nr_impurities = config.get<int>("impurities"); 
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
            int nvec_lanczos; 
            CU value ;
            CU value1 ;

            int maxspin = (nI == ni) ? 1 : 2 ;

            auto& T = this->template get<ExcitationOperator  <U,2>>("T");
            auto& L = this->template get<DeexcitationOperator<U,2>>("L");
            auto& gf_ip = this->template get<vector<vector<vector<vector<CU>>>>>("gf_ip");
            const auto& occ_hf = this->template get<MOSpace<U>>("occ");
            const auto& vrt_hf = this->template get<MOSpace<U>>("vrt");

            auto& Ea = this->template get<vector<vector<real_type_t<U>>>>("Ea");
            auto& Eb = this->template get<vector<vector<real_type_t<U>>>>("Eb");

            auto& gf_ea = this-> put("gf_ea", new vector<vector<vector<vector<CU>>>>) ;

            SpinorbitalTensor<U> Dab("D(ab)", arena, group, {vrt,occ}, {1,0}, {1,0});
            SpinorbitalTensor<U> Gieab("G(am,ef)", arena, group, {vrt,occ}, {1,1}, {2,0});

            SpinorbitalTensor<U> Nij("N(ij)", arena, group, {vrt,occ}, {0,1}, {0,1});

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

          vector<CU> spec_func(omegas.size()) ;

    /* remove all self-energy and total Green's function files those are already there in the directory..
     */  

       for (int nspin = 0; nspin < maxspin ; nspin++)   
       { 
        for (int i = 0;i < nr_impurities;i++)
        { 
          std::stringstream stream;
          stream << "gomega_"<<nspin<<"_"<<i<< ".txt";
          std::string fileName = stream.str();
          std::ifstream gfile(fileName.c_str());
          if (gfile) remove(fileName.c_str());
        } 
       }

       for (int nspin = 0; nspin < maxspin ; nspin++)   
       { 
        for (int i = 0;i < nr_impurities;i++)
        { 
          std::stringstream stream;
          stream << "self_energy_"<<nspin<<"_"<<i<< ".txt";
          std::string fileName = stream.str();
          std::ifstream sigmafile(fileName.c_str());
          if (sigmafile) remove(fileName.c_str());
        } 
       }

     /* start calculating all GF elements..
      */  

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
              RL /= norm;
              LL /= norm;

              printf("print norm: %10f\n", norm);

              Iterative<U>::run(dag, arena);

              nvec_lanczos = alpha.size() ; 

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

            U piinverse = 1.0/M_PI ;

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

//              this->log(arena) << "Computing Green's function at " << fixed << setprecision(6) << o << endl ;

             for(int i=(nvec_lanczos-1);i >= 0;i--){  
              alpha_temp = {alpha[i],0.} ;
              beta_temp  = {beta[i],0.} ;
              gamma_temp = {gamma[i],0.} ;

              value = (com_one)/(omega - alpha_temp - beta_temp*gamma_temp*value1) ;                 
              value1 = value ;
             }

             if (orbright == orbleft) spec_func[omega_counter] += value*norm*norm;
             if (orb_range == "full") gf_ea[nspin][omega_counter][orbleft][orbright] = value*norm*norm ;
             if (orb_range == "diagonal") gf_ea[nspin][omega_counter][0][0] = value*norm*norm ;
  //         std::ofstream gomega;
////           gomega.open ("gomega.dat", std::ofstream::out);
  //         gomega.open ("gomega.dat", ofstream::out|std::ios::app);
////           gomega << o.real() << " " << -piinverse*value.imag()*norm*norm << std::endl ; 
  //          gomega << o.real() << " " << piinverse*value.imag()*norm*norm << std::endl ; 
  //         gomega.close();

//              printf("real value : %.15f\n", value.real()*norm*norm);
//              printf("imaginary value : %.15f\n", value.imag()*norm*norm);
              omega_counter += 1 ;

             }
            }
           }
          } 

            U piinverse = 1/M_PI ;
            std::ofstream gomega;
            gomega.open ("gomega_ea.dat", ofstream::out|std::ios::app);

            for (int i=0 ; i < omegas.size() ; i++){
//               gomega << omegas[i].real() << " " <<(-1/M_PI)*spec_func[i].imag() << std::endl ; 
            }

            gomega.close();

         const SymmetryBlockedTensor<U>& cA_ = vrt_hf.Calpha;
         const SymmetryBlockedTensor<U>& ca_ = vrt_hf.Cbeta;
         const SymmetryBlockedTensor<U>& cI_ = occ_hf.Calpha;
         const SymmetryBlockedTensor<U>& ci_ = occ_hf.Cbeta;

         int nirreps = 1;

         vector<vector<U>> cA(nirreps), ca(nirreps), cI(nirreps), ci(nirreps);

         const vector<int>& N = occ_hf.nao;
 
         int norb = N[0]; 

         vector<U> c_full ; 
         vector<U> fock;

       /* Reading one electron integrals here. It will be nicer to read them separately within a function.. 
        */   

          std::ifstream one_diag("one_diag.txt");
          std::istream_iterator<U> start(one_diag), end;
          vector<U> diagonal(start, end);
          std::copy(diagonal.begin(), diagonal.end(),std::back_inserter(integral_diagonal)); 

          std::ifstream onsite("onsite.txt");
          std::istream_iterator<U> start_v(onsite), end_v;
          vector<U> int_onsite(start_v, end_v);
          std::copy(int_onsite.begin(), int_onsite.end(), std::back_inserter(v_onsite)) ;

   for (int nspin = 0;nspin < maxspin;nspin++)
   {
    for (int i = 0;i < nirreps;i++)
     {
       vector<int> irreps = {i,i};
       (nspin == 0) ? cA_.getAllData(irreps, cA[i]) : ca_.getAllData(irreps, ca[i]);
       assert(cA[i].size() == N[i]*nA[i]);
       assert(ca[i].size() == N[i]*na[i]);
       (nspin == 0) ? cI_.getAllData(irreps, cI[i]) : ci_.getAllData(irreps, ci[i]);
       assert(cI[i].size() == N[i]*nI[i]);
       assert(ci[i].size() == N[i]*ni[i]);
     }        

     c_full.clear() ;  
    for (int i = 0;i < nirreps;i++)
    {
      (nspin == 0) ? c_full.insert(c_full.begin(),cI[i].begin(),cI[i].end()) : c_full.insert(c_full.begin(),ci[i].begin(),ci[i].end());
      (nspin == 0) ? c_full.insert(c_full.end(),cA[i].begin(),cA[i].end()) : c_full.insert(c_full.end(),ca[i].begin(),ca[i].end());
    }

      fock.clear() ;

    for (int i = 0;i < nirreps;i++)
    {
       (nspin == 0) ? fock.assign(Ea[i].begin(), Ea[i].end()) : fock.assign(Eb[i].begin(), Eb[i].end());
    }

        U new_value ;

     for (int p = 0; p < norb ;p++)
     {
//      for (int i = 0; i < 1; i++) 
//      {
//           new_value = new_value + c_full[p*norb+i]*c_full[p*norb+i]*fock[p] ; 
//           printf("value of MO coefficients %10d %10f\n",p, c_full[p*norb+(orbital-1)]);
//        }
       }

//         printf("value of AO fock  %10f\n",new_value);

    vector<CU> self_energy_ao(omegas.size(),{0.,0.}) ;

     std::ifstream iffile("g_0_omega.dat");
     if (iffile) remove("g_0_omega.dat");

    for (int omega = 0; omega < omegas.size() ;omega++)
    {
      vector<CU> gf_ao(nr_impurities*nr_impurities,{0.,0.}) ;
      vector<int> ipiv(nr_impurities) ;
      vector<CU> gf_tmp(norb*norb) ;
      vector<CU> gf_zero_inv(norb*norb) ;
      vector<CU> G_inv(nr_impurities*nr_impurities) ;

      for (int p = 0; p < norb ;p++)
      {
        for (int q = 0; q < norb ;q++)
        {
          if (q == p)
          {
            gf_zero_inv[p*norb+q] = omegas[omega] - fock[q] ;
            gf_tmp[p*norb+q] = gf_ip[nspin][omega][p][q] + gf_ea[nspin][omega][p][q] ; 
          }
          if (q != p )
          { 
            gf_zero_inv[p*norb+q] =  {0.,0.} ;
            gf_tmp[p*norb+q]  = 0.5*( gf_ip[nspin][omega][p][q] - gf_ip[nspin][omega][p][p] - gf_ip[nspin][omega][q][q]) ; 
            gf_tmp[p*norb+q] += 0.5*( gf_ea[nspin][omega][p][q] - gf_ea[nspin][omega][p][p] - gf_ea[nspin][omega][q][q]) ; 
          }
//           printf("Real and Imaginary value of the Green's function %10f %15f %15f\n",omegas[omega].imag(), gf_tmp[0].real(), gf_tmp[0].imag());

          for (int i = 0; i < nr_impurities; i++) 
          {
           for (int j = 0; j < nr_impurities; j++) 
           {
            gf_ao[i*nr_impurities + j] = gf_ao[i*nr_impurities + j] + c_full[p*norb+i]*c_full[q*norb+j]*gf_tmp[p*norb+q] ; 

            if (omega == 0) cout << omegas[omega].imag() << " " << i << " " << j << " " << gf_ao[i*nr_impurities+j].real() << " " << gf_ao[i*nr_impurities+j].imag() << std::endl ; 
           }  
          }
        }
      }

//       for (int i = 0; i < nr_impurities; i++) 
//        {
//         for (int j = 0; j < nr_impurities; j++) 
//         {
//           if (omega == 0) cout << omegas[omega].imag() << " " << i << " " << j << " " << gf_ao[i*nr_impurities+j].real() << " " << gf_ao[i*nr_impurities+j].imag() << std::endl ; 
//         }  
//        }

  /* calculate bare Green's function..
   */  

    std::ofstream gzero_omega;

    for (int u = 0; u < nr_impurities; u++)
     {
     for (int v = 0; v < nr_impurities; v++)
      {
       CU Hybridization{0.,0.} ;
       for (int b = nr_impurities; b < norb; b++)
       {
        Hybridization += (integral_diagonal[u*norb+b]*integral_diagonal[v*norb+b])/(omegas[omega] - integral_diagonal[b*norb+b]) ;
       }
        if (v == u) 
        {
         G_inv[u*nr_impurities+v] = omegas[omega] - integral_diagonal[u*norb+v] -  Hybridization + 0.5*v_onsite[u]  ;
        }
        else
        {
         G_inv[u*nr_impurities+v] = - integral_diagonal[u*norb+v] -  Hybridization  ;
        } 

//         gzero_omega << omegas[omega].imag() << " " << u << " " << v << " " << G_inv[u*nr_impurities+v].real() << " " << G_inv[u*nr_impurities+v].imag() << std::endl ; 
      }
     }
       gzero_omega.close();

   /* Dyson Equation  
    */
       for (int b = 0 ; b < nr_impurities; b++)
       {
         std::stringstream stream;
          stream << "gomega_"<<nspin<<"_"<<b<< ".txt";
          std::string fileName = stream.str();
         std::ofstream gomega;
          gomega.open (fileName.c_str(), ofstream::out|std::ios::app);
          gomega << nspin << " " << b << " " << omegas[omega].imag() << " " << gf_ao[b*nr_impurities+b].real() << " " << gf_ao[b*nr_impurities+b].imag() << std::endl ; 
         gomega.close();
       }
         getrf(nr_impurities, nr_impurities, gf_ao.data(), nr_impurities, ipiv.data() ) ;

         getri(nr_impurities, gf_ao.data(), nr_impurities, ipiv.data()) ;

         axpy (nr_impurities*nr_impurities, -1.0, gf_ao.data(), 1, G_inv.data(), 1);

      for (int b = 0 ; b < nr_impurities; b++)
       {
         std::stringstream stream;
          stream << "self_energy_"<<nspin<<"_"<<b<< ".txt";
          std::string fileName = stream.str();
         std::ofstream gomega1;
          gomega1.open (fileName.c_str(), ofstream::out|std::ios::app);
          gomega1 << nspin << " " << b << " " << omegas[omega].imag() << " " << G_inv[b*nr_impurities+b].real() << " " << G_inv[b*nr_impurities+b].imag() << std::endl ; 
         gomega1.close();
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
            const SpinorbitalTensor<U>& WABEJ = H.getABCI();
            const SpinorbitalTensor<U>& WABEF = H.getABCD();

            auto& T = this->template get<ExcitationOperator<U,2>>("T");

            auto& XMI = this->template gettmp<SpinorbitalTensor<U>>("XMI");
            auto& GIM = this->template gettmp<SpinorbitalTensor<U>>("GIM");

            auto& D = this->template gettmp<Denominator<U>>("D");
//          auto& lanczos = this->template gettmp<Lanczos<unique_vector<U>>>("lanczos");
//          auto& lanczos = this->template gettmp<Lanczos<ExcitationOperator<U,2,1>>>("lanczos");
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

            int nvec_lanczos; 
            U value ;
            U delta_value ;
            U value1 ;
            U alpha_temp ;
            U beta_temp ;
            U gamma_temp ;

            printf("<RL|RL>: %.15f\n", scalar(RL*RL));
            printf("<LL1|LL1>: %.15f\n", scalar(LL(1)["m"]*LL(1)["m"]));
            printf("<LL|LL>: %.15f\n", scalar(LL*LL));
            printf("<LL|RL>: %.15f\n", scalar(RL*LL));
            //printf("<Rr|Ri>: %.15f\n", scalar(Rr*Ri));

            //printf("<B|Rr>: %.15f\n", scalar(b*Rr));
            //printf("<B|Ri>: %.15f\n", scalar(b*Ri));

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
 
            printf("<Z1|Z1>: %.15f\n", scalar(Z(1)*Z(1)));
            printf("<Z2|Z2>: %.15f\n", 0.5*scalar(Z(2)*Z(2)));
            printf("<Z|Z>: %.15f\n",   scalar(Z*Z));
            printf("<Y1|Y1>: %.15f\n", scalar(Y(1)*Y(1)));
            printf("<Y2|Y2>: %.15f\n", 0.5* scalar(Y(2)*Y(2)));
            printf("<Y|Y>: %.15f\n", scalar(Y*Y));
            //printf("<Z2|Z2>: %.15f\n", 0.5*scalar(Z(2)*Z(2)));
            //printf("<Z|Z>: %.15f\n", scalar(Z*Z));
            //printf("<Zi|Zi>: %.15f\n", scalar(Zi*Zi));

            //printf("<Ur|Ur>: %.15f\n", scalar(Z*Z));
            //printf("<Ui|Ui>: %.15f\n", scalar(Zi*Zi));
            
            lanczos.extrapolate_tridiagonal(RL, LL, Z, Y, D, alpha, beta, gamma);

              value  = 1. ;
              value1 = 0. ;
              nvec_lanczos = alpha.size() ; 

            if (nvec_lanczos > 2) {
             for(int i=(nvec_lanczos-1);i >= 0;i--){  
              alpha_temp = alpha[i] ;
              beta_temp  = beta[i] ;
              gamma_temp = gamma[i] ;
              value = 1.0/(alpha_temp + beta_temp*gamma_temp*value1) ;                 
              value1 = value ;
             }
            }
             
              old_value.push_back(value) ;

              if (nvec_lanczos <= 2) {
                delta_value = 1.0 ;}
              else{
                delta_value = old_value[nvec_lanczos-2] - value ;
              }

              printf("have passed this step 1: %.10f\n", beta[beta.size()-1]);
              printf("old_value: %.10f\n", old_value[nvec_lanczos-1]);
              printf("new value: %.10f\n", value);

//            this->conv() = max(pow(beta[beta.size()-1],2), pow(gamma[gamma.size()-1],2));
           this->conv() = max(pow(beta[beta.size()-1],2), pow(gamma[gamma.size()-1],2));
//             this->conv() = aquarius::abs(delta_value) ;

//            lanczos.getSolution(alpha, beta, gamma);
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
impurities int,
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
