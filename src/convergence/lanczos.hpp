#ifndef _AQUARIUS_LANCZOS_HPP_
#define _AQUARIUS_LANCZOS_HPP_

#include "util/global.hpp"
#include <iterator>
#include <vector>

#include "tensor/composite_tensor.hpp"
#include "input/config.hpp"
#include "task/task.hpp"
#include "cc/complex_denominator.hpp"
#include "operator/excitationoperator.hpp"
#include "operator/mooperator.hpp"
#include "operator/deexcitationoperator.hpp"
#include "davidson.hpp"
#include "diis.hpp"
using namespace aquarius::op;
using namespace aquarius::task;
using namespace aquarius::input;
using namespace aquarius::tensor;
namespace aquarius
{
namespace convergence
{

template<typename T, typename V>
class Lanczos : public task::Destructible
{
    private:
        Lanczos(const Lanczos& other);

        Lanczos& operator=(const Lanczos& other);

    protected:
        typedef T U;
        typedef complex_type_t<U> CU;
        unique_vector<ExcitationOperator<T,1,2>> old_c_r;
        unique_vector<DeexcitationOperator<V,1,2>> old_c_l;
        unique_vector<ExcitationOperator<T,2,1>> old_c_r_particle;
        unique_vector<DeexcitationOperator<V,2,1>> old_c_l_particle;
        vector<CU> b;
        marray<CU,2> e;
        int maxextrap, nextrap, nvec;
        vector<CU> v;
        bool continuous;

        void parse(const input::Config& config)
        {
            maxextrap = config.get<int>("order");
            continuous = config.get<string>("compaction") == "continuous";
        }

        void init(int nvec_)
        {
            nextrap = 0;
            nvec = nvec_;
            b.resize(nextrap);
            e.resize(nextrap, nextrap);
            v.resize(nextrap, nextrap);
        }

        void addVectors(const op::ExcitationOperator  <T,1,2>& c_r, const op::DeexcitationOperator  <V,1,2>& c_l)
        {
            nextrap++;

            old_c_r.clear();
            old_c_l.clear();

            old_c_r.emplace_back(c_r);
            old_c_l.emplace_back(c_l);

        }

        void addVectors(const op::ExcitationOperator  <T,2,1>& c_r, const op::DeexcitationOperator  <V,2,1>& c_l)
        {
            nextrap++;

            old_c_r_particle.clear();
            old_c_l_particle.clear();

            old_c_r_particle.emplace_back(c_r);
            old_c_l_particle.emplace_back(c_l);

        }

    public:
        Lanczos(const input::Config& config, const int nvec)
        {
            parse(config);
            reset(nvec);
        }

        void extrapolate_tridiagonal(op::ExcitationOperator  <T,1,2>& c_r,op::DeexcitationOperator<V,1,2>& c_l,op::ExcitationOperator  <T,1,2>& hc_r, op::DeexcitationOperator<V,1,2>& hc_l, const op::Denominator<T>& D, unique_vector<T>& alpha, unique_vector<T>& beta, unique_vector<T>& gamma)
///     template <typename c_container, typename cl_container>
///     enable_if_t<is_same<typename decay_t<c_container>::value_type, T>::value && is_same<typename decay_t<cl_container>::value_type, V>::value, vector<U>>
///     extrapolate_tridiagonal(c_container&& c_r, cl_container&& c_l, const c_container&& hc_r, const cl_container&& hc_l, const op::Denominator<T>& D, unique_vector<T>& alpha, unique_vector<T>& beta, unique_vector<T>& gamma)
        {
            using slice::all;

            /* calculate alpha(i) = pT(i)hc_rq(i)
             */ 

             alpha.push_back(scalar(c_l*hc_r)) ;             

            /*copy array to a vector 
             */

            /*do some printing
             */   
            
//           vector<U> temp1;
//           vector<U> temp2;
//           vector<U> temp3;

//           temp1.clear() ;
//           temp2.clear() ;
//           temp3.clear() ;

//           hc_r(1)({0,0},{0,1})({0}).getAllData(temp1);
//           hc_r(2)({0,0},{0,1})({0,0,0}).getAllData(temp2);
//           hc_r(2)({1,0},{0,2})({0,0,0}).getAllData(temp3);

//          for (int ii = 0 ; ii< temp1.size(); ii++){
//           printf("print values of c_r(1) : %.10f \n",temp1[ii]);
//         }
//          for (int ii = 0 ; ii< temp2.size(); ii++){
//           printf("print values of c_r(2) : %.10f \n",temp2[ii]);
//         }
//          for (int ii = 0 ; ii< temp3.size(); ii++){
//           printf("print values of c_r(2) : %.10f \n",temp3[ii]);
//         }

             double temp ;

             unique_vector<ExcitationOperator<T,1,2>> r;
             unique_vector<DeexcitationOperator<V,1,2>> s;

//            vector<double> r(vecsize); 
//            vector<double> s(vecsize); 

            /* calculate r = hc_r - gamma(i-1)q(i-1) - alpha(i)q(i)
             */

             r.clear() ;
             s.clear() ;

           if (nextrap > 0) {
             r.emplace_back(hc_r) ;
             r[0] -= alpha[nextrap]*c_r ;
             r[0] -= gamma[nextrap-1]*old_c_r[0] ;
           }else{
             r.emplace_back(hc_r) ;
             r[0] -= alpha[nextrap]*c_r ;
           }

//             if (nextrap > 0) printf(" old c_r: %.10f \n",scalar(old_c_r[0]*old_c_r[0]));

            /* calculate s = hc_l - beta(i-1)pT(i-1) - alpha(i)pT(i)
             */

            if (nextrap > 0) {
             s.emplace_back(hc_l) ;
             s[0] -= alpha[nextrap]*c_l ;
             s[0] -= beta[nextrap-1]*old_c_l[0] ;
           }else{
             s.emplace_back(hc_l) ;
             s[0] -= alpha[nextrap]*c_l ;
          }

             temp = scalar(r[0]*s[0]) ; 

             beta.emplace_back (sqrt(aquarius::abs(scalar(r[0]*s[0])))) ;

             if (beta[nextrap] > 1.0e-10)
             {
              gamma.emplace_back (temp/beta[nextrap])  ; 
             }else
             {
              gamma.emplace_back(0.) ; 
             }

             addVectors(c_r, c_l);

             if (beta[nextrap-1] > 1.0e-10)
             {
              c_l = s[0]/gamma[nextrap-1] ; 
              c_r = r[0]/beta[nextrap-1] ; 
             }

//          printf("test orthogonality: %.10f \n",scalar(c_r*old_c_l[0]));
//          printf("test orthogonality: %.10f \n",scalar(c_l*old_c_r[0]));
//          printf("test normalization: %.10f \n",scalar(c_r*c_l));

//          printf("print alpha: %.10f \n",alpha[nextrap-1]);
//          printf("print beta: %.10f \n",beta[nextrap-1]);
//          printf("print gamma: %.10f \n",gamma[nextrap-1]);
        }

        void extrapolate_tridiagonal(op::ExcitationOperator  <T,2,1>& c_r,op::DeexcitationOperator<V,2,1>& c_l,op::ExcitationOperator  <T,2,1>& hc_r, op::DeexcitationOperator<V,2,1>& hc_l, const op::Denominator<T>& D, unique_vector<T>& alpha, unique_vector<T>& beta, unique_vector<T>& gamma)
        {
            using slice::all;

            /* calculate alpha(i) = pT(i)hc_rq(i)
             */ 

             alpha.push_back(scalar(hc_r*c_l)) ;             

             double temp ;

             unique_vector<ExcitationOperator<T,2,1>> r;
             unique_vector<DeexcitationOperator<V,2,1>> s;

            /* calculate r = hc_r - gamma(i-1)q(i-1) - alpha(i)q(i)
             */

             r.clear() ;
             s.clear() ;

           if (nextrap > 0) {
             r.emplace_back(hc_r) ;
             r[0] -= alpha[nextrap]*c_r ;
             r[0] -= gamma[nextrap-1]*old_c_r_particle[0] ;
           }else{
             r.emplace_back(hc_r) ;
             r[0] -= alpha[nextrap]*c_r ;
           }

            /* calculate s = hc_l - beta(i-1)pT(i-1) - alpha(i)pT(i)
             */

            if (nextrap > 0) {
             s.emplace_back(hc_l) ;
             s[0] -= alpha[nextrap]*c_l ;
             s[0] -= beta[nextrap-1]*old_c_l_particle[0] ;
           }else{
             s.emplace_back(hc_l) ;
             s[0] -= alpha[nextrap]*c_l ;
          }

             temp = scalar(r[0]*s[0]) ; 

             beta.emplace_back (sqrt(aquarius::abs(scalar(r[0]*s[0])))) ;
             if (beta[nextrap] > 1.0e-10)
             {
              gamma.emplace_back (temp/beta[nextrap])  ; 
             }else
             {
              gamma.emplace_back(0.) ; 
             }

             addVectors(c_r, c_l);

             if (beta[nextrap-1] > 1.0e-10)
             {
              c_l = s[0]/gamma[nextrap-1] ; 
              c_r = r[0]/beta[nextrap-1] ; 
             } 
 
        }

        void reset(int nvec = 1)
        {
            init(nvec);
        }
};

}
}

#endif
