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
//        typedef typename T::dtype U;
        typedef T U;
        typedef complex_type_t<U> CU;
        unique_vector<T> old_c_r;
        unique_vector<T> old_c_l;
        unique_vector<T> old_hc_r;
        unique_vector<T> old_hc_l;
        vector<T> new_c;
        vector<T> new_hc;
//        unique_ptr<ExcitationOperator<T,1,2>> rhs;
//        unique_vector<ExcitationOperator<T,1,2>> rhs;
        vector<CU> b;
        marray<CU,2> e;
        int maxextrap, nextrap, nvec;
        vector<CU> v;
        bool continuous;

        void parse(const input::Config& config)
        {
            nextrap = 0;
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
            vector<U> temp1;
            vector<U> temp2;
            vector<U> temp3;


            temp1.clear();
            temp2.clear(); 
            temp3.clear(); 

            c_r(1)({0,0},{0,1})({0}).getAllData(temp1);
            c_r(2)({0,0},{0,1})({0,0,0}).getAllData(temp2);
            c_r(2)({1,0},{0,2})({0,0,0}).getAllData(temp3);

            old_c_r.clear() ;

           for (int ii = 0 ; ii< temp1.size(); ii++){
             old_c_r.emplace_back(temp1[ii]);
           }
            for (int ii = 0 ; ii< temp2.size(); ii++){
             old_c_r.emplace_back(temp2[ii]);
           }

            for (int ii = 0 ; ii< temp3.size(); ii++){
             old_c_r.emplace_back(temp3[ii]);
           }

            temp1.clear();
            temp2.clear(); 
            temp3.clear(); 

             c_l(1)({0,1},{0,0})({0}).getAllData(temp1);
             c_l(2)({0,1},{0,0})({0}).getAllData(temp2);
             c_l(2)({0,2},{1,0})({0}).getAllData(temp3);

            old_c_l.clear() ;

            for (int ii = 0 ; ii< temp1.size(); ii++){
             old_c_l.emplace_back(temp1[ii]);
           }
            for (int ii = 0 ; ii< temp2.size(); ii++){
             old_c_l.emplace_back(temp2[ii]);
           }

            for (int ii = 0 ; ii< temp3.size(); ii++){
             old_c_l.emplace_back(temp3[ii]);
           }
        }

        void getRoot(T& c_r, T& c_i, T& hc_r, T& hc_i)
        {
            getRoot(c_r, c_i);

            hc_r = 0;
            hc_i = 0;
            for (int extrap = 0;extrap < nextrap;extrap++)
            {
                hc_r += old_hc_r[extrap]*v[extrap].real();
                hc_r -= old_hc_l[extrap]*v[extrap].imag();
                hc_i += old_hc_r[extrap]*v[extrap].imag();
                hc_i += old_hc_l[extrap]*v[extrap].real();
            }
        }

        void getRoot(T& c_r, T& c_i)
        {
            c_r = 0;
            c_i = 0;
            for (int extrap = 0;extrap < nextrap;extrap++)
            {
                c_r += old_c_r[extrap]*v[extrap].real();
                c_r -= old_c_l[extrap]*v[extrap].imag();
                c_i += old_c_r[extrap]*v[extrap].imag();
                c_i += old_c_l[extrap]*v[extrap].real();
            }
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

             vector<U> temp1;
             vector<U> temp2;
             vector<U> temp3;

             temp1.clear() ;
             temp2.clear() ;
             temp3.clear() ;

            double sum1 = 0.0 ;
             new_c.clear() ; 
             new_hc.clear() ; 
             hc_r(1)({0,0},{0,1})({0}).getAllData(temp1);
             hc_r(2)({0,0},{0,1})({0,0,0}).getAllData(temp2);
             hc_r(2)({1,0},{0,2})({0,0,0}).getAllData(temp3);

            for (int ii = 0 ; ii< temp1.size(); ii++){
             new_hc.emplace_back(temp1[ii]);
           }
            for (int ii = 0 ; ii< temp2.size(); ii++){
             new_hc.emplace_back(temp2[ii]);
           }

            for (int ii = 0 ; ii< temp3.size(); ii++){
             new_hc.emplace_back(temp3[ii]);
           }
             sum1 = c_ddot (temp2.size(), &temp2[0], 1, &temp2[0], 1) ; 
//             sum1 += c_ddot (temp3.size(), &temp3[0], 1, &temp3[0], 1) ; 

             printf("print explicit sum of hc_r : %.10f \n",sum1);

             int vecsize= temp1.size() + temp2.size() + temp3.size() ; 

             printf("print vecsize : %10d \n",temp2.size());
             printf("print vecsize : %10d \n",nvec);

             temp1.clear();
             temp2.clear(); 
             temp3.clear(); 

             c_r(1)({0,0},{0,1})({0}).getAllData(temp1);
             c_r(2)({0,0},{0,1})({0,0,0}).getAllData(temp2);
             c_r(2)({1,0},{0,2})({0,0,0}).getAllData(temp3);

            for (int ii = 0 ; ii< temp1.size(); ii++){
             new_c.emplace_back(temp1[ii]);
           }
            for (int ii = 0 ; ii< temp2.size(); ii++){
             new_c.emplace_back(temp2[ii]);
           }

            for (int ii = 0 ; ii< temp3.size(); ii++){
             new_c.emplace_back(temp2[ii]);
           }

             double temp = alpha[nextrap] ;

             printf("print <Z|Z>: %.10f \n",scalar(hc_r(2)*hc_r(2)));

             printf("print nextrap: %10d \n",temp1.size());
             printf("print nextrap: %10d \n",temp2.size());
             printf("print nextrap: %10d \n",temp3.size());

            vector<double> r(vecsize); 
            vector<double> s(vecsize); 

            /* calculate r = hc_r - gamma(i-1)q(i-1) - alpha(i)q(i)
             */

           if (nextrap > 0) {

           for (int vec = 0; vec < vecsize; vec++) {
              r[vec] = new_hc[vec] - temp*new_c[vec] - gamma[nextrap-1]*old_c_r[vec]  ; 
            }
           }else{
            for (int vec = 0; vec < vecsize; vec++) {
              r[vec] = new_hc[vec] - temp*new_c[vec] ; 
            }
          }

            /* calculate s = hc_l - beta(i-1)pT(i-1) - alpha(i)pT(i)
             */

            new_c.clear() ; 

            temp1.clear();
            temp2.clear(); 
            temp3.clear(); 

             c_l(1)({0,1},{0,0})({0}).getAllData(temp1);
             c_l(2)({0,1},{0,0})({0,0,0}).getAllData(temp2);
             c_l(2)({0,2},{1,0})({0,0,0}).getAllData(temp3);


            for (int ii = 0 ; ii< temp1.size(); ii++){
             new_c.emplace_back(temp1[ii]);
           }
            for (int ii = 0 ; ii< temp2.size(); ii++){
             new_c.emplace_back(temp2[ii]);
           }
            for (int ii = 0 ; ii< temp3.size(); ii++){
             new_c.emplace_back(temp3[ii]);
           }

            new_hc.clear() ; 

            temp1.clear();
            temp2.clear(); 
            temp3.clear(); 

             hc_l(1)({0,1},{0,0})({0}).getAllData(temp1);
             hc_l(2)({0,1},{0,0})({0}).getAllData(temp2);
             hc_l(2)({0,2},{1,0})({0,0,0}).getAllData(temp3);

            for (int ii = 0 ; ii< temp1.size(); ii++){
             new_hc.emplace_back(temp1[ii]);
           }
            for (int ii = 0 ; ii< temp2.size(); ii++){
             new_hc.emplace_back(temp2[ii]);
           }
            for (int ii = 0 ; ii< temp3.size(); ii++){
             new_hc.emplace_back(temp3[ii]);
           }


           if (nextrap > 0) {
              for (int vec = 0; vec < vecsize; vec++) {
                s[vec] = new_hc[vec] - temp*new_c[vec] - beta[nextrap-1] * old_c_l[vec] ; 
              }
           } else{
              for (int vec = 0; vec < vecsize; vec++) {
                s[vec] = new_hc[vec] - temp*new_c[vec] ; 
              }
            }

             temp = c_ddot (vecsize, s.data(), 1, r.data(), 1) ; 

             beta.push_back (sqrt(aquarius::abs(temp))) ;
             gamma.push_back (temp/beta[nextrap])  ; 

              addVectors(c_r, c_l);

              vector<tkv_pair<double>> pairs;
              for (int vec = 0; vec < temp1.size(); vec++) {
                pairs.push_back(tkv_pair<double>(vec, r[vec]/beta[nextrap-1])) ;
              }

              c_r(1)({0,0},{0,1})({0}).writeRemoteData(pairs);

              pairs.clear() ;

              for (int vec = temp1.size(); vec < (temp1.size()+temp2.size()); vec++) {
                pairs.push_back(tkv_pair<double>(vec-temp1.size(), r[vec]/beta[nextrap-1])) ;
              }

              c_r(2)({0,0},{0,1})({0}).writeRemoteData(pairs);

              pairs.clear() ;

              for (int vec = temp1.size()+temp2.size(); vec < (temp1.size()+temp2.size()+temp3.size()) ; vec++) {
                pairs.push_back(tkv_pair<double>(vec-temp1.size()-temp2.size(), r[vec]/beta[nextrap-1])) ;
              }

              c_r(2)({1,0},{0,2})({0,0,0}).writeRemoteData(pairs);

              pairs.clear() ;

              for (int vec = 0; vec < temp1.size(); vec++) {
                pairs.push_back(tkv_pair<double>(vec, s[vec]/gamma[nextrap-1])) ;
              }

              c_l(1)({0,1},{0,0})({0}).writeRemoteData(pairs);
        
              pairs.clear() ;

              for (int vec = temp1.size(); vec < (temp1.size()+temp2.size()); vec++) {
                pairs.push_back(tkv_pair<double>(vec-temp1.size(), s[vec]/gamma[nextrap-1])) ;
              }

              c_l(2)({0,1},{0,0})({0}).writeRemoteData(pairs);

              pairs.clear() ;

              for (int vec = (temp1.size()+temp2.size()); vec < (temp1.size()+temp2.size()+temp3.size()); vec++) {
                pairs.push_back(tkv_pair<double>(vec-temp1.size()-temp2.size(), s[vec]/gamma[nextrap-1])) ;
              }

               c_l(2)({0,2},{1,0})({0,0,0}).writeRemoteData(pairs);


            printf("scalar product c_l: %.10f \n",scalar(c_l*c_l));
            printf("scalar product c_r: %.10f \n",scalar(c_r*c_r));
            printf("print nextrap: %10d \n",nextrap);
            printf("print alpha: %.10f \n",alpha[nextrap-1]);
            printf("print beta: %.10f \n",beta[nextrap-1]);
            printf("print gamma: %.10f \n",gamma[nextrap-1]);
        }

        void getSolution(T& c_r, T& c_l)
        {
            if (continuous)
            {
                c_r = old_c_r[nextrap-1];
                c_l = old_c_l[nextrap-1];
            }
            else
            {
                getRoot(c_r, c_l);
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
