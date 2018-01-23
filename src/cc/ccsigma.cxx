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

    public:
        CCSDSIGMA(const string& name, Config& config)
        : Iterative<CU>(name, config), krylov_config(config.get("krylov"))
        {
            vector<Requirement> reqs;
            reqs.emplace_back("ccsd.T", "T");
            reqs.emplace_back("ccsd.L", "L");
            reqs.emplace_back("ccsd.Hbar", "Hbar");
            this->addProduct("ccsd.ipgf", "gf", reqs);

            orbital = config.get<int>("orbital");
            double from = config.get<double>("omega_min");
            double to = config.get<double>("omega_max");
            int n = config.get<double>("npoint");
            double eta = config.get<double>("eta");

            double delta = (to-from)/max(1,n-1);
            for (int i = 0;i < n;i++)
            {
                omegas.emplace_back(from+delta*i, eta);
            }
        }

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::CCSDSIGMA);
REGISTER_TASK(aquarius::cc::CCSDSIGMA<double>, "ccsdsigma",spec);
