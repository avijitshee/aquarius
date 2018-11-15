#include "util/global.hpp"

#include "task/task.hpp"
#include "operator/2eoperator.hpp"
#include "operator/space.hpp"
#include "scf/uhf.hpp"
#include "operator/excitationoperator.hpp"

using namespace aquarius::tensor;
using namespace aquarius::task;
using namespace aquarius::input;
using namespace aquarius::op;
using namespace aquarius::symmetry;

namespace aquarius
{
namespace cc
{

template <typename U>
class Assemble_Files: public Task 
{
    protected:
        typedef complex_type_t<U> CU;

        string path_alpha_ip, path_beta_ip, path_gamma_ip, path_norm_ip ;
        string path_alpha_ea, path_beta_ea, path_gamma_ea, path_norm_ea ; 
        int nlanczos ;

    public:
        Assemble_Files(const string& name, Config& config): Task(name, config)
        {
          vector<Requirement> reqs;
          reqs.emplace_back("occspace", "occ");
          reqs.emplace_back("vrtspace", "vrt");

          path_alpha_ip = config.get<string>("file_alpha_ip");  
          path_beta_ip  = config.get<string>("file_beta_ip");  
          path_gamma_ip = config.get<string>("file_gamma_ip");  
          path_norm_ip  = config.get<string>("file_norm_ip");  

          path_alpha_ea = config.get<string>("file_alpha_ea");  
          path_beta_ea  = config.get<string>("file_beta_ea");  
          path_gamma_ea = config.get<string>("file_gamma_ea");  
          path_norm_ea  = config.get<string>("file_norm_ea");  

          nlanczos = config.get<int>("nlanczos");

          this->addProduct(Product("ccsd.ipalpha", "alpha_ip", reqs));
          this->addProduct(Product("ccsd.ipbeta",  "beta_ip", reqs));
          this->addProduct(Product("ccsd.ipgamma", "gamma_ip", reqs));
          this->addProduct(Product("ccsd.ipnorm",  "norm_ip", reqs));

          this->addProduct(Product("ccsd.eaalpha", "alpha_ea", reqs));
          this->addProduct(Product("ccsd.eabeta",  "beta_ea", reqs));
          this->addProduct(Product("ccsd.eagamma", "gamma_ea", reqs));
          this->addProduct(Product("ccsd.eanorm",  "norm_ea", reqs));
        }

        bool run(TaskDAG& dag, const Arena& arena)
        {

            const auto& occ = this->template get<MOSpace<U>>("occ");
            const auto& vrt = this->template get<MOSpace<U>>("vrt");

            int norb = occ.nao[0] ;

            auto& alpha_ip = this-> put("alpha_ip", new vector<vector<U>>) ;
            auto& beta_ip = this-> put("beta_ip", new vector<vector<U>>) ;
            auto& gamma_ip = this-> put("gamma_ip", new vector<vector<U>>) ;
            auto& norm_ip = this-> put("norm_ip", new vector<U>) ;

            auto& alpha_ea = this-> put("alpha_ea", new vector<vector<U>>) ;
            auto& beta_ea = this-> put("beta_ea", new vector<vector<U>>) ;
            auto& gamma_ea = this-> put("gamma_ea", new vector<vector<U>>) ;
            auto& norm_ea = this-> put("norm_ea", new vector<U>) ;

            alpha_ip.resize(norb*(norb+1)/2) ;
            beta_ip.resize(norb*(norb+1)/2) ;
            gamma_ip.resize(norb*(norb+1)/2) ;
            norm_ip.resize(norb*(norb+1)/2) ;

            alpha_ea.resize(norb*(norb+1)/2) ;
            beta_ea.resize(norb*(norb+1)/2) ;
            gamma_ea.resize(norb*(norb+1)/2) ;
            norm_ea.resize(norb*(norb+1)/2) ;

            for (int n = 0;n < (norb*(norb+1)/2);n++){
              alpha_ip[n].resize(nlanczos) ;
              beta_ip[n].resize(nlanczos) ;
              gamma_ip[n].resize(nlanczos) ;

              alpha_ea[n].resize(nlanczos) ;
              beta_ea[n].resize(nlanczos) ;
              gamma_ea[n].resize(nlanczos) ;
            }
                                                                
             ifstream if_alpha_ip(path_alpha_ip);
             string line;
                                                                
             while (getline(if_alpha_ip, line))
             {
                U val;
                int p, q ;
                istringstream(line) >> p >> q >> val;
                alpha_ip[p][q] = val ; 
             }     
           
             ifstream if_beta_ip(path_beta_ip);

             while (getline(if_beta_ip, line))
             {
                U val;
                int p, q ;
                istringstream(line) >> p >> q >> val;
                beta_ip[p][q] = val ; 
             } 

             ifstream if_gamma_ip(path_gamma_ip);

             while (getline(if_gamma_ip, line))
             {
                U val;
                int p, q ;
                istringstream(line) >> p >> q >> val;
                gamma_ip[p][q] = val ; 
             }

             ifstream if_norm_ip(path_norm_ip);

             int countline = 0 ;
             while (getline(if_norm_ip, line))
             {
                U val;
                istringstream(line) >> val;
                norm_ip[countline] = val ; 
                countline++ ;
              }

             ifstream if_alpha_ea(path_alpha_ea);
                                                                
             while (getline(if_alpha_ea, line))
             {
                U val;
                int p, q ;
                istringstream(line) >> p >> q >> val;
                alpha_ea[p][q] = val ; 
             }     

             ifstream if_beta_ea(path_beta_ea);

             while (getline(if_beta_ea, line))
             {
                U val;
                int p, q ;
                istringstream(line) >> p >> q >> val;
                beta_ea[p][q] = val ; 
             } 

             ifstream if_gamma_ea(path_gamma_ea);

             while (getline(if_gamma_ea, line))
             {
                U val;
                int p, q ;
                istringstream(line) >> p >> q >> val;
                gamma_ea[p][q] = val ; 
             }

             ifstream if_norm_ea(path_norm_ea);

             countline = 0 ;
             while (getline(if_norm_ea, line))
             {
                U val;
                istringstream(line) >> val;
                norm_ea[countline] = val ; 
                countline++ ;
              }

        }
 };
}
}


static const char* spec = R"!(

    file_alpha_ip string, 
    file_beta_ip  string,
    file_gamma_ip string,
    file_norm_ip  string,

    file_alpha_ea string,
    file_beta_ea  string,
    file_gamma_ea string,
    file_norm_ea  string,
    nlanczos int, 
)!";

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::Assemble_Files);
REGISTER_TASK(CONCAT(aquarius::cc::Assemble_Files<double>),"assemble_files",spec);
