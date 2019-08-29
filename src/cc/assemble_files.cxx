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

        string path_alpha_ip, path_beta_ip, path_gamma_ip, path_norm_ip, path_lanczos_ip ;
        string path_alpha_ea, path_beta_ea, path_gamma_ea, path_norm_ea, path_lanczos_ea ; 

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
          path_lanczos_ip  = config.get<string>("file_lanczos_ip");  

          path_alpha_ea = config.get<string>("file_alpha_ea");  
          path_beta_ea  = config.get<string>("file_beta_ea");  
          path_gamma_ea = config.get<string>("file_gamma_ea");  
          path_norm_ea  = config.get<string>("file_norm_ea");  
          path_lanczos_ea  = config.get<string>("file_lanczos_ea");  

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

            vector<int> lanczos_ip ;
            vector<int> lanczos_ea ;

            if (arena.rank == 0)
            {
             ifstream if_lanczos_ip(path_lanczos_ip);
             string line1;
                                                                
             while (getline(if_lanczos_ip, line1))
             {
                int val;
                int p ;
                istringstream(line1) >> p >> val;
                lanczos_ip.emplace_back(val) ; 
             }

             ifstream if_lanczos_ea(path_lanczos_ea);
             while (getline(if_lanczos_ea, line1))
             {
                int val;
                int p ;
                istringstream(line1) >> p >> val;
                lanczos_ea.emplace_back(val) ; 
             }
            }


            alpha_ip.resize(norb*(norb+1)/2) ;
            beta_ip.resize(norb*(norb+1)/2) ;
            gamma_ip.resize(norb*(norb+1)/2) ;
            norm_ip.resize(norb*(norb+1)/2) ;

            alpha_ea.resize(norb*(norb+1)/2) ;
            beta_ea.resize(norb*(norb+1)/2) ;
            gamma_ea.resize(norb*(norb+1)/2) ;
            norm_ea.resize(norb*(norb+1)/2) ;

            for (int n = 0;n < (norb*(norb+1)/2);n++){
              alpha_ip[n].resize(lanczos_ip[n]) ;
              beta_ip[n].resize(lanczos_ip[n]) ;
              gamma_ip[n].resize(lanczos_ip[n]) ;

              alpha_ea[n].resize(lanczos_ea[n]) ;
              beta_ea[n].resize(lanczos_ea[n]) ;
              gamma_ea[n].resize(lanczos_ea[n]) ;
            }

            if (arena.rank == 0)
            {
                                                                
             ifstream if_alpha_ip(path_alpha_ip);
             string line1;
                                                                
             while (getline(if_alpha_ip, line1))
             {
                U val;
                int p, q ;
                istringstream(line1) >> p >> q >> val;
                alpha_ip[p][q] = val ; 
             }     
           
             ifstream if_beta_ip(path_beta_ip);
             string line2;

             while (getline(if_beta_ip, line2))
             {
                U val;
                int p, q ;
                istringstream(line2) >> p >> q >> val;
                beta_ip[p][q] = val ; 
             } 

             ifstream if_gamma_ip(path_gamma_ip);
             string line3;

             while (getline(if_gamma_ip, line3))
             {
                U val;
                int p, q ;
                istringstream(line3) >> p >> q >> val;
                gamma_ip[p][q] = val ; 
             }

             ifstream if_norm_ip(path_norm_ip);
             string line4;

             int countline = 0 ;
             while (getline(if_norm_ip, line4))
             {
                U val;
                istringstream(line4) >> val;
                norm_ip[countline] = val ; 
                countline++ ;
              }


             ifstream if_alpha_ea(path_alpha_ea);
             string line5;
                                                                
             while (getline(if_alpha_ea, line5))
             {
                U val;
                int p, q ;
                istringstream(line5) >> p >> q >> val;
                alpha_ea[p][q] = val ; 
             }     

             ifstream if_beta_ea(path_beta_ea);
             string line6;

             while (getline(if_beta_ea, line6))
             {
                U val;
                int p, q ;
                istringstream(line6) >> p >> q >> val;
                beta_ea[p][q] = val ; 
             } 

             ifstream if_gamma_ea(path_gamma_ea);
             string line7;

             while (getline(if_gamma_ea, line7))
             {
                U val;
                int p, q ;
                istringstream(line7) >> p >> q >> val;
                gamma_ea[p][q] = val ; 
             }

             ifstream if_norm_ea(path_norm_ea);
             string line8;

             countline = 0 ;
             while (getline(if_norm_ea, line8))
             {
                U val;
                istringstream(line8) >> val;
                norm_ea[countline] = val ; 
                countline++ ;
              }
             return true ;
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
    file_lanczos_ip  string,

    file_alpha_ea string,
    file_beta_ea  string,
    file_gamma_ea string,
    file_norm_ea  string,
    file_lanczos_ea  string,
)!";

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::Assemble_Files);
REGISTER_TASK(CONCAT(aquarius::cc::Assemble_Files<double>),"assemble_files",spec);
