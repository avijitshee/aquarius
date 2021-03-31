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
class Assemble_Files_Linsolver: public Task 
{
    protected:
        typedef complex_type_t<U> CU;

        string path_gf_ip ;
        string path_gf_ea ; 
        int nfreq ;

    public:
        Assemble_Files_Linsolver(const string& name, Config& config): Task(name, config)
        {
          vector<Requirement> reqs;
          reqs.emplace_back("occspace", "occ");
          reqs.emplace_back("vrtspace", "vrt");

          path_gf_ip = config.get<string>("file_gf_ip");  
          path_gf_ea = config.get<string>("file_gf_ea");  
          nfreq = config.get<int>("nomega");  

          this->addProduct(Product("cc.gf", "gf", reqs));

        }

        bool run(TaskDAG& dag, const Arena& arena)
        {
            const auto& occ = this->template get<MOSpace<U>>("occ");
            const auto& vrt = this->template get<MOSpace<U>>("vrt");

            int nI = occ.nalpha[0];
            int ni = occ.nbeta[0];

            int norb = occ.nao[0] ;
//          int maxspin = (nI == ni) ? 1 : 2 ;
            int maxspin =  2 ;

            auto& gf = this-> put("gf", new vector<vector<vector<CU>>>) ;
            vector<vector<vector<CU>>> gf_tmp ;

            gf_tmp.resize(maxspin) ;
            gf.resize(maxspin) ;

            for (int nspin = 0;nspin < maxspin;nspin++){
              gf_tmp[nspin].resize(norb*norb);
            }  

            for (int nspin = 0;nspin < maxspin;nspin++){
              for (int i = 0;i < norb*norb ;i++){
                gf_tmp[nspin][i].resize(nfreq);
              }
            }


            for (int nspin = 0;nspin < maxspin;nspin++){
              gf[nspin].resize(nfreq);
            }  

            for (int nspin = 0;nspin < maxspin;nspin++){
              for (int i = 0;i < nfreq ;i++){
                gf[nspin][i].resize(norb*norb);
              }
            }

                                                                
             ifstream if_gf_ip(path_gf_ip);
             string line;
                                                                
             while (getline(if_gf_ip, line))
             {
                U valr;
                U vali;
                int p, q, r ;
                istringstream(line) >> p >> q >> r >> valr >> vali;
                gf_tmp[p][q][r] = {valr, vali} ;
             }     
           
             ifstream if_gf_ea(path_gf_ea);
                                                                
             while (getline(if_gf_ea, line))
             {
                U valr;
                U vali;
                int p, q, r ;
                istringstream(line) >> p >> q >> r >> valr >> vali;
                CU val = {valr, vali} ;
                gf_tmp[p][q][r] += val ;
             }     

            for (int nspin = 0;nspin < maxspin;nspin++){
              for (int i = 0;i < norb*norb ;i++){
                for (int w = 0;w < nfreq ;w++){

                    gf[nspin][w][i] = gf_tmp[nspin][i][w] ;

                }
              }
            }

              return true ;
        }
 };
}
}


static const char* spec = R"!(

    file_gf_ip string, 
    file_gf_ea string,
    nomega int,
)!";

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::Assemble_Files_Linsolver);
REGISTER_TASK(CONCAT(aquarius::cc::Assemble_Files_Linsolver<double>),"assemble_files_linsolver",spec);
