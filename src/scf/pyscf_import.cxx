#include "pyscf_import.hpp"
#include "util/global.hpp"

using namespace aquarius::input;
using namespace aquarius::tensor;
using namespace aquarius::integrals;
using namespace aquarius::task;
using namespace aquarius::time;
using namespace aquarius::op;
using namespace aquarius::symmetry;

namespace aquarius
{
namespace scf
{
template <typename T>
pyscf_import<T>::pyscf_import(const string& name, Config& config)
: Iterative<T>(name, config), frozen_core(config.get<bool>("frozen_core")),
  path_focka(config.get<string>("filename_focka")),path_fockb(config.get<string>("filename_fockb")),path_overlap(config.get<string>("filename_overlap")),diis(config.get("diis"), 2)
{
    damp_density = config.get<double>("damping_density");
    vector<Requirement> reqs;
    reqs += Requirement("molecule", "molecule");
    reqs += Requirement("ovi", "S");
    reqs += Requirement("1ehamiltonian", "H");
    this->addProduct(Product("double", "energy", reqs));
    this->addProduct(Product("double", "convergence", reqs));
    this->addProduct(Product("double", "S2", reqs));
    this->addProduct(Product("double", "multiplicity", reqs));
    this->addProduct(Product("occspace", "occ", reqs));
    this->addProduct(Product("vrtspace", "vrt", reqs));
    this->addProduct(Product("Ea", "Ea", reqs));
    this->addProduct(Product("Eb", "Eb", reqs));
    this->addProduct(Product("Fa", "Fa", reqs));
    this->addProduct(Product("Fb", "Fb", reqs));
    this->addProduct(Product("Da", "Da", reqs));
    this->addProduct(Product("Db", "Db", reqs));
    this->addProduct(Product("S", "S", reqs));
}

template <typename T>
bool pyscf_import<T>::run(TaskDAG& dag, const Arena& arena)
{
   const Molecule& molecule = this->template get<Molecule>("molecule");
   const PointGroup& group = molecule.getGroup();
   int norb = molecule.getNumOrbitals()[0];
   int nalpha = molecule.getNumAlphaElectrons();
   int nbeta = molecule.getNumBetaElectrons();
   int nirreps = group.getNumIrreps() ;

   vector<int> shapeNN = {NS,NS};
   vector<vector<int>> sizenn  = {{norb},{norb}};

    this->put("Fa", new SymmetryBlockedTensor<T>("Fa", arena, PointGroup::C1(), 2, sizenn, shapeNN, false));
    this->put("Fb", new SymmetryBlockedTensor<T>("Fb", arena, PointGroup::C1(), 2, sizenn, shapeNN, false));
    this->put("Da", new SymmetryBlockedTensor<T>("Da", arena, PointGroup::C1(), 2, sizenn, shapeNN, true));
    this->put("Db", new SymmetryBlockedTensor<T>("Db", arena, PointGroup::C1(), 2, sizenn, shapeNN, true));
//    this->put("S", new SymmetryBlockedTensor<T>("S", arena, PointGroup::C1(), 2, sizenn, shapeNN, false));

    this->puttmp("dF",     new SymmetryBlockedTensor<T>("dF",     arena, PointGroup::C1(), 2, sizenn, shapeNN, false));
    this->puttmp("Ca",     new SymmetryBlockedTensor<T>("Ca",     arena, PointGroup::C1(), 2, sizenn, shapeNN, false));
    this->puttmp("Cb",     new SymmetryBlockedTensor<T>("Cb",     arena, PointGroup::C1(), 2, sizenn, shapeNN, false));
    this->puttmp("dDa",    new SymmetryBlockedTensor<T>("dDa",    arena, PointGroup::C1(), 2, sizenn, shapeNN, false));
    this->puttmp("dDb",    new SymmetryBlockedTensor<T>("dDb",    arena, PointGroup::C1(), 2, sizenn, shapeNN, false));
    this->puttmp("S^-1/2", new SymmetryBlockedTensor<T>("S^-1/2", arena, PointGroup::C1(), 2, sizenn, shapeNN, false));

    occ_alpha.resize(nirreps);
    occ_beta.resize(nirreps);

    E_alpha.resize(nirreps);
    E_beta.resize(nirreps);

    for (int i = 0;i < nirreps;i++)
    {
        E_alpha[i].resize(norb);
        E_beta[i].resize(norb);
    }
//    get_overlap(arena);
    calcSMinusHalf();

    CTF_Timer_epoch ep(this->name.c_str());
    ep.begin();

    Iterative<T>::run(dag, arena);

    ep.end();

    if (this->isUsed("S2") || this->isUsed("multiplicity"))
    {
        calcS2();
    }

    this->put("energy", new T(this->energy()));
    this->put("convergence", new T(this->conv()));

    int nfrozen = 0;

    if (nfrozen > nalpha || nfrozen > nbeta)
        Logger::error(arena) << "There are not enough valence electrons for this multiplicity" << endl;

    vector<pair<real_type_t<T>,int>> E_alpha_occ;
    vector<pair<real_type_t<T>,int>> E_beta_occ;
    for (int i = 0;i < nirreps;i++)
    {
        for (int j = 0;j < occ_alpha[i];j++)
        {
            E_alpha_occ.push_back(make_pair(E_alpha[i][j],i));
        }
        for (int j = 0;j < occ_beta[i];j++)
        {
            E_beta_occ.push_back(make_pair(E_beta[i][j],i));
        }
    }
    assert(E_alpha_occ.size() == nalpha);
    assert(E_beta_occ.size() == nbeta);

    sort(E_alpha_occ.begin(), E_alpha_occ.end());
    sort(E_beta_occ.begin(), E_beta_occ.end());

    vector<int> nfrozen_alpha(nirreps);
    vector<int> nfrozen_beta(nirreps);
    for (int i = 0;i < nfrozen;i++)
    {
        nfrozen_alpha[E_alpha_occ[i].second]++;
        nfrozen_beta[E_beta_occ[i].second]++;
    }

    Logger::log(arena) << "Dropping MOs: " << nfrozen_alpha << ", " << nfrozen_beta << endl;

    vector<int> vrt_alpha(nirreps);
    vector<int> vrt_beta(nirreps);
    vector<int> vrt0_alpha(nirreps);
    vector<int> vrt0_beta(nirreps);
    for (int i = 0;i < nirreps;i++)
    {
        vrt_alpha[i] = norb-occ_alpha[i];
        vrt_beta[i] = norb-occ_beta[i];
        vrt0_alpha[i] = occ_alpha[i];
        vrt0_beta[i] = occ_beta[i];
        occ_alpha[i] -= nfrozen_alpha[i];
        occ_beta[i] -= nfrozen_beta[i];
    }

    for (int i = 0;i < nirreps;i++){
      cout <<  "occupied alpha electrons " << occ_alpha[i] << endl ; 
      cout <<  "occupied beta electrons " << occ_beta[i] << endl ; 
    }

    vector<int> zero(norb, 0);
    this->put("occ", new MOSpace<T>(SymmetryBlockedTensor<T>("CI", this->template gettmp<SymmetryBlockedTensor<T>>("Ca"),
                                                             {zero,nfrozen_alpha},
                                                             {{norb},occ_alpha}),
                                    SymmetryBlockedTensor<T>("Ci", this->template gettmp<SymmetryBlockedTensor<T>>("Cb"),
                                                             {zero,nfrozen_beta},
                                                             {{norb},occ_beta})));

    this->put("vrt", new MOSpace<T>(SymmetryBlockedTensor<T>("CA", this->template gettmp<SymmetryBlockedTensor<T>>("Ca"),
                                                             {zero,vrt0_alpha},
                                                             {{norb},vrt_alpha}),
                                       SymmetryBlockedTensor<T>("Ca", this->template gettmp<SymmetryBlockedTensor<T>>("Cb"),
                                                             {zero,vrt0_beta},
                                                             {{norb},vrt_beta})));
    vector<int> shapeN{NS};
    vector<vector<int>> sizena{{norb}};
    vector<vector<int>> sizenb{{norb}};
    for (int i = 0;i < nirreps;i++) sizena[0][i] -= nfrozen_alpha[i];
    for (int i = 0;i < nirreps;i++) sizenb[0][i] -= nfrozen_beta[i];

    auto& Ea = this->put("Ea", new vector<vector<real_type_t<T>>>(nirreps));
    auto& Eb = this->put("Eb", new vector<vector<real_type_t<T>>>(nirreps));

    for (int i = 0;i < nirreps;i++)
    {
        sort(E_alpha[i].begin(), E_alpha[i].end());
        
        Ea[i].assign(E_alpha[i].begin()+nfrozen_alpha[i], E_alpha[i].end());
    }

    for (int i = 0;i < nirreps;i++)
    {
        sort(E_beta[i].begin(), E_beta[i].end());
        for (int j = 0;j < norb;j++)
        {
//         printf("orbital#:  %d energy: %10f %10f\n",j, E_alpha[i][j], E_beta[i][j]);
        }
        Eb[i].assign(E_beta[i].begin()+nfrozen_beta[i], E_beta[i].end());
    }

    return true;
}

template <typename T>
void pyscf_import<T>::iterate(const Arena& arena)
{
   const Molecule& molecule = this->template get<Molecule>("molecule");
   const PointGroup& group = molecule.getGroup();
   int norb = molecule.getNumOrbitals()[0];
   int nalpha = molecule.getNumAlphaElectrons();
   int nbeta = molecule.getNumBetaElectrons();
   int nirreps = group.getNumIrreps() ;

    buildFock();
    DIISExtrap();
    calcEnergy();
    diagonalizeFock();

    vector<pair<real_type_t<T>,int>> E_alpha_sorted;
    vector<pair<real_type_t<T>,int>> E_beta_sorted;
    for (int i = 0;i < nirreps;i++)
    {
        occ_alpha[i] = 0;
        occ_beta[i] = 0;
        for (int j = 0;j < norb;j++)
        {
            E_alpha_sorted.push_back(make_pair(E_alpha[i][j],i));
        }
        for (int j = 0;j < norb;j++)
        {
            E_beta_sorted.push_back(make_pair(E_beta[i][j],i));
            printf("orbital#:  %d energy: %10f %10f\n",j, E_alpha_sorted[j].first, E_beta_sorted[j].first);
        }
    }

    sort(E_alpha_sorted.begin(), E_alpha_sorted.end());
    sort(E_beta_sorted.begin(), E_beta_sorted.end());

    for (int i = 0;i < nalpha;i++)
    {
        occ_alpha[E_alpha_sorted[i].second]++;
    }
    for (int i = 0;i < nbeta;i++)
    {
        occ_beta[E_beta_sorted[i].second]++;
    }

    Logger::log(arena) << "Iteration " << this->iter() << " occupation = " << occ_alpha << ", " << occ_beta << endl;

    calcDensity();
//    diagonalizeDensity();

    auto& dDa = this->template gettmp<SymmetryBlockedTensor<T>>("dDa");
    auto& dDb = this->template gettmp<SymmetryBlockedTensor<T>>("dDb");

    switch (this->convtype)
    {
        case Iterative<T>::MAX_ABS:
            this->conv() = max(dDa.norm(00), dDb.norm(00));
            break;
        case Iterative<T>::RMSD:
            this->conv() = (dDa.norm(2)+dDb.norm(2))/sqrt(2*norb*norb);
            break;
        case Iterative<T>::MAD:
            this->conv() = (dDa.norm(1)+dDb.norm(1))/(2*norb*norb);
            break;
    }
}

template <typename T>
void pyscf_import<T>::get_overlap(const Arena& arena)
{
   const Molecule& molecule = this->template get<Molecule>("molecule");
   const PointGroup& group = molecule.getGroup();
   const int norb = molecule.getNumOrbitals()[0];
   int nirrep = group.getNumIrreps() ;

   auto& S = this->template get<SymmetryBlockedTensor<T>>("S");

    vector<vector<T>> overlap(nirrep) ; 

    for (int i = 0;i < nirrep;i++)
    {
      overlap[i].resize(norb*norb, (T)0);

    if (arena.rank == 0)
    {
      ifstream ifs(path_overlap);
      string line;

      while (getline(ifs, line))
      {
          T val;
          int p, q;
          istringstream(line) >> val >> p >> q ;

          overlap[i][(q-1)*norb+(p-1)]  = val; 
          overlap[i][(p-1)*norb+(q-1)]  = val; 
      }
     }
    }

    for (int i = 0;i < nirrep;i++)
    {
        vector<int> irreps(2,i);

        if (arena.rank == 0)
        {
            //PROFILE_FLOPS(2*norb[i]*norb[i]);
            arena.comm().Reduce(overlap[i], MPI_SUM);

            vector<tkv_pair<T>> pairs(norb*norb);

            for (int p = 0;p < norb*norb;p++)
            {
                pairs[p].d = overlap[i][p];
                pairs[p].k = p;
            }

            S.writeRemoteData(irreps, pairs);
        }
        else
        {
            //PROFILE_FLOPS(2*norb[i]*norb[i]);
            arena.comm().Reduce(overlap[i], MPI_SUM, 0);

            S.writeRemoteData(irreps);
        }
    }
}

template <typename T>
void pyscf_import<T>::calcSMinusHalf()
{
   const Molecule& molecule = this->template get<Molecule>("molecule");
   const PointGroup& group = molecule.getGroup();
   const int norb = molecule.getNumOrbitals()[0];
   int nalpha = molecule.getNumAlphaElectrons();
   int nbeta = molecule.getNumBetaElectrons();
   int nirreps = group.getNumIrreps() ;

    auto& S = this->template get<SymmetryBlockedTensor<T>>("S");
    auto& Smhalf = this->template gettmp<SymmetryBlockedTensor<T>>("S^-1/2");

    for (int i = 0;i < nirreps;i++)
    {
        //cout << "S " << (i+1) << endl;
        //vector<T> vals;
        //S({i,i}).getAllData(vals);
        //printmatrix(norb[i], norb[i], vals.data(), 6, 3, 108);
    }

    for (int i = 0;i < nirreps;i++)
    {
        if (norb == 0) continue;

        vector<int> irreps(2,i);
        vector<real_type_t<T>> E(norb);

        if (S.arena.rank == 0)
        {
            vector<T> s;
            vector<T> smhalf(norb*norb);

            S.getAllData(irreps, s, 0);
            assert(s.size() == norb*norb);

            //PROFILE_FLOPS(26*norb[i]*norb[i]*norb[i]);
            int info = heev('V', 'U', norb, s.data(), norb, E.data());
            assert(info == 0);

            fill(smhalf.begin(), smhalf.end(), (T)0);
            //PROFILE_FLOPS(2*norb[i]*norb[i]*norb[i]);
            for (int j = 0;j < norb;j++)
            {
                ger(norb, norb, 1/sqrt(E[j]), &s[j*norb], 1, &s[j*norb], 1, smhalf.data(), norb);
            }

            vector<tkv_pair<T>> pairs(norb*norb);

            for (int j = 0;j < norb*norb;j++)
            {
                pairs[j].k = j;
                pairs[j].d = smhalf[j];
            }

            Smhalf.writeRemoteData(irreps, pairs);
        }
        else
        {
            S.getAllData(irreps, 0);
            Smhalf.writeRemoteData(irreps);
        }
    }
}

template <typename T>
void pyscf_import<T>::diagonalizeFock()
{
   const Molecule& molecule = this->template get<Molecule>("molecule");
   const PointGroup& group = molecule.getGroup();
   const int norb = molecule.getNumOrbitals()[0];
   int nalpha = molecule.getNumAlphaElectrons();
   int nbeta = molecule.getNumBetaElectrons();
   int nirreps = group.getNumIrreps() ;

    auto& S  = this->template get  <SymmetryBlockedTensor<T>>("S");
    auto& Fa = this->template get   <SymmetryBlockedTensor<T>>("Fa");
    auto& Fb = this->template get   <SymmetryBlockedTensor<T>>("Fb");
    auto& Ca = this->template gettmp<SymmetryBlockedTensor<T>>("Ca");
    auto& Cb = this->template gettmp<SymmetryBlockedTensor<T>>("Cb");

    for (int i = 0;i < nirreps;i++)
    {
        //cout << "F " << (i+1) << endl;
        //vector<T> vals;
        //Fa({i,i}).getAllData(vals);
        //printmatrix(norb[i], norb[i], vals.data(), 6, 3, 108);
    }

    for (int i = 0;i < nirreps;i++)
    {
//        if (norb == 0) continue;

        vector<int> irreps(2,i);

        if (S.arena.rank == 0)
        {
            vector<T> s;
            S.getAllData(irreps, s, 0);
            assert(s.size() == norb*norb);

            for (int spin : {0,1})
            {
                auto& F = (spin == 0 ? Fa : Fb);
                auto& C = (spin == 0 ? Ca : Cb);
                auto& E = (spin == 0 ? E_alpha[i] : E_beta[i]);

                int info;
                vector<T> fock, ctsc(norb*norb);
                vector<tkv_pair<T>> pairs(norb*norb);
                vector<T> tmp(s);

                F.getAllData(irreps, fock, 0);
                assert(fock.size() == norb*norb);
                //PROFILE_FLOPS(9*norb[i]*norb[i]*norb[i]);

                info = hegv(AXBX, 'V', 'U', norb, fock.data(), norb, tmp.data(), norb, E.data());

                if (info != 0) throw runtime_error(str("check diagonalization: Info in hegv: %d", info));

                assert(info == 0);

                S.arena.comm().Bcast(E);

                vector<pair<real_type_t<T>,int>> E_sort;
                for (int j = 0;j < norb;j++)
                {
                  E_sort.push_back(make_pair(E[j],j));
                }

                sort(E_sort.begin(), E_sort.end());

                for (int j = 0;j < norb;j++)
                {
                  cout << E_sort[j].second << endl ;
                }

//              for (int j = 0;j < norb;j++)
//              {
//                  T sign = 0;
//                  for (int k = 0;k < norb;k++)
//                  {
//                      if (aquarius::abs(fock[k+j*norb]) > 1e-10)
//                      {
//                          sign = (fock[k+j*norb] < 0 ? -1 : 1);
//                          break;
//                      }
//                  }
//                  //PROFILE_FLOPS(norb[i]);
//                  scal(norb, sign, &fock[j*norb], 1);
//              }

                for (int i = 0;i < norb;i++)
                {
                 for (int j = 0;j < norb;j++)
                  {
                    pairs[i*norb+j].k = i*norb+j;
                    pairs[i*norb+j].d = fock[j+E_sort[i].second*norb];
                  }
                }

                C.writeRemoteData(irreps, pairs);
            }
        }
        else
        {
            S.getAllData(irreps, 0);
            Fa.getAllData(irreps, 0);
            S.arena.comm().Bcast(E_alpha[i], 0);
            Ca.writeRemoteData(irreps);
            Fb.getAllData(irreps, 0);
            S.arena.comm().Bcast(E_beta[i], 0);
            Cb.writeRemoteData(irreps);
        }
    }
}

template <typename T>
void pyscf_import<T>::diagonalizeDensity()
{
   const Molecule& molecule = this->template get<Molecule>("molecule");
   const PointGroup& group = molecule.getGroup();
   const int norb = molecule.getNumOrbitals()[0];
   int nalpha = molecule.getNumAlphaElectrons();
   int nbeta = molecule.getNumBetaElectrons();
   int nirreps = group.getNumIrreps() ;

    auto& Da = this->template get   <SymmetryBlockedTensor<T>>("Da");
    auto& Db = this->template get   <SymmetryBlockedTensor<T>>("Db");
    auto& Ca = this->template gettmp<SymmetryBlockedTensor<T>>("Ca");
    auto& Cb = this->template gettmp<SymmetryBlockedTensor<T>>("Cb");

    for (int i = 0;i < nirreps;i++)
    {
        if (norb == 0) continue;

        vector<int> irreps(2,i);

        if (Da.arena.rank == 0)
        {

            for (int spin : {0,1})
            {
                auto& D = (spin == 0 ? Da : Db);
                auto& C = (spin == 0 ? Ca : Cb);

                int info;
                vector<T> dens, ctsc(norb*norb);
                vector<tkv_pair<T>> pairs(norb*norb);
                vector<T> tmp(norb*norb, 0.);
                vector<T> occ(norb, 0.) ;

                D.getAllData(irreps, dens, 0);
                assert(dens.size() == norb*norb);
                //PROFILE_FLOPS(9*norb[i]*norb[i]*norb[i]);

                cout << "diagonal values of density" << endl ;
                for (int j = 0 ;j < norb; j++)
                {
          //          cout << j << " " << dens[j+j*norb] << endl ;
                }  

                info = hegv(AXBX, 'V', 'U', norb, dens.data(), norb, tmp.data(), norb, occ.data());

                if (info != 0) throw runtime_error(str("check diagonalization: Info in hegv: %d", info));

                assert(info == 0);

                for (int j = 0;j < norb;j++)
                {
                    T sign = 0;
                    for (int k = 0;k < norb;k++)
                    {
                        if (aquarius::abs(dens[k+j*norb]) > 1e-10)
                        {
                            sign = (dens[k+j*norb] < 0 ? -1 : 1);
                            break;
                        }
                    }
                    //PROFILE_FLOPS(norb[i]);
                    scal(norb, sign, &dens[j*norb], 1);
                }

                for (int j = 0;j < norb*norb;j++)
                {
                    pairs[j].k = j;
                    pairs[j].d = dens[j];
                }
                C.writeRemoteData(irreps, pairs);
            }
        }
        else
        {
            Da.getAllData(irreps, 0);
            Ca.writeRemoteData(irreps);
            Db.getAllData(irreps, 0);
            Cb.writeRemoteData(irreps);
        }
    }
}

template <typename T>
void pyscf_import<T>::calcS2()
{
   const Molecule& molecule = this->template get<Molecule>("molecule");
   const PointGroup& group = molecule.getGroup();
   const int norb = molecule.getNumOrbitals()[0];
   int nalpha = molecule.getNumAlphaElectrons();
   int nbeta = molecule.getNumBetaElectrons();
   int nirreps = group.getNumIrreps() ;

    auto& S = this->template get<SymmetryBlockedTensor<T>>("S"); // this must be set to a unit matrix

    vector<int> zero(norb, 0);
    SymmetryBlockedTensor<T> Ca_occ("CI", this->template gettmp<SymmetryBlockedTensor<T>>("Ca"),
                                    {zero,zero}, {{norb},occ_alpha});
    SymmetryBlockedTensor<T> Cb_occ("Ci", this->template gettmp<SymmetryBlockedTensor<T>>("Cb"),
                                    {zero,zero}, {{norb},occ_beta});

    SymmetryBlockedTensor<T> Delta("Delta", S.arena, PointGroup::C1(), 2, {{nalpha},{nbeta}}, {NS,NS}, false);
    SymmetryBlockedTensor<T> tmp("tmp", S.arena, PointGroup::C1(), 2, {{nalpha},{norb}}, {NS,NS}, false);

    int ndiff = abs(nalpha-nbeta);
    int nmin = min(nalpha, nbeta);

    double S2 = ((ndiff/2)*(ndiff/2+1) + nmin);

    tmp["ai"] = Ca_occ["ja"]*S["ij"];
    Delta["ab"] = tmp["ai"]*Cb_occ["ib"];

    S2 -= aquarius::abs(scalar(Delta*conj(Delta)));

    this->put("S2", new T(S2));
    this->put("multiplicity", new T(sqrt(4*S2+1)));
}

template <typename T>
void pyscf_import<T>::calcEnergy()
{

    auto& H  = this->template get<SymmetryBlockedTensor<T>>("H");
    auto& Fa = this->template get<SymmetryBlockedTensor<T>>("Fa");
    auto& Fb = this->template get<SymmetryBlockedTensor<T>>("Fb");
    auto& Da = this->template get<SymmetryBlockedTensor<T>>("Da");
    auto& Db = this->template get<SymmetryBlockedTensor<T>>("Db");

    /*
     * E = (1/2)Tr[D(F+H)]
     *
     *   = (1/2)Tr[Da*(Fa+H) + Db*(Fb+H)]
     */
    Fa["ab"] += H["ab"];
    Fb["ab"] += H["ab"];

    this->energy()  = 0.5*scalar(Da["ab"]*Fa["ab"]);
    this->energy() += 0.5*scalar(Db["ab"]*Fb["ab"]);
    this->energy() += 1288.9039479060002 ;
    Fa["ab"] -= H["ab"];
    Fb["ab"] -= H["ab"];
}

template <typename T>
void pyscf_import<T>::calcDensity()
{
   const Molecule& molecule = this->template get<Molecule>("molecule");
   const PointGroup& group = molecule.getGroup();
   const int norb = molecule.getNumOrbitals()[0];
   int nalpha = molecule.getNumAlphaElectrons();
   int nbeta = molecule.getNumBetaElectrons();
   int nirreps = group.getNumIrreps() ;

    auto& dDa = this->template gettmp<SymmetryBlockedTensor<T>>("dDa");
    auto& dDb = this->template gettmp<SymmetryBlockedTensor<T>>("dDb");
    auto& Da  = this->template get   <SymmetryBlockedTensor<T>>("Da");
    auto& Db  = this->template get   <SymmetryBlockedTensor<T>>("Db");

    vector<int> zero(norb, 0);
    SymmetryBlockedTensor<T> Ca_occ("CI", this->template gettmp<SymmetryBlockedTensor<T>>("Ca"),
                                    {zero,zero}, {{norb},occ_alpha});
    SymmetryBlockedTensor<T> Cb_occ("Ci", this->template gettmp<SymmetryBlockedTensor<T>>("Cb"),
                                    {zero,zero}, {{norb},occ_beta});
    /*
     * D[ab] = C[ai]*C[bi]
     */
 
     dDa["ab"]  = Da["ab"];
     dDb["ab"]  = Db["ab"];
     Da["ab"]  = Ca_occ["ai"]*Ca_occ["bi"];
     Db["ab"]  = Cb_occ["ai"]*Cb_occ["bi"];
     dDa["ab"] -= Da["ab"];
     dDb["ab"] -= Db["ab"];
}

template <typename T>
void pyscf_import<T>::buildFock()
{
   const Molecule& molecule = this->template get<Molecule>("molecule");
   const PointGroup& group = molecule.getGroup();
   const int norb = molecule.getNumOrbitals()[0];
   int nirrep = group.getNumIrreps() ;

    auto& Fa = this->template get<SymmetryBlockedTensor<T>>("Fa");
    auto& Fb = this->template get<SymmetryBlockedTensor<T>>("Fb");

    Arena& arena = Fa.arena;

    vector<vector<T>> focka(nirrep), fockb(nirrep);
    T diff ;

    for (int i = 0;i < nirrep;i++)
    {
        vector<int> irreps(2,i);

         focka[i].resize(norb*norb, (T)0);
         fockb[i].resize(norb*norb, (T)0);
     
    if (arena.rank == 0)
    {
      ifstream ifs(path_focka);
      ifstream ifb(path_fockb);
      string linea,lineb;

      int countline = 0 ;

      while (getline(ifs, linea))
      {
          countline++ ;

          T val;
          int p, q;
          istringstream(linea) >> val >> p >> q ;

          focka[i][(q-1)*norb+(p-1)]  = val; 
          focka[i][(p-1)*norb+(q-1)]  = val; 
      }

      countline = 0 ;
      while (getline(ifb, lineb))
      {
          countline++ ;

          T val;
          int p, q;
          istringstream(lineb) >> val >> p >> q ;

          fockb[i][(q-1)*norb+(p-1)]  = val; 
          fockb[i][(p-1)*norb+(q-1)]  = val; 

      } 
    }
   }

    for (int i = 0;i < nirrep;i++)
    {
        vector<int> irreps(2,i);

        if (arena.rank == 0)
        {
            //PROFILE_FLOPS(2*norb[i]*norb[i]);
            arena.comm().Reduce(focka[i], MPI_SUM);
            arena.comm().Reduce(fockb[i], MPI_SUM);

            vector<tkv_pair<T>> pairs(norb*norb);

            for (int p = 0;p < norb*norb;p++)
            {
                pairs[p].d = focka[i][p];
                pairs[p].k = p;
            }

            Fa.writeRemoteData(irreps, pairs);

            for (int p = 0;p < norb*norb;p++)
            {
                pairs[p].d = fockb[i][p];
                pairs[p].k = p;
            }

            Fb.writeRemoteData(irreps, pairs);
        }
        else
        {
            //PROFILE_FLOPS(2*norb[i]*norb[i]);
            arena.comm().Reduce(focka[i], MPI_SUM, 0);
            arena.comm().Reduce(fockb[i], MPI_SUM, 0);

            Fa.writeRemoteData(irreps);
            Fb.writeRemoteData(irreps);
        }
    }
}

template <typename T>
void pyscf_import<T>::DIISExtrap()
{
    auto& S      = this->template get   <SymmetryBlockedTensor<T>>("S");
    auto& Smhalf = this->template gettmp<SymmetryBlockedTensor<T>>("S^-1/2");
    auto& dF     = this->template gettmp<SymmetryBlockedTensor<T>>("dF");
    auto& Fa     = this->template get   <SymmetryBlockedTensor<T>>("Fa");
    auto& Fb     = this->template get   <SymmetryBlockedTensor<T>>("Fb");
    auto& Da     = this->template get   <SymmetryBlockedTensor<T>>("Da");
    auto& Db     = this->template get   <SymmetryBlockedTensor<T>>("Db");

    /*
     * Generate the residual:
     *
     *   R = FDS - SDF
     *
     * Then, convert to the orthonormal basis:
     *
     *   ~    -1/2    -1/2
     *   R = S     R S.
     *
     * In this basis we have
     *
     *   ~    -1/2    -1/2  ~    1/2
     *   F = S     F S    , C = S    C, and
     *
     *   ~   ~ ~T    1/2    T  1/2    1/2    1/2
     *   D = C C  = S    C C  S    = S    D S.
     *
     * And so,
     *
     *   ~    -1/2    -1/2  1/2    1/2    1/2    1/2  -1/2    -1/2
     *   R = S     F S     S    D S    - S    D S    S     F S
     *
     *        ~ ~
     *     = [F,D] = 0 at convergence.
     */
    {
        SymmetryBlockedTensor<T> tmp1("tmp", Fa);
        SymmetryBlockedTensor<T> tmp2("tmp", Fa);

        tmp1["ab"]  =     Fa["ac"]*    Da["cb"];
        tmp2["ab"]  =   tmp1["ac"]*     S["cb"];
        tmp1["ab"]  =      S["ac"]*    Da["cb"];
        tmp2["ab"] -=   tmp1["ac"]*    Fa["cb"];
        tmp1["ab"]  = Smhalf["ac"]*  tmp2["cb"];
          dF["ab"]  =   tmp1["ac"]*Smhalf["cb"];

        tmp1["ab"]  =     Fb["ac"]*    Db["cb"];
        tmp2["ab"]  =   tmp1["ac"]*     S["cb"];
        tmp1["ab"]  =      S["ac"]*    Db["cb"];
        tmp2["ab"] -=   tmp1["ac"]*    Fb["cb"];
        tmp1["ab"]  = Smhalf["ac"]*  tmp2["cb"];
          dF["ab"] +=   tmp1["ac"]*Smhalf["cb"];
    }

    diis.extrapolate(ptr_vector<SymmetryBlockedTensor<T>>{&Fa, &Fb},
                     ptr_vector<SymmetryBlockedTensor<T>>{&dF});
}

}
}

static const char* spec = R"(

    filename_focka?
        string focka.txt,
    filename_fockb?
        string fockb.txt,
    filename_overlap?
        string overlap.txt,
    frozen_core?
        bool false,
    convergence?
        double 1e-12,
    max_iterations?
        int 150,
    damping_density?
        double 0., 
    conv_type?
        enum { MAXE, RMSE, MAE },
    diis?
    {
        damping?
            double 0.0,
        start?
            int 8,
        order?
            int 6,
        jacobi?
            bool false
    }

)";

INSTANTIATE_SPECIALIZATIONS(aquarius::scf::pyscf_import);
REGISTER_TASK(CONCAT(aquarius::scf::pyscf_import<double>), "pyscf_import",spec);
