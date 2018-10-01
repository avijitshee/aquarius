Description of Some of the Keywords introduced in regard to the Coupled Cluster Green's Function calculation
=============================================================================================================

1> grid : imaginary => Matsubara frequency
          real      => real frequency
2> beta : inverse temperature.
3> eta  : broadening or small imaginary part to calculate Green's function in real frequency plane.
4> gftype : symmetrized => Lanczos solver
            nonsymmetrized => Linear equation solver
5> omega_min : minimum value in a real frequency window.
6> omega_max : maximum value in a real frequency window.
7> impurities : total number of impurities when calculation is carried out for an impurity model.
8> npoints/frequency_points : total number of frequency points to be considered.
9> orbital : the orbital for which we want to evaluate diagonal Green's function element. Used in IPGF, EAGF both for Lanczos solver and linear equation solver.
10> orbital_range: full => to denote that we want to calculate all Green's function elements.
                   diagonal => we want to calculate only one diagonal Green's function element.
11> 
