# Multi-adaptive Stochastic Topology Optimization
# Code via FEniCS 2017.1
# 
# * Adapted from Laurain A. (2017): "A level set-based 
#   structural optimization code using fenics"
# * Current features:
#   - Sample average approximations
#   - Adaptive sample size selection
#   - Mesh adaptation wrt given goal
#   - Mesh adaptation wrt shape derivatives
#   - Variance reducing optimal step size estimation
# 
# Author        : han.altintas
# Last update   : 07.09.2026

from dolfin import *
from fenics import *
from matplotlib import cm
from matplotlib import pyplot as pp
import numpy as np
import os
from scipy.optimize import root_scalar
from math import *
import random
import time
from dolfin_dg.dolfin import FixedFractionMarkerParallel    # Need to be installed from "dolfin_dg" Github repo
import ufl
import pickle

pp.switch_backend('qtagg')


# Initializations and main loop
def _main(CASE_NAME):

    # Create saving folder
    caseName = CASE_NAME

    folderName = os.path.join(os.path.dirname(__file__), caseName)
    if not os.path.isdir(folderName): os.makedirs(folderName) 


    # Optimization tolerance
    EPS_TOL         = 1E-6                                          # Stopping criterion for optimization steps, 1E-4 for det. case
    EPS_GEOM        = 1E-11                                         # OBSOLETE

    # Number of elements on working domain
    NUM_ELEM_NX     = 60  * 1                                       # Number of elements in x-direction
    NUM_ELEM_NY     = 120 * 1                                       # Number of elements in y-direction

    # Set mesh diagonals leaning left, right or crossed
    MESH_DIAG       = 'crossed'                                     # Mesh diagonality
    
    # Dimensions of working domain
    LEN_X           = 1.0                                           # x-length of working domain
    LEN_Y           = 2.0                                           # y-length of working domain

    # Define problem parameters
    CONST_EPS       = 1E-3                                          # Ersatz material epsilon parameter

    # Define line search parameters
    LS_BETA0_INIT   = 0.01                                          # Line-search parameters (if used)
    LS_ITER_MAX     = 3                                             #
    LS_GAMMA1       = 0.8                                           #
    LS_GAMMA2       = 0.8                                           #
    HJE_TIMESPAN    = 250                                           # HJE solution timespan, t, T=10-20 for det. case

    # Define Sobolev metric used in bilinear form
    BFORM_ALPHA1    = 1E3                                           # Descent direction problem parameters
    BFORM_ALPHA2    = 1E0                                           #

    # Define MC sample sizes (enter 1 for each in deterministic cases)
    MC_SAMPLE_N     = 30                                            # Size of full MC sample
    MC_BATCH_N      = 30                                            # Size of mini-batch

    # Define KLE parameters for random variables
    KLE_CORR_LEN    = 1                                             # KLE correlation length
    KLE_N_MAX       = 100                                           # KLE maximum number of terms
    KLE_ENERGY_REQ  = 0.9                                           # KLE energy requirement percentage
    KLE_LOAD_MEAN   = 90.0                                          # KLE mean of random load
    KLE_LOAD_STD    = 30.0   * 1                                    # KLE std. dev. of random load (multiply 0 if deterministic)
    KLE_E_MEAN      = 1.0                                           # KLE mean of random material Young modulus
    KLE_E_STD       = 3.0    * 0                                    # KLE std. dev. of random material Young modulus
    KLE_NU_MEAN     = 0.3                                           # KLE mean of random material Poisson ratio
    KLE_NU_STD      = 0.0                                           # KLE std. dev. of random material Poisson ratio

    # Target volume fraction
    VOL_FRAC        = 0.3                                           # Optimization target volume fraction

    # Penalty parameters
    PENAL_MU_INIT   = 100.0 #0.01                                   # Volume penalty parameter
    PENAL_MU_FACT   = 1.0   #1.05                                   # Volume penalty parameter growth factor (if not constant)
    PENAL_MU_MAX    = 500                                           # Volume penalty parameter max. value (if not constant)

    # Adaptive sampling parameters
    ADAPT_SAMP_TYPE = 'inner'                                       # Adaptive sampling strategy type: 'inner', 'norm'
    ADAPT_SAMP_NU   = 5.8                                           # Adaptive sampling strategy parameter, nu
    ADAPT_SAMP_THE  = 0.6                                           # Adaptive sampling strategy parameter, theta
    ADAPT_SAMP_MAX  = MC_SAMPLE_N                                   # Adaptive sampling strategy max. sample size

    # Plotting options
    PLOT_PERIOD     = 5                                             # Plotting period
    PLOT_GEOM       = [200, 10, 800, 1600]                          # Plot position and size

    # Mesh refinement algorithm parameters
    MESH_REF_FLAG   = 1                                             # Mesh refinement flag
    MESH_REF_TOL    = 0.1                                           # Mesh refinement tolerance
    MESH_REF_FRAC   = 0.1                                           # Mesh refinement fraction in marking
    MESH_REF_MDOFS  = 2*(240+1)*(120+1) + 2*240*120                 # OBSOLETE
    MESH_REF_NX_MAX = 60  * 3                                       # Number of uniformly refined mesh elements in x-direction         
    MESH_REF_NY_MAX = 120 * 3                                       # Number of uniformly refined mesh elements in y-direction

    # Mesh refinement global parameters
    parameters['allow_extrapolation'] = True                        # Parameters related to FEniCS refine function
    parameters['refinement_algorithm'] = 'plaza_with_parent_facets'
    
    # Verbose solver dialogs
    set_log_active(False)


    # Define point load configuration as a list of
    # ['location', 'magnitude', 'polar angle']
    # loadConf  = [[Point(LEN_X/2, 0.0), 1.0, 90.0]]
    loadConf = [[[LEN_X, 0.0*LEN_Y], 10.0, KLE_LOAD_MEAN]]    
    # loadConf = [[[LEN_X, 0.5*LEN_Y], KLE_LOAD_MEAN, 0.0]]      


    # Compute KLE parameters with given requirements
    KLE_TRUNC_N = kle_energy_check(KLE_N_MAX, KLE_CORR_LEN, KLE_ENERGY_REQ)
    kleParam    = [ kle_param(KLE_TRUNC_N, KLE_CORR_LEN, KLE_LOAD_MEAN, KLE_LOAD_STD), \
                    kle_param(KLE_TRUNC_N, KLE_CORR_LEN, KLE_E_MEAN, KLE_E_STD), \
                    kle_param(KLE_TRUNC_N, KLE_CORR_LEN, KLE_NU_MEAN, KLE_NU_STD)]


    # Initialize counters and flags
    iterMax  = 300 #int(2*NUM_ELEM_NX)
    iterNum  = 0
    iterLineSearch = 0
    stopCond = False


    # Uniform mesh on working domain
    meshDomX, meshDomY = np.meshgrid(np.linspace(0.0, LEN_X, NUM_ELEM_NX+1), np.linspace(0.0, LEN_Y, NUM_ELEM_NY+1))   
    mesh = RectangleMesh(Point(0.0, 0.0), Point(LEN_X, LEN_Y), NUM_ELEM_NX, NUM_ELEM_NY, MESH_DIAG)
    meshUnif = RectangleMesh(Point(0.0, 0.0), Point(LEN_X, LEN_Y), MESH_REF_NX_MAX, MESH_REF_NY_MAX, MESH_DIAG)


    # Set vector FEM space on mesh
    vectorFEMSpaceV = VectorFunctionSpace(mesh, 'Lagrange', 1)


    # Define Dirichlet boundaries
    # class DirichletBoundary1(SubDomain):

    #     def inside(self, x, on_boundary):

    #         # near(x[0], 0.0)
    #         return near(x[0], 0.0) and x[1] > EPS_GEOM + LEN_Y*2/3
        

    # class DirichletBoundary2(SubDomain):

    #     def inside(self, x, on_boundary):

    #         # near(x[0], 0.0)
    #         return near(x[0], 0.0) and near(x[1], LEN_Y)

    # Define Dirichlet boundaries
    class DirichletBoundary1(SubDomain):

        def inside(self, x, on_boundary):

            return abs(x[0]) < DOLFIN_EPS and abs(x[1] - 0.75*LEN_Y) < DOLFIN_EPS 
        

    class DirichletBoundary2(SubDomain):

        def inside(self, x, on_boundary):

            return abs(x[0]) < DOLFIN_EPS and abs(x[1] - LEN_Y) < DOLFIN_EPS 
        

    dirichletBoundary1 = DirichletBoundary1()
    dirichletBoundary2 = DirichletBoundary2()

    boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    boundaries.set_all(0)          
    dirichletBoundary1.mark(boundaries, 1)
    dirichletBoundary2.mark(boundaries, 2)  

    # Define boundary measure
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries) 


    # Set boundary conditions (half-wheel case is set)
    boundaryCond = [DirichletBC(vectorFEMSpaceV.sub(0), \
                                Constant(0.0), \
                                dirichletBoundary1, \
                                method='pointwise'), \
                    DirichletBC(vectorFEMSpaceV.sub(1), \
                                Constant(0.0), \
                                dirichletBoundary1, \
                                method='pointwise'), \
                    DirichletBC(vectorFEMSpaceV.sub(0), \
                                Constant(0.0), \
                                dirichletBoundary2, \
                                method='pointwise'), \
                    DirichletBC(vectorFEMSpaceV.sub(1), \
                                Constant(0.0), \
                                dirichletBoundary2, \
                                method='pointwise')]

        
    # Define level-set function
    # NOTE: Could LSF be defined on FunctionSpace(mesh, ...) with Function or Expression?
    # psiMatrix = - np.cos((6.0*pi*(meshDomX-0.0))) * np.cos(6.0*pi*(meshDomY-0.0)) - 0.1 \
    #             + np.minimum(5.0/LEN_Y * (meshDomY-1.0) + 5.0, 0.0) \
    #             + np.maximum(100.0 * ( meshDomX+meshDomY-LEN_X-LEN_Y+0.1), 0.0) \
    #             + np.maximum(100.0 * (-meshDomX+meshDomY-LEN_Y+0.1), 0.0)  
    psiMatrix = - np.cos(8.0/LEN_X*pi*meshDomX) * np.cos(4.0*pi*meshDomY) - 0.5 \
                # + np.maximum(100.0*(meshDomX+meshDomY-LEN_X-LEN_Y+0.1),.0) \
                # + np.maximum(100.0*(meshDomX-meshDomY-LEN_X+0.1),.0) \
                # + np.maximum(200.0*(0.01-meshDomX**2-(meshDomY-LEN_Y/2)**2),.0) 

    psiDotMatrixArray = []
    

    # Define Lame parameters
    # lameMu    = CONST_E / (2*(1+CONST_NU))
    # lameLam   = CONST_E*CONST_NU / ((1+CONST_NU)*(1-2*CONST_NU))


    # Set scalar FEM space and cost functionals
    scalarFEMSpaceV = FunctionSpace(mesh, 'Lagrange', 1)
    costFunc        = np.zeros(iterMax)
    volumeUnit      = project(Expression('1.0', degree=2), scalarFEMSpaceV)


    # Initialize solution vector for each load
    solutionVec     = [0]*MC_SAMPLE_N
    etaKVec         = [0]*MC_SAMPLE_N
    markedElems     = [0]*MC_SAMPLE_N
    goalVec         = [0]*MC_SAMPLE_N
    thetaVec        = [0]*MC_SAMPLE_N
    etaKAuxVec      = [0]*MC_SAMPLE_N
    goalVecAux      = [0]*MC_SAMPLE_N
    shapeDeriv      = [0]*MC_SAMPLE_N
    complianceVal   = [0]*MC_SAMPLE_N
    weights         = [0]*MC_SAMPLE_N


    # Get vertex coordinates
    totalDim    = mesh.geometry().dim()
    dofsScalarV = scalarFEMSpaceV.tabulate_dof_coordinates().reshape((-1, totalDim))
    dofsVectorV = vectorFEMSpaceV.tabulate_dof_coordinates().reshape((-1, totalDim))

    pxScalarV   = (dofsScalarV[:,0]/LEN_X) * (2*NUM_ELEM_NX)
    pyScalarV   = (dofsScalarV[:,1]/LEN_Y) * (2*NUM_ELEM_NY)

    pxVectorV   = (dofsVectorV[:,0]/LEN_X) * (2*NUM_ELEM_NX)
    pyVectorV   = (dofsVectorV[:,1]/LEN_Y) * (2*NUM_ELEM_NY)


    if MESH_DIAG == 'crossed':

        dofsMaxScalarV, dofsMaxVectorV = ((NUM_ELEM_NX+1)*(NUM_ELEM_NY+1) + NUM_ELEM_NX*NUM_ELEM_NY) * np.array([1,2])

    elif MESH_DIAG == 'left' or MESH_DIAG == 'right':
    
        dofsMaxScalarV, dofsMaxVectorV = ((NUM_ELEM_NX+1)*(NUM_ELEM_NY+1)) * np.array([1,2])

    else:

        error('Invalid direction for mesh diagonals!')


    # Initialize level-set function
    psiFunc = Function(scalarFEMSpaceV)
    psiFunc = compute_lsf(pxScalarV, pyScalarV, \
                          psiFunc, psiMatrix, \
                          dofsMaxScalarV)


    # Define domain and get initial shape
    class Omega(SubDomain):

        def inside(self, x, on_boundary):

            return 0.0 <= x[0] <= LEN_X and 0.0 <= x[1] <= LEN_Y and psiFunc(x) < 0.0

    domains = MeshFunction('size_t', mesh, mesh.topology().dim())


    # Define domain measure and unit outward normal
    dX     = Measure('dx', domain=mesh, subdomain_data=boundaries)
    nVec   = FacetNormal(mesh)

    # Compute volume of working domain
    VOL_TOTAL = LEN_X * LEN_Y


    # Measure starting time
    timeStart = time.time()

    
    # Initial visualization of shape
    pp.ion()
    pp.show()
    ax = pp.subplot()
    mngr = pp.get_current_fig_manager()
    mngr.window.setGeometry(PLOT_GEOM[0], PLOT_GEOM[1], PLOT_GEOM[2], PLOT_GEOM[3])
    ax.contourf(psiMatrix, np.arange((psiMatrix.min() - psiMatrix.max()), 0, abs(psiMatrix.min() - psiMatrix.max())/1000), \
                extent=[0.0, LEN_X, 0.0, LEN_Y], \
                cmap=cm._colormaps['coolwarm'])
    ax.set_aspect('equal', 'box')
    # plot(mesh)
    pp.draw()
    pp.pause(0.01)
    pp.savefig(folderName + '/iter_0' + '.png', bbox_inches='tight')


    # Main iteration loop
    lineSearchBeta0     = LS_BETA0_INIT
    lineSearchBeta      = lineSearchBeta0

    varBatchSize = MC_BATCH_N
    varBatchSizeArray = []

    errBoundArray, errBoundSolArray, errBoundAuxArray = [], [], []

    djMean, djVar, lipConstArray, lineSearchBetaArray = [], [], [], []
    shapeDerivPrev = 0.0

    totalDOFs = vectorFEMSpaceV.dim()
    linearSolver = PETScLUSolver()
    
    while iterNum < iterMax and stopCond == False: 

        timeLoop0 = time.time()
        meshRefineCond = True

        # Compute cost functional
        compCost = 0.0
        meshArray, meshDiams = [], []
        shapeDerivSampleArray = []

        shapeDerivMean = 0.0 # assemble(Form(Constant(0) * dX)) 
        shapeDerivVar = 0.0
        thetaVecMean = 0.0*thetaVec[0]

        # Initialize MC mean in each iteration
        # meanSolutionVec = []

        # Initialize penalty parameters
        if iterNum < 1:

            penalCostMu  = PENAL_MU_INIT

        # Compute volume cost
        volOmega = assemble(volumeUnit * dX(1))

        
        # Generate i.i.d. Gaussian random variables
        np.random.seed(iterNum)
        xiRandom = np.random.randn(KLE_TRUNC_N, MC_SAMPLE_N)


        while meshRefineCond:

            etaKMax, etahSolutionMean, etahShapeDerivMean, goalMean, goalAuxMean, complianceMean = [], 0.0, 0.0, 0.0, 0.0, 0.0

            for iterLoads in range(0, len(loadConf)):

                for iterMC in range(0, varBatchSize):

                    # Set vector FEM space on mesh
                    vectorFEMSpaceV = VectorFunctionSpace(mesh, 'Lagrange', 1)

                    dirichletBoundary1 = DirichletBoundary1()
                    dirichletBoundary2 = DirichletBoundary2()

                    boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
                    boundaries.set_all(0)          
                    dirichletBoundary1.mark(boundaries, 1)
                    dirichletBoundary2.mark(boundaries, 2)  

                    boundaryCond = [DirichletBC(vectorFEMSpaceV.sub(0), \
                                                Constant(0.0), \
                                                dirichletBoundary1, \
                                                method='pointwise'), \
                                    DirichletBC(vectorFEMSpaceV.sub(1), \
                                                Constant(0.0), \
                                                dirichletBoundary1, \
                                                method='pointwise'), \
                                    DirichletBC(vectorFEMSpaceV.sub(0), \
                                                Constant(0.0), \
                                                dirichletBoundary2, \
                                                method='pointwise'), \
                                    DirichletBC(vectorFEMSpaceV.sub(1), \
                                                Constant(0.0), \
                                                dirichletBoundary2, \
                                                method='pointwise')]
                    
                    domains = MeshFunction('size_t', mesh, mesh.topology().dim())

                
                    # Set current shape
                    omega = Omega()
                    domains.set_all(0)
                    omega.mark(domains, 1)
                    dX = Measure('dx', domain=mesh, subdomain_data=domains)
                    ds = Measure('ds', domain=mesh, subdomain_data=boundaries) 

                    # Compute volume cost
                    volOmega = assemble(volumeUnit * dX(1))

                    boundaryArray = [dirichletBoundary1, dirichletBoundary2]

                    # Solve PDE for each MC sample
                    print('     Solving main variational problem for sample ' + str(iterMC+1))

                    solutionVec[iterMC], \
                    markedElems[iterMC], \
                    etaKVec[iterMC], \
                    goalVec[iterMC], \
                    thetaVec[iterMC], \
                    etaKAuxVec[iterMC], \
                    goalVecAux[iterMC], \
                    shapeDeriv[iterMC], \
                    complianceVal[iterMC], \
                    weights[iterMC] = solve_weak_forms(vectorFEMSpaceV, dX, ds, \
                                                        CONST_EPS, \
                                                        boundaryCond, boundaryArray, \
                                                        loadConf[iterLoads], \
                                                        kleParam, xiRandom[:, iterMC], \
                                                        MESH_REF_FLAG, MESH_REF_FRAC, \
                                                        penalCostMu, \
                                                        VOL_FRAC, volOmega, VOL_TOTAL, \
                                                        [BFORM_ALPHA1, BFORM_ALPHA2], \
                                                        CASE_NAME)
                    

                    # Compute MC mean of error indicators
                    etahSolutionSample   = np.sum(etaKVec[iterMC])
                    etahShapeDerivSample = np.sum(etaKAuxVec[iterMC])
                    etahSolutionMean    += (1/varBatchSize) * etahSolutionSample
                    etahShapeDerivMean  += (1/varBatchSize) * etahShapeDerivSample
                    etahMean             = weights[iterMC][0] * etahSolutionMean + weights[iterMC][1] * etahShapeDerivMean
                    # etahMean             = np.sqrt(etahSolutionMean) + np.sqrt(etahShapeDerivMean)

                    # Compute MC mean of goal functions
                    goalMean    += (1/varBatchSize) * goalVec[iterMC]
                    goalAuxMean += (1/varBatchSize) * goalVecAux[iterMC]
                    
                    # Compute sample-wise max. error indicators
                    etaKMax.append(max(np.max(etaKVec[iterMC]), np.max(etaKAuxVec[iterMC])))

                    # Compute MC mean of cost function
                    complianceMean += (1/varBatchSize) * complianceVal[iterMC]

                    # print(' eta_max               : %.2f' % etaKMax[iterMC])
                    # print(' eta_sum               : %.2f' % etahSample)
                    # print(' eta_mean              : %.2f' % etahMean)

        
            # Capture the finest mesh size for maximum cell error indicator
            meshRefIdx = np.argmax(etaKMax)


            if MESH_REF_FLAG: 

                
                totalDOFs = VectorFunctionSpace(mesh, 'Lagrange', 1).dim()
                hmin = mesh.hmin()
                hminUnif = meshUnif.hmin()
                
                # Check if residual error is satisfied or maximum allowed DoFs are attained
                meshRefineCond = (etahMean > MESH_REF_TOL) and (hmin > hminUnif) # (totalDOFs < MESH_REF_MDOFS)

                if meshRefineCond:

                    mesh = refine(mesh, markedElems[meshRefIdx], redistribute=True)

            else:

                meshRefineCond = False
                
            print('     Total DOFs: %.0f' % totalDOFs)


        fineMesh = mesh
        mesh = RectangleMesh(Point(0.0, 0.0), Point(LEN_X, LEN_Y), NUM_ELEM_NX, NUM_ELEM_NY, MESH_DIAG)

        # Set vector FEM space on mesh
        vectorFEMSpaceV = VectorFunctionSpace(mesh, 'Lagrange', 1)

        dirichletBoundary1 = DirichletBoundary1()
        dirichletBoundary2 = DirichletBoundary2()

        boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        boundaries.set_all(0)          
        dirichletBoundary1.mark(boundaries, 1)
        dirichletBoundary2.mark(boundaries, 2)  

        boundaryCond = [DirichletBC(vectorFEMSpaceV.sub(0), \
                                    Constant(0.0), \
                                    dirichletBoundary1, \
                                    method='pointwise'), \
                        DirichletBC(vectorFEMSpaceV.sub(1), \
                                    Constant(0.0), \
                                    dirichletBoundary1, \
                                    method='pointwise'), \
                        DirichletBC(vectorFEMSpaceV.sub(0), \
                                    Constant(0.0), \
                                    dirichletBoundary2, \
                                    method='pointwise'), \
                        DirichletBC(vectorFEMSpaceV.sub(1), \
                                    Constant(0.0), \
                                    dirichletBoundary2, \
                                    method='pointwise')]
        
        domains = MeshFunction('size_t', mesh, mesh.topology().dim())

    
        # Set current shape
        omega = Omega()
        domains.set_all(0)
        omega.mark(domains, 1)
        dX = Measure('dx', domain=mesh, subdomain_data=domains)
        ds = Measure('ds', domain=mesh, subdomain_data=boundaries) 
        nVec = FacetNormal(mesh)

        # Compute volume cost
        volOmega = assemble(volumeUnit * dX(1))
        

        # Compute mean total cost
        costFunc[iterNum] = complianceMean + (penalCostMu/2) * (VOL_FRAC - volOmega / VOL_TOTAL)**2


        # Perform gradient descent iterations
        if False: # iterNum > 0 and costFunc[iterNum] > costFunc[iterNum-1] and iterLineSearch < LS_ITER_MAX:

            # Perform line-search via decreasing stepsizes
            iterLineSearch += 1
            lineSearchBeta *= LS_GAMMA1

            psiMatrix, psiFunc = [psiMatrixPrev, psiFuncPrev]

            # Solve HJE to generate LSF points
            psiMatrix, _ = solve_hje(thetaMatrix, psiMatrix, \
                                        LEN_X, LEN_Y, NUM_ELEM_NX, NUM_ELEM_NY, \
                                        lineSearchBeta)
            
            # Update current LSF
            psiFunc = compute_lsf(pxScalarV, pyScalarV, \
                                    psiFunc, psiMatrix, \
                                    dofsMaxScalarV)
            
            print('Line search iteration: %s' % iterLineSearch)

        else:

            print('\n')
            print('*** ' + CASE_NAME + ' ***')
            print('*** *** *** *** ITERATION NUMBER %s *** *** *** ***' % iterNum)                   
            print('Total cost value                 : %.4f' % costFunc[iterNum])
            print('Mean compliance value            : %.4f' % complianceMean)
            print('Mean goal value                  : %.4f' % goalMean)
            print('Mean aux. goal value             : %.4f' % goalAuxMean)
            print('Volume fraction                  : %.4f' % (volOmega / VOL_TOTAL)) 
            print('Mini-batch size                  : %s/%s' % (varBatchSize, MC_SAMPLE_N))
            
            # Set stepsize
            if iterLineSearch == LS_ITER_MAX:

                lineSearchBeta0 = max(lineSearchBeta0 * LS_GAMMA2, 0.1 * LS_BETA0_INIT)

            if iterLineSearch == 0:

                lineSearchBeta0 = min(lineSearchBeta0 / LS_GAMMA2, 1.0)


            # Reset stepsize and LS index
            iterLineSearch, lineSearchBeta, iterNum = [0, lineSearchBeta0, iterNum+1]

            # lineSearchBeta = LS_BETA0_INIT / iterNum


            # Compute batch mean of shape derivative and descent direction
            psiFuncPrev, psiMatrixPrev = [psiFunc, psiMatrix]
            psiMean = np.multiply(psiMatrix, 0)
            psiDotMean = np.multiply(psiMatrix, 0)

            # Set vector FEM space on refined mesh
            refinedFEMSpaceV = VectorFunctionSpace(fineMesh, 'Lagrange', 1)

            dirichletBoundary1 = DirichletBoundary1()
            dirichletBoundary2 = DirichletBoundary2()

            boundaries = MeshFunction('size_t', fineMesh, fineMesh.topology().dim()-1, 0)
            boundaries.set_all(0)          
            dirichletBoundary1.mark(boundaries, 1)
            dirichletBoundary2.mark(boundaries, 2)  

            boundaryCond = [DirichletBC(refinedFEMSpaceV.sub(0), \
                                        Constant(0.0), \
                                        dirichletBoundary1, \
                                        method='pointwise'), \
                            DirichletBC(refinedFEMSpaceV.sub(1), \
                                        Constant(0.0), \
                                        dirichletBoundary1, \
                                        method='pointwise'), \
                            DirichletBC(refinedFEMSpaceV.sub(0), \
                                        Constant(0.0), \
                                        dirichletBoundary2, \
                                        method='pointwise'), \
                            DirichletBC(refinedFEMSpaceV.sub(1), \
                                        Constant(0.0), \
                                        dirichletBoundary2, \
                                        method='pointwise')]
            
            domains = MeshFunction('size_t', fineMesh, fineMesh.topology().dim())

        
            # Set current shape
            omega = Omega()
            domains.set_all(0)
            omega.mark(domains, 1)
            dX = Measure('dx', domain=fineMesh, subdomain_data=domains)
            ds = Measure('ds', domain=fineMesh, subdomain_data=boundaries) 
            nVec = FacetNormal(fineMesh)
            
            

            print('     Averaging computed shape derivatives ...')

            for iterMC in range(0, varBatchSize):
                
                solutionVec[iterMC] = project(solutionVec[iterMC].leaf_node(), refinedFEMSpaceV)
                thetaVec[iterMC]    = project(thetaVec[iterMC].leaf_node(), refinedFEMSpaceV)
                
                shapeDerivVecSample = assemble(shapeDeriv[iterMC])

                # Average over mini batch
                shapeDerivMean += (1/varBatchSize) * shapeDerivVecSample
                shapeDerivVar  += (1/varBatchSize) * (shapeDerivVecSample*shapeDerivVecSample) \
                                    - shapeDerivMean*shapeDerivMean
                shapeDerivSampleArray.append(shapeDerivVecSample)

                thetaVecMean += (1/varBatchSize) * thetaVec[iterMC]
                
            thetaDum = thetaVecMean


            # Record shape derivative statistics
            djMean.append(norm(shapeDerivMean))
            djVar.append(norm(shapeDerivVar))


            # Estimate Lipschitz constant of shape deriv.
            thetaProj   = project(thetaDum, refinedFEMSpaceV)
            maxTheta = np.max(np.max(np.abs(thetaProj.vector().get_local())))

            if iterNum > 1:

                time_step = lineSearchBeta * min(LEN_X/NUM_ELEM_NX, LEN_Y/NUM_ELEM_NY) / maxTheta
                lipConst = sqrt(2*norm(shapeDerivMean) + 2*norm(shapeDerivPrev)) / (HJE_TIMESPAN*time_step*maxTheta*volOmega)
                lipConstArray.append(lipConst)

                print('\nCov: ', norm(shapeDerivVar), '\n')

                if MC_SAMPLE_N == 1:

                    lineSearchBeta = min(100.0, max(0.001, 1 / lipConst))

                else:

                    lineSearchBeta = min(100.0, max(0.001, 1 / ((1 + ADAPT_SAMP_NU**2 + ADAPT_SAMP_THE**2) * lipConst)))


            lineSearchBetaArray.append(lineSearchBeta)

            shapeDerivPrev = shapeDerivMean


            # # Compute norm of theta vector field
            # # Compute the gradient of the tensor field
            # gradTheta = grad(thetaDum)

            # # Create a TensorFunctionSpace to handle the second-order tensor field
            # # gradTheta is a tensor (2x2), so we need a TensorFunctionSpace for that
            # tensorFEMSpaceV = TensorFunctionSpace(fineMesh, 'P', 1)  # P1 tensor space for 2x2 tensor field

            # # Project gradient onto the TensorFunctionSpace
            # gradThetaProj   = project(gradTheta, tensorFEMSpaceV)
            # gradThetaVals   = gradThetaProj.vector().get_local()
            # numVertices     = fineMesh.num_vertices()
            # gradThetaComps  = gradThetaVals.reshape((numVertices, 2, 2))  # 2x2 tensor components

            # # Compute L-inf norm (maximum absolute row sum of the tensor)
            # lInfNorms = np.max(np.sum(np.abs(gradThetaComps), axis=2), axis=1)

            # # Compute global L-inf norm (the maximum across all vertices)
            # thetaTilde1 = np.max(lInfNorms)

            # # Compute H-1 norm norm
            # thetaTilde2 = norm(thetaDum, 'H1')
            

            # Compute and print theoretical combined error bound
            errBoundSol = etahSolutionMean
            errBoundAux = etahShapeDerivMean
            errBound    = etahMean

            errBoundSolArray.append(errBoundSol)
            errBoundAuxArray.append(errBoundAux)
            errBoundArray.append(errBound)

            print('Mean absolute goal  error        : %.4f' % etahSolutionMean)
            print('Mean absolute shape der. error   : %.4f' % etahShapeDerivMean)
            print('Combined error bound estimate    : %.4f' % errBound)


            thetaLSF = project(thetaDum, vectorFEMSpaceV)
            thetaVector = thetaLSF.vector().get_local()
            thetaMatrix = [np.zeros((NUM_ELEM_NY+1, NUM_ELEM_NX+1)), np.zeros((NUM_ELEM_NY+1, NUM_ELEM_NX+1))]
            
            for dof in range(0, dofsMaxVectorV, 2):

                if np.rint(pxVectorV[dof]) % 2 == 0.0:

                    cx, cy = np.int_(np.rint([pxVectorV[dof]/2, pyVectorV[dof]/2]))

                    thetaMatrix[0][cy, cx] = thetaVector[dof]
                    thetaMatrix[1][cy, cx] = thetaVector[dof+1]


            # Update LSF via theta vector field
            psiMatrix, _ = solve_hje(thetaMatrix, psiMatrix, \
                                        LEN_X, LEN_Y, NUM_ELEM_NX, NUM_ELEM_NY, \
                                        lineSearchBeta, HJE_TIMESPAN)
                            

            # Perform approximate augmented inner product test or norm test
            if varBatchSize > 1 and not(varBatchSize == MC_SAMPLE_N):

                print('     Testing MC sample size ... ')
                
                if ADAPT_SAMP_TYPE == 'inner':

                    isOrthogonalityValid, validBatchSizeOT = orthogonality_test_check(shapeDerivSampleArray, \
                                                                                        shapeDerivMean, \
                                                                                        varBatchSize, \
                                                                                        ADAPT_SAMP_NU)
                    
                    isInnerProductValid, validBatchSizeIT  = inner_product_test_check(shapeDerivSampleArray, \
                                                                                        shapeDerivMean, \
                                                                                        varBatchSize, \
                                                                                        ADAPT_SAMP_THE)
                    
                    if (not isOrthogonalityValid) or (not isInnerProductValid):
                        
                        if validBatchSizeOT == MC_SAMPLE_N or validBatchSizeIT == MC_SAMPLE_N:

                            print('     Using maximum allowed sample size. ')

                        else:

                            print('     Augmented IT test fails, sample size is updated. ')
                        
                        varBatchSize = min(ADAPT_SAMP_MAX, max(validBatchSizeOT, validBatchSizeIT))

                    else:

                        print('     Augmented IT test succeeds. ')


                elif ADAPT_SAMP_TYPE == 'norm':

                    isNormValid, validBatchSizeNT = norm_test_check(shapeDerivSampleArray, \
                                                                    shapeDerivMean, \
                                                                    varBatchSize, \
                                                                    ADAPT_SAMP_THE)

                    if not isNormValid:

                        if validBatchSizeNT == MC_SAMPLE_N:

                            print('     Using maximum allowed sample size. ')

                        else:

                            print('     Norm test fails, sample size is updated. ')
                        
                        varBatchSize = min(ADAPT_SAMP_MAX, validBatchSizeNT)

                    else:

                        print('     Norm test succeeds. ')

                else:

                    error('\n\n Invalid adaptive sampling strategy is entered! \n\n')

            else:

                print('     Using maximum allowed sample size. ')


            varBatchSizeArray.append(varBatchSize)


            # Reinitialize HJE periodically
            if np.mod(iterNum, 1) == 0: 

                psiMatrix = reinit_hje(psiMatrix, \
                                       LEN_X, LEN_Y, \
                                       NUM_ELEM_NX, NUM_ELEM_NY)     

                psiFunc = compute_lsf(pxScalarV, pyScalarV, \
                                      psiFunc, psiMatrix, \
                                      dofsMaxScalarV)
                

            # Update penalty parameters
            penalCostMu   = min(PENAL_MU_FACT * penalCostMu, PENAL_MU_MAX)


            # Stopping condition
            if iterNum > 20 and max(abs(costFunc[iterNum-6:iterNum-1] - costFunc[iterNum-1])) < EPS_TOL * costFunc[iterNum-1]: 

                stopCond = True


            # sigVMLSF = project(sigVM, scalarFEMSpaceV)

            # Visualize current shape
            if np.mod(iterNum, PLOT_PERIOD) == 0 or iterNum == 1 or iterNum == iterMax or stopCond == True:   
                
                pp.close()
                ax = pp.subplot()
                mngr = pp.get_current_fig_manager()
                mngr.window.setGeometry(PLOT_GEOM[0], PLOT_GEOM[1], PLOT_GEOM[2], PLOT_GEOM[3])
                ax.contourf(psiMatrix, np.arange((psiMatrix.min() - psiMatrix.max()), 0, abs(psiMatrix.min() - psiMatrix.max())/1000), \
                            extent=[0.0, LEN_X, 0.0, LEN_Y], \
                            cmap=cm._colormaps['coolwarm'])
                ax.set_aspect('equal', 'box')
                # plot(sigVMLSF)
                pp.draw()
                pp.pause(0.01)
                pp.savefig(folderName + '/iter_' + str(iterNum) + '.png', bbox_inches='tight')

                if MESH_REF_FLAG:

                    pp.close()
                    ax = pp.subplot()
                    mngr = pp.get_current_fig_manager()
                    mngr.window.setGeometry(PLOT_GEOM[0], PLOT_GEOM[1], PLOT_GEOM[2], PLOT_GEOM[3])
                    ax.set_aspect('equal', 'box')
                    plot_handles = plot(fineMesh)
                    for handle in plot_handles:
                        handle.set_linewidth(0.2)
                    # plot(thetaDum)
                    pp.draw()
                    pp.pause(0.01)
                    pp.savefig(folderName + '/mesh_' + str(iterNum) + '.png', bbox_inches='tight')


            # Measure individual loop time
            timeLoop1 = time.time() - timeLoop0
            print('     Est. remaining time         : %.2f [h]' % (timeLoop1*(iterMax - iterNum)/3600))


    # pp.figure()
    # mngr1 = pp.get_current_fig_manager()
    # mngr1.window.setGeometry(50, 100, 640, 480)
    # pp.plot(varBatchSizeArray, 'o')
    # pp.xlabel('iter')
    # pp.ylabel('mini-batch')
    # pp.savefig(folderName + '/batch_size' + '.png', bbox_inches='tight')
    # pp.show()

    # pp.figure()
    # mngr2 = pp.get_current_fig_manager()
    # mngr2.window.setGeometry(50, 100, 640, 480)
    # pp.plot(costFunc[0:iterNum-1], 'o')
    # pp.xlabel('iter')
    # pp.ylabel('cost')
    # pp.savefig(folderName + '/cost_func' + '.png', bbox_inches='tight')
    # pp.show()

    # pp.figure()
    # mngr2 = pp.get_current_fig_manager()
    # mngr2.window.setGeometry(50, 100, 640, 480)
    # pp.plot(errBoundArray, 'o')
    # pp.xlabel('iter')
    # pp.ylabel('error')
    # pp.savefig(folderName + '/error_bnd' + '.png', bbox_inches='tight')
    # pp.show()
    # pp.close()

    # Print total time elapsed
    timeTotal = time.time() - timeStart
    print('     Total comp. time      : %.2f [h]' % (timeTotal/3600))
    print(' ')


    # Save which optimization variables to be recorded
    with open(caseName + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([varBatchSizeArray, \
                     costFunc[0:iterNum-1], \
                     errBoundArray, \
                     lineSearchBetaArray, \
                     djMean, \
                     djVar, \
                     totalDOFs, \
                     (timeTotal/3600), \
                     errBoundSolArray, \
                     errBoundAuxArray], f)

    return



# Callbacks
def compute_lsf(px, py, psi, psiMat, dofsMax):
    
    # Compute LSF directly or interpolate between nodes
    for dof in range(0, dofsMax):

        if np.rint(px[dof]) % 2 == 0.0:

            cx, cy = np.int_(np.rint([px[dof]/2, py[dof]/2]))                                            
            psi.vector()[dof] = psiMat[cy, cx]

        else:

            cx, cy = np.int_(np.floor([px[dof]/2, py[dof]/2]))                      
            psi.vector()[dof] = 0.25 * (psiMat[cy, cx] + psiMat[cy+1,cx] \
                                + psiMat[cy, cx+1] + psiMat[cy+1, cx+1])
                
    return psi  


def solve_weak_forms(femSpace, dx, ds, epsErsatz, boundCond, boundArray, loadConf, kleParam, randVec, \
                     refineFlag, refineFrac, penalCostMu, volFrac, volOmega, volTotal, bFormAuxParams, \
                     CASE_NAME):

    # Define FEM spaces for main problem
    u, v = [TrialFunction(femSpace), TestFunction(femSpace)]

    # Define FEM spaces for aux. problem
    th, ph = [TrialFunction(femSpace), TestFunction(femSpace)]

    # Define normal vectors
    nVec   = FacetNormal(femSpace.mesh())


    # Compute KLE for Lame parameters
    # xp, yp = SpatialCoordinate(femSpace.mesh())
    xp  = femSpace.mesh().coordinates()[:, 0]
    yp  = femSpace.mesh().coordinates()[:, 1]

    lameLamVec, lameMuVec = kle_lame_param(x=xp, y=yp, kleParamE=kleParam[1], kleParamNu=kleParam[2], randVec=randVec)

    V0 = FunctionSpace(femSpace.mesh(), 'Lagrange', 1)
    lameLam, lameMu = [ Function(V0), Function(V0)]

    # Set the values of lambda and mu
    lameLam.vector()[:] = lameLamVec
    lameMu.vector()[:] = lameMuVec

    # print('\n', lameMu.compute_vertex_values(femSpace.mesh()), np.shape(lameMu.compute_vertex_values(femSpace.mesh())))
    # exit()


    # Collect KLE parameters for random load vector
    N   = kleParam[0]['N']
    std = kleParam[0]['std']
    ev  = kleParam[0]['ev']
    ef  = kleParam[0]['ef']
    ix  = kleParam[0]['ix']
    iy  = kleParam[0]['iy']
    kleTailLoad = 0.0

    for k in range(0, N):

        ii = ix[k]
        jj = iy[k]

        # Compute truncated KLE
        kleTailLoad += std * np.sqrt(ev[ii]*ev[jj]) * np.multiply(ef[ii](loadConf[0][0]), ef[jj](loadConf[0][1])) * randVec[k]
        
        # kleTailLoad = std * randVec[k]

    # kleTailLoad = std * np.random.rand()
    

    # Set bilinear form of main problem
    bform = (2.0*(lameMu)) * inner(sym(grad(u)), sym(grad(v))) + (lameLam) * div(u) * div(v)


    # Define load vector and impose Neumann BC as a list of
    # ['location', 'magnitude', 'polar angle']
    loadMag  = loadConf[1]                                              # Add KLE tail for random load magnitude
    loadAng  = loadConf[2] + kleTailLoad                                # Add KLE tail for random load angle
    loadPosx = DeltaX(eps=1E-11, 
                      x0=np.array([loadConf[0][0], loadConf[0][1]]), 
                      degree=1)                                         # Add KLE tail for random load x-position
    loadPosy = DeltaY(eps=1E-11, 
                      x0=np.array([loadConf[0][0], loadConf[0][1]]), 
                      degree=1)

    gx = Constant( loadMag * np.cos(loadAng * (pi/180))) * loadPosx
    gy = Constant(-loadMag * np.sin(loadAng * (pi/180))) * loadPosy


    # Define main variational problem
    a = epsErsatz * bform * dx(0) + bform * dx(1)
    L = inner(gx, v) * ds(0) + inner(gy, v) * ds(0) 

    # Solve main variational problem
    u = Function(femSpace)    
    solve(a == L, u, boundCond, solver_parameters={'linear_solver': 'mumps'})
    uh = u


    # Define aux. variational problem
    alpha1 = bFormAuxParams[0]
    alpha2 = bFormAuxParams[1]

    euh, Duh = [sym(grad(uh)), grad(uh)]

    sigh = (2*lameMu) * euh + lameLam * tr(euh) * Identity(2)
    phih = inner(sigh, euh)
    Sh  = 2 * Duh.T * sigh - phih * Identity(2)
    
    b  = (alpha1 * inner(grad(th), grad(ph)) + alpha2 * inner(th, ph)) * dx \
         + 1.0E11 * (inner(dot(th, nVec), dot(ph, nVec)) * (ds(0) + ds(1) + ds(2)))
    
    dJ = inner(Sh, grad(ph)) * dx(1) + epsErsatz * inner(Sh, grad(ph)) * dx(0)
    dV = Constant(-penalCostMu * (volFrac - volOmega / volTotal) / volTotal) * (div(ph) * dx(1))
    dL = dJ + dV

    # Solve aux. variational problem
    th = Function(femSpace)    
    solve(b == -dL, th, boundCond, solver_parameters={'linear_solver': 'mumps'})
    thh = th


    # Define goal functions (quantity of interest)
    u, th = [TrialFunction(femSpace), TrialFunction(femSpace)]

    sig = (2.0*lameMu) * sym(grad(u)) + lameLam * div(u) * Identity(2)
    eps = sym(grad(u))
    phi = inner(sig, eps) 


    # Define and compute goal function for main problem
    goal = inner(gx, v) * ds(0) + inner(gy, v) * ds(0) 

    uc = goal.arguments()[0]
    goalh = ufl.replace(goal, {uc: uh})
    goalVal = assemble(goalh)

    # Define and compute goal function for aux. problem
    dJAux = inner(Sh, grad(th)) * dx(1) + epsErsatz * inner(Sh, grad(th)) * dx(0)
    dVAux = Constant(-penalCostMu * (volFrac - volOmega / volTotal) / volTotal) * (div(th) * dx(1))
    goalAux = -(dJAux + dVAux)

    thc = goalAux.arguments()[0]
    goalhAux = ufl.replace(goalAux, {thc: thh})
    goalValAux = assemble(goalhAux)


    # Define and compute compliance cost
    cost = phi * dx(1) + epsErsatz * phi * dx(0) # inner(gx, u) * ds(0) + inner(gy, u) * ds(0)

    uc = cost.arguments()[0]
    costh = ufl.replace(cost, {uc: uh})
    costVal = assemble(costh)
    

    if refineFlag:

        # Solve adjoint problem manually (verified with dwrEstimator)
        # Construct enriched dual space and BCs
        e = femSpace.ufl_element()
        dualSpace = FunctionSpace(femSpace.mesh(), e.reconstruct(degree=e.degree() + 1))

        dualBoundCond = [DirichletBC(dualSpace.sub(0), \
                                     Constant(0.0), \
                                     boundArray[0], \
                                     method='pointwise'), \
                         DirichletBC(dualSpace.sub(1), \
                                     Constant(0.0), \
                                     boundArray[0], \
                                     method='pointwise'), \
                         DirichletBC(dualSpace.sub(0), \
                                     Constant(0.0), \
                                     boundArray[1], \
                                     method='pointwise'), \
                         DirichletBC(dualSpace.sub(1), \
                                     Constant(0.0), \
                                     boundArray[1], \
                                     method='pointwise')]
        
        z, w = [TestFunction(dualSpace), TrialFunction(dualSpace)]

        # Set bilinear form of adjoint PDE
        dform = (2.0*(lameMu)) * inner(sym(grad(z)), sym(grad(w))) + (lameLam) * div(z) * div(w)
        aStar = epsErsatz * dform * dx(0) + dform * dx(1)

        # Set RHS of adjoint PDE
        uc = goal.arguments()[0]
        MStar = ufl.replace(goal, {uc: w})

        # Solve adjoint variational problem
        z = Function(dualSpace)    
        solve(aStar == MStar, z, dualBoundCond, solver_parameters={'linear_solver': 'mumps'})
        zh = z


        # Evaluate the residual in the enriched space
        z = TestFunction(dualSpace)

        bformz = (2.0*(lameMu)) * inner(sym(grad(uh)), sym(grad(z))) + (lameLam) * div(uh) * div(z)
        az = epsErsatz * bformz * dx(0) + bformz * dx(1)
        Lz = inner(gx, z) * ds(0) + inner(gy, z) * ds(0) 

        res = Lz - az
        z = res.arguments()[0]

        # Break the function space into elements
        DG0 = FunctionSpace(dualSpace.mesh(), 'DG', 0)
        dg_0 = TestFunction(DG0)

        # Compute elementwise error indicators
        dwr = ufl.replace(res, {z: (zh - project(zh, femSpace))*dg_0})

        dwrVec = assemble(dwr)
        dwrVec.abs()
        eta = Function(DG0, dwrVec)

        etaVals = eta.vector().get_local()


        # Solve adjoint problem related to aux. problem
        psi, ksi = [TestFunction(dualSpace), TrialFunction(dualSpace)]

        # Set bilinear form of adjoint PDE
        nVec = FacetNormal(dualSpace.mesh())

        bStar = (alpha1 * inner(grad(psi), grad(ksi)) + alpha2 * inner(psi, ksi)) * dx \
                + 1.0E11 * (inner(dot(psi, nVec), dot(ksi, nVec)) * (ds(0) + ds(1) + ds(2)))

        # Set RHS of aux. adjoint PDE
        thc = goalAux.arguments()[0]
        dJStar = ufl.replace(goalAux, {thc: ksi})

        # Solve adjoint variational problem
        psi = Function(dualSpace)    
        solve(bStar == dJStar, psi, dualBoundCond, solver_parameters={'linear_solver': 'mumps'})
        psih = psi


        # Evaluate the aux. residual in the enriched space
        psi = TestFunction(dualSpace)

        bpsi = (alpha1 * inner(grad(thh), grad(psi)) + alpha2 * inner(thh, psi)) * dx \
               + 1.0E11 * (inner(dot(thh, nVec), dot(psi, nVec)) * (ds(0) + ds(1) + ds(2)))
        dJpsi = inner(Sh, grad(psi)) * dx(1) + epsErsatz * inner(Sh, grad(psi)) * dx(0)
        dVpsi = Constant(-penalCostMu * (volFrac - volOmega / volTotal) / volTotal) * (div(psi) * dx(1))
        Lpsi  = -(dJpsi + dVpsi)

        resAux = Lpsi - bpsi
        psi = resAux.arguments()[0]


        # Compute elementwise error indicators
        dwrAux = ufl.replace(resAux, {psi: (psih - project(psih, femSpace))*dg_0})

        dwrVecAux = assemble(dwrAux)
        dwrVecAux.abs()
        etaAux = Function(DG0, dwrVecAux)

        etaValsAux = etaAux.vector().get_local()


        # Compute elementwise markers
        mesh = dualSpace.mesh()

        # Put the values of the projection into a cell function
        etaCell = MeshFunction('double', mesh, mesh.topology().dim(), 0.0)

        # cinf    = np.linalg.norm(etaVals, np.inf)
        # dinf    = np.linalg.norm(etaValsAux, np.inf)

        # w1, w2 = 1 / cinf, 1 / dinf

        w1, w2 = np.abs(goalVal), np.abs(goalValAux)

        for c in cells(mesh):
            
            etaCell[c] = w1 * eta.vector()[c.index()] + w2 * etaAux.vector()[c.index()]


        # Dorfler mark elements
        markerFunc = FixedFractionMarkerParallel(frac=refineFrac)
        marks = markerFunc.mark(etaCell)

    else:

        marks, etaVals, etaValsAux = [], 0.0, 0.0

        w1, w2 = 0.0, 0.0

    return uh, marks, etaVals, goalVal, thh, etaValsAux, goalValAux, dL, costVal, [w1, w2]


def solve_hje(the, psi, lx, ly, nx, ny, beta, span):
    
    # Iterate HJE solution with upwind scheme N-times
    for i in range(span):

        Dym = ny * np.repeat(np.diff(psi, axis=0), [2]+[1]*(ny-1), axis=0) / ly
        Dyp = ny * np.repeat(np.diff(psi, axis=0), [1]*(ny-1)+[2], axis=0) / ly

        Dxm = nx * np.repeat(np.diff(psi), [2]+[1]*(nx-1), axis=1) / lx
        Dxp = nx * np.repeat(np.diff(psi), [1]*(nx-1)+[2], axis=1) / lx


        psiDot = 0.5 * (the[0] * (Dxp + Dxm) + the[1] * (Dyp + Dym)) \
                 - 0.5 * (np.abs(the[0]) * (Dxp - Dxm) + np.abs(the[1]) * (Dyp - Dym))


        # Check and satisfy CFL condition
        theMax = np.max(abs(the[0]) + abs(the[1]))
        # dt = beta*lx / (nx*theMax)
        dt = beta * min(lx/nx, ly/ny) / theMax
        
        # Iterate LSF on time
        psi = psi - dt * psiDot

    return psi, psiDot
    

def reinit_hje(psi, lx, ly, nx, ny):

    # Reinitialize HJE to avoid too steep or flat LSF
    Dxs = nx * (np.repeat(np.diff(psi), [2]+[1]*(nx-1), axis=1) \
          + np.repeat(np.diff(psi), [1]*(nx-1)+[2], axis=1)) / (2*lx) 
    Dys = ny * (np.repeat(np.diff(psi, axis=0), [2]+[1] * (ny-1), axis=0) \
          + np.repeat(np.diff(psi, axis=0), [1]*(ny-1)+[2], axis=0)) / (2*ly)
     
    sgn = psi / np.power(psi**2 + ((lx/nx)**2) * (Dxs**2 + Dys**2), 0.5)


    for i in range(0, 2):

        Dym = ny * np.repeat(np.diff(psi, axis=0), [2]+[1]*(ny-1), axis=0) / ly 
        Dyp = ny * np.repeat(np.diff(psi, axis=0), [1]*(ny-1)+[2], axis=0) / ly

        Dxm = nx*np.repeat(np.diff(psi), [2]+[1]*(nx-1), axis=1) / lx 
        Dxp = nx*np.repeat(np.diff(psi), [1]*(nx-1)+[2], axis=1) / lx    

        Kp  = np.sqrt((np.maximum(Dxm, 0))**2 + (np.minimum(Dxp, 0))**2 \
              + (np.maximum(Dym, 0))**2 + (np.minimum(Dyp, 0))**2)
        Km  = np.sqrt((np.minimum(Dxm, 0))**2 + (np.maximum(Dxp, 0))**2 \
              + (np.minimum(Dym, 0))**2 + (np.maximum(Dyp, 0))**2)       
            

        psiDot  = np.maximum(sgn, 0) * Kp + np.minimum(sgn, 0) * Km


        # Iterate LSF on time
        psi  = psi - (0.5*lx/nx) * (psiDot - sgn)

    return psi


def kle_gauss(N, d, l):

    # Compute Karhunen-Loeve expansion of Gaussian
    # i.i.d. random variables on [-d, d] with
    # correlation length, l.
    def funcEven(omg, d, l):

        return (1/l) * np.tan(omg * d) + omg
    
    def funcOdd(omg, d, l):

        return (1/l) - np.multiply(omg, np.tan(np.multiply(omg, d)))
    

    omgs  = np.zeros(N)
    shift = 1E-6
    ind   = 0


    for i in range(0, ceil(N/2)+1):

        # Interval where both funcEven and funcOdd
        # have a zero
        interv = [max((2*i-1)*pi/(2*d) + shift, 0), (2*i+1)*pi/(2*d) - shift]

        if i > 0 and (2*i) <= N:

            sol  = root_scalar(funcEven, args=(d, l), bracket=interv)
            ind += 1
            
            omgs[ind-1] = sol.root

        if (2*i+1) <= N:

            sol  = root_scalar(funcOdd, args=(d, l), bracket=interv)
            ind += 1
            
            omgs[ind-1] = sol.root
    

    # Compute KLE eigenvalues
    eigVals = np.divide(2/l, np.multiply(omgs, omgs) + l**(-2))


    # Compute KLE eigenfunctions
    # and derivatives
    eigFuncs  = []
    eigDerivs = []

    for i in range(1, N+1):

        oddBool = np.mod(i-1, 2)
        
        if oddBool:

            # Eigenfunctions for eigenvalues of odd n
            eigFuncsHandle = lambda z, ii=i, dd=d: np.sin(omgs[ii-1]*z) \
                                                    / (np.sqrt(dd - np.sin(2*omgs[ii-1]*dd)/(2*omgs[ii-1])))

            # Derivatives for eigenvalues of odd n
            eigDerivsHandle = lambda z, ii=i, dd=d: (omgs[ii-1]*np.cos(omgs[ii-1]*z)) \
                                                    / (np.sqrt(dd - np.sin(2*omgs[ii-1]*dd)/(2*omgs[ii-1])))

        else:
            
            # Eigenfunctions for eigenvalues of even n
            eigFuncsHandle = lambda z, ii=i, dd=d: np.cos(omgs[ii-1]*z) \
                                                    / (np.sqrt(dd + np.sin(2*omgs[ii-1]*dd)/(2*omgs[ii-1])))

            # Derivatives for eigenvalues of even n
            eigDerivsHandle = lambda z, ii=i, dd=d: (-omgs[ii-1]*np.sin(omgs[ii-1]*z)) \
                                                    / (np.sqrt(dd + np.sin(2*omgs[ii-1]*dd)/(2*omgs[ii-1])))


        eigFuncs.append(eigFuncsHandle)
        eigDerivs.append(eigDerivsHandle)
    

    return eigVals, eigFuncs, eigDerivs


def kle_param(N, corrLength, mu, std):

    # Assign KLE parameters on pre-defined domain
    # [-d, d] with given truncation number, correlation
    # length and standart deviation
    ev, ef, _ = kle_gauss(N, 2.0, corrLength)


    # Compute indices
    temp = np.outer(ev, ev)
    evec = temp.flatten('F')

    ind  = np.argsort(evec)
    ind  = ind[::-1]
    
    subInd = np.unravel_index(ind, np.shape(temp))

    indx = subInd[0]
    indy = subInd[1]


    # Assign KLE parameters
    kleParam = {
                'N'  : N,    \
                'mu' : mu,   \
                'std': std,  \
                'ev' : ev,   \
                'ef' : ef,   \
                'ix' : indx, \
                'iy' : indy, \
               }

    return kleParam


def kle_energy_check(Nmax, corrLength, energyRateReq):

    # Determine KLE truncation number based on
    # the required energy capture
    ev, _, _ = kle_gauss(Nmax, 1, corrLength)
    evSumTot = np.sum(ev)


    # Initialize energy rate check
    energyRate = 0.0
    cnt = 0
    N   = 10

    while energyRate < energyRateReq:

        # Compute current energy rate
        evSumCurr  = np.sum(ev[0:N])
        energyRate = evSumCurr / evSumTot

        # Increase N if condition is not met
        if energyRate < energyRateReq:

            N = ceil(1.1 * N)

        elif energyRate > energyRateReq + 5E-3:

            N = ceil(0.9 * N)

        cnt += 1


    print('\n KLE with N =', N, 'number of terms, ', int(100*energyRate), \
          '%', 'of total energy with Nmax =', Nmax, 'is captured. \n')

    return N


def orthogonality_test_check(gradSampleArray, gradAverage, batchSize, nu):

    # Check whether orthogonality test is passed
    coef = 1 / (batchSize - 1) / batchSize
    lhs = 0.0
    gs = np.reshape(gradAverage, -1)
    normgs = np.linalg.norm(gs)


    for i in range(0, len(gradSampleArray)):

        gi = np.reshape(gradSampleArray[i], -1)
        lhs += coef * np.linalg.norm(gi - np.multiply((np.dot(gi, gs) / normgs**2), gs))**2


    rhs = (nu**2) * normgs**2

    batchSizeReq = ceil(lhs / rhs * batchSize)

    return (lhs <= rhs), batchSizeReq


def inner_product_test_check(gradSampleArray, gradAverage, batchSize, the):

    # Check whether inner product test is passed
    coef = 1 / (batchSize - 1) / batchSize
    lhs = 0.0
    gs = np.reshape(gradAverage, -1)
    normgs = np.linalg.norm(gs)


    for i in range(0, len(gradSampleArray)):

        gi = np.reshape(gradSampleArray[i], -1)
        lhs += coef * (np.dot(gi, gs) - normgs**2)**2


    rhs = (the**2) * normgs**4

    batchSizeReq = ceil(lhs / rhs * batchSize)

    return (lhs <= rhs), batchSizeReq


def norm_test_check(gradSampleArray, gradAverage, batchSize, the):

    # Check whether norm test is passed
    coef = 1 / (batchSize - 1) / batchSize
    lhs = 0.0
    gs = np.reshape(gradAverage, -1)
    normgs = np.linalg.norm(gs)


    for i in range(0, len(gradSampleArray)):

        gi = np.reshape(gradSampleArray[i], -1)
        lhs += coef * np.linalg.norm(gi - gs)**2


    rhs = (the**2) * normgs**2

    batchSizeReq = ceil(lhs / rhs * batchSize)

    return (lhs <= rhs), batchSizeReq


class DeltaX(UserExpression):

    # Define smoothed Dirac delta on 2D case for x-direction
    def __init__(self, eps, x0, **kwargs):

        self.eps = eps
        self.x0 = x0
        UserExpression.__init__(self, **kwargs) 

    def eval(self, values, x):

        eps = self.eps
        values[0] = eps/np.sqrt(np.linalg.norm(x-self.x0)**2 + eps**2)
        values[1] = 0

    def value_shape(self): return (2, )


class DeltaY(UserExpression):

    # Define smoothed Dirac delta on 2D case for y-direction
    def __init__(self, eps, x0, **kwargs):

        self.eps = eps
        self.x0 = x0
        UserExpression.__init__(self, **kwargs) 

    def eval(self, values, x):

        eps = self.eps
        values[0] = 0
        values[1] = eps/np.sqrt(np.linalg.norm(x-self.x0)**2 + eps**2)

    def value_shape(self): return (2, )


class RandomField(UserExpression):

    # Construct KLE for random field-type variables
    def __init__(self, kleParam, randVec, **kwargs):

        self.N   = kleParam['N']
        self.mu  = kleParam['mu']
        self.std = kleParam['std']
        self.ev  = kleParam['ev']
        self.ef  = kleParam['ef']
        self.ix  = kleParam['ix']
        self.iy  = kleParam['iy']

        self.rv  = randVec

        UserExpression.__init__(self, **kwargs) 

    def eval(self, values, x):

        for k in range(0, self.N):

            ii = self.ix[k]
            jj = self.iy[k]

            # Compute truncated KLE
            values += self.std * np.sqrt(self.ev[ii]*self.ev[jj]) \
                        * np.multiply(self.ef[ii](x[0]), self.ef[jj](x[1])) * self.rv[k]
            
        values += self.mu

    def value_shape(self): return (1, )


class RandomFieldLambda(UserExpression):

    # Construct KLE for Lame random fields
    def __init__(self, kleParamE, kleParamNu, randVec, **kwargs):

        # KLE parameters for Young modulus (E)
        self.NE   = kleParamE['N']
        self.muE  = kleParamE['mu']
        self.stdE = kleParamE['std']
        self.evE  = kleParamE['ev']
        self.efE  = kleParamE['ef']
        self.ixE  = kleParamE['ix']
        self.iyE  = kleParamE['iy']

        # KLE parameters for Poisson ratio (nu)
        self.NNu   = kleParamNu['N']
        self.muNu  = kleParamNu['mu']
        self.stdNu = kleParamNu['std']
        self.evNu  = kleParamNu['ev']
        self.efNu  = kleParamNu['ef']
        self.ixNu  = kleParamNu['ix']
        self.iyNu  = kleParamNu['iy']

        self.rv  = randVec

        UserExpression.__init__(self, **kwargs) 

    def eval(self, values, x):

        # KLE for Young modulus
        Ew = self.muE
        
        for k in range(0, self.NE):

            ii = self.ixE[k]
            jj = self.iyE[k]

            # Compute truncated KLE
            Ew += self.stdE * np.sqrt(self.evE[ii]*self.evE[jj]) \
                        * np.multiply(self.efE[ii](x[0]), self.efE[jj](x[1])) * self.rv[k]
            
        # KLE for Poisson ratio
        nuw = self.muNu
        
        for k in range(0, self.NNu):

            ii = self.ixNu[k]
            jj = self.iyNu[k]

            # Compute truncated KLE
            nuw += self.stdNu * np.sqrt(self.evNu[ii]*self.evNu[jj]) \
                        * np.multiply(self.efNu[ii](x[0]), self.efNu[jj](x[1])) * self.rv[k]
            
        # Lame parameter lambda
        values += Ew*nuw / ((1+nuw)*(1-2*nuw))

    def value_shape(self): return (1, )


class RandomFieldMu(UserExpression):

    # Construct KLE for Lame random fields
    def __init__(self, kleParamE, kleParamNu, randVec, **kwargs):

        # KLE parameters for Young modulus (E)
        self.NE   = kleParamE['N']
        self.muE  = kleParamE['mu']
        self.stdE = kleParamE['std']
        self.evE  = kleParamE['ev']
        self.efE  = kleParamE['ef']
        self.ixE  = kleParamE['ix']
        self.iyE  = kleParamE['iy']

        # KLE parameters for Poisson ratio (nu)
        self.NNu   = kleParamNu['N']
        self.muNu  = kleParamNu['mu']
        self.stdNu = kleParamNu['std']
        self.evNu  = kleParamNu['ev']
        self.efNu  = kleParamNu['ef']
        self.ixNu  = kleParamNu['ix']
        self.iyNu  = kleParamNu['iy']

        self.rv  = randVec

        UserExpression.__init__(self, **kwargs) 

    def eval(self, values, x):

        # KLE for Young modulus
        Ew = self.muE
        
        for k in range(0, self.NE):

            ii = self.ixE[k]
            jj = self.iyE[k]

            # Compute truncated KLE
            Ew += self.stdE * np.sqrt(self.evE[ii]*self.evE[jj]) \
                        * np.multiply(self.efE[ii](x[0]), self.efE[jj](x[1])) * self.rv[k]
            
        # KLE for Poisson ratio
        nuw = self.muNu
        
        for k in range(0, self.NNu):

            ii = self.ixNu[k]
            jj = self.iyNu[k]

            # Compute truncated KLE
            nuw += self.stdNu * np.sqrt(self.evNu[ii]*self.evNu[jj]) \
                        * np.multiply(self.efNu[ii](x[0]), self.efNu[jj](x[1])) * self.rv[k]
            
        # Lame parameter mu
        values += Ew / (2*(1+nuw))

    def value_shape(self): return (1, )


def kle_lame_param(x, y, kleParamE, kleParamNu, randVec):

    # KLE parameters for Young modulus (E)
    NE   = kleParamE['N']
    muE  = kleParamE['mu']
    stdE = kleParamE['std']
    evE  = kleParamE['ev']
    efE  = kleParamE['ef']
    ixE  = kleParamE['ix']
    iyE  = kleParamE['iy']

    # KLE parameters for Poisson ratio (nu)
    NNu   = kleParamNu['N']
    muNu  = kleParamNu['mu']
    stdNu = kleParamNu['std']
    evNu  = kleParamNu['ev']
    efNu  = kleParamNu['ef']
    ixNu  = kleParamNu['ix']
    iyNu  = kleParamNu['iy']

    rv  = randVec

    # KLE for Young modulus
    Ew = muE
    
    for k in range(0, NE):

        ii = ixE[k]
        jj = iyE[k]

        # Compute truncated KLE
        Ew += stdE * np.sqrt(evE[ii]*evE[jj]) \
                    * np.multiply(efE[ii](x), efE[jj](y)) * rv[k]
        
    # KLE for Poisson ratio
    nuw = muNu
    
    for k in range(0, NNu):

        ii = ixNu[k]
        jj = iyNu[k]

        # Compute truncated KLE
        nuw += stdNu * np.sqrt(evNu[ii]*evNu[jj]) \
                    * np.multiply(efNu[ii](x), efNu[jj](y)) * rv[k]
        
    # Lame parameter lambda
    lamw = Ew*nuw / ((1+nuw)*(1-2*nuw))

    # Lame parameter mu
    muw = Ew / (2*(1+nuw))

    return lamw, muw


# Run code
if __name__ == '__main__':

    # Name array for recording data
    CASE_NAME_ARRAY = ['TEST_CASE_NAME']

    for i in range(0, len(CASE_NAME_ARRAY)):

        _main(CASE_NAME_ARRAY[i])
