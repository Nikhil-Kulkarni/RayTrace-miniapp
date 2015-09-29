This is the XRayTrace miniapp.  It is a miniapp that contains the ray-propagation portion of a 3D coupled atomic-physics/ray-propagation code used to simulate ASE (Amplified Spontaneous Emission) and seeded X-ray lasers [1-2].  The purpose of this miniapp is to test new ideas and programming models in a lightweight application that perfectly mirrors the behavior of the ray propagation kernel in the full application.  It solves the 3D ray and amplification equations, and is auto-generated from the application source code and therefore is always current with the application in use. 

[1] Berrill, Mark Allen. Modeling of laser-created plasmas and soft x-ray lasers. PhD. dissertation, Colorado State University (2010).
[2] Y. Wang, E. Granados, F. Pedaci, D. Alessi, B. Luther, M. Berrill, J. Rocca, "Phase-coherent, injection-seeded, table-top soft-X-ray lasers at 18.9 nm and 13.9 nm", Nature Photonics, 2, 94-98 (2008).


This folder contains miniapps that replicates the general behavior of the RayTrace program.
All miniapps are designed to be light weight with a minimal set of dependencies that can be 
easily compiled in a stand alone environment. 
Note that this is an independent application that can be deployed.

Currently the miniapps are:

1) CreateImage - Tests the behavior of the create_image function and the ray-tracing portion of the code.  This does not run any of the code related to the atomic physics, IO, converting data between grids, etc.
To create the data set (requires full code):
    generateCreateImageData <result_file> <length> <time> test.dat
To run the miniapp
    CreateImage test.dat


Building:
The build directory has several different example configure scripts.  A basic configure script is:
    cmake                               \
       -D CMAKE_BUILD_TYPE="Release"    \
       -D CMAKE_C_COMPILER=mpicc        \
       -D CMAKE_CXX_COMPILER=mpixx      \
          -D CFLAGS=""                  \
          -D CXXFLAGS=""                \
          -D CXX_STD=98                 \
       -D USE_OPENACC=0                 \
       -D USE_CUDA=1                    \
          -D CUDA_FLAGS="-arch sm_35"   \
          -D CUDA_HOST_COMPILER="/usr/bin/gcc" \
       -D USE_KOKKOS=0                  \
          -D KOKKOS_DIRECTORY=~/kokkos  \
          -D KOKKOS_WRAPPER=~/kokkos/nvcc_wrapper \
       -D PREFIX="Path to desired install" \
       ../src
Where CMAKE_BUILD_TYPE is build type ("Debug" or "Release"), CMAKE_C_COMPILER and CMAKE_CXX_COMPILER are the C and C++ compilers, CFLAGS and CXXFLAGS sets user-defined compiler flags (such as "-h pragma=acc,msgs" for Cray with OpenACC), (CXX_STD is the C++ standard to use (98,11,14), USE_OPENACC, USE_CUDA and USE_KOKKOS turn on/off the options for the miniapp, CUDA_FLAGS sets user-defined flags for Cuda, CUDA_HOST_COMPILER sets the host compiler for Cuda, KOKKOS_DIRECTORY sets the install path for Kokkos, and KOKKOS_WRAPPER poitns to the Kokkos nvcc_wrapper (if kokkos was compiled with cuda).  PREFIX is an optional path to install the miniapp, it will default to the build directory if not set.  Note that if you are compiling with OpenACC support, Cuda and Kokkos must be disabled.  Cuda and Kokkos can be enabled together.  


Running:
The CreateImage miniapp will be installed in the bin folder and takes one argument which is the input file.  It will then run all tests for that input.  All of the inputs represent actual runs of the full application in a production environment.  The problem size is representative of the work that a single node will receive.  The main application uses MPI across nodes, and the miniapp inputs contain the work that a single node gets (typically rank 0).  In production the fully application also handles threading across the cores, so while the serial time is printed for the miniapp the actual application will be ~ N times faster than the serial speed where N is the number of physical cpu cores available.  This is done in the full application using threads with independent variables.  This is not yet fully represented in the miniapp, but comparisons to serial are still useful.  The miniapp reports the performance of create_image routine once per configuration.  The full application will loop over length, time, and convergence calling this routine taking a much longer time, but still allowing this miniapp to measure the same performance as the actual application.  Note that any data structures, memory copies, etc that are in the RayTrace::create_image interface change every iteration and cannot be cached.

The inputs available are:
  ASE_small.dat - This is a small single node ASE (Amplified Spontaneous Emission) calculation that is typically done on a workstation with 8-32 cores.
  ASE_medium.dat - This is a medium ASE calculation that is typically done on a single workstation with 32 cores.
  seed_small.dat - This is a small single node seeded calculation that is similar to ASE_small.  Note that this benchmark takes ~10 longer than the ASE benchmark.  In the actual application seeded calculations require ~ 10x the number of rays, but 1/4th the number of iterations.  So seeded calculations typically take ~2x as long in the final application.  The miniapp compares the performance of a single iteration.  
  seed_medium.dat - This is the seeded version of ASE_medium.dat





