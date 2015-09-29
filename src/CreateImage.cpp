// This is a mini-application to mimics part of the full Ray-Trace code
// This miniapp mimics the behavior of create_image

#include "RayTrace.h"
#include <iostream>
#include <stdint.h>
#include <math.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    #include <windows.h>
    #define get_time(x) QueryPerformanceCounter(x)
    #define get_frequency(f) QueryPerformanceFrequency(f)
    #define get_diff(start,end,f) \
        static_cast<double>(end.QuadPart-start.QuadPart)/static_cast<double>(f.QuadPart)
    #define TIME_TYPE LARGE_INTEGER
#else
    #include <sys/time.h>
    #define get_time(x) gettimeofday(x,NULL);
    #define get_frequency(f) (*f=timeval())
    #define get_diff(start,end,f) 1e-6*static_cast<double>( \
        0xF4240*(static_cast<int64_t>(end.tv_sec)-static_cast<int64_t>(start.tv_sec)) + \
                (static_cast<int64_t>(end.tv_usec)-static_cast<int64_t>(start.tv_usec)) )
    #define TIME_TYPE timeval
#endif

#ifdef USE_KOKKOS
    #include <Kokkos_Core.hpp>
#endif


inline void fread2( void *ptr, size_t size, size_t count, FILE *fid )
{
    size_t N = fread(ptr,size,count,fid);
    if ( N != count ) {
        std::cerr << "Failed to read desired count\n";
        exit(-1);
    }
}


inline int check_ans( const double *image0, const double *I_ang0,
    const RayTrace::create_image_struct& data )
{
    size_t N_image = data.euv_beam->nx*data.euv_beam->ny*data.euv_beam->nv;
    size_t N_ang   = data.euv_beam->na*data.euv_beam->nb;
    double error[2]={0,0};
    double norm0[2]={0,0};
    double norm1[2]={0,0};
    for (size_t i=0; i<N_image; i++) {
        error[0] += (image0[i]-data.image[i])*(image0[i]-data.image[i]);
        norm0[0] += image0[i]*image0[i];
        norm1[0] += data.image[i]*data.image[i];
    }
    for (size_t i=0; i<N_ang; i++) {
        error[1] += (I_ang0[i]-data.I_ang[i])*(I_ang0[i]-data.I_ang[i]);
        norm0[1] += I_ang0[i]*I_ang0[i];
        norm1[1] += data.I_ang[i]*data.I_ang[i];
    }
    norm0[0] = sqrt(norm0[0]);
    norm0[1] = sqrt(norm0[1]);
    norm1[0] = sqrt(norm1[0]);
    norm1[1] = sqrt(norm1[1]);
    error[0] = sqrt(error[0])/norm0[0];
    error[1] = sqrt(error[1])/norm0[1];
    const double tol = 5e-6;    // RayTrace uses single precision for some calculations (may need to adjust to 1e-5)
    //bool pass = error[0]<=tol && error[1]<=tol;
    bool pass = (norm0[0]-norm1[0])/norm0[0]<=tol && (norm0[1]-norm1[1])/norm0[1]<=tol;
    if ( !pass ) {
        std::cerr << "  Answers do not match:" << std::endl;
        std::cerr << "    image: " << error[0] << " " << norm0[0] << " " << norm1[0] << std::endl;
        std::cerr << "    I_ang: " << error[1] << " " << norm0[1] << " " << norm1[1] << std::endl;
    }
    return pass ? 0:1;
}



/******************************************************************
* The main program                                                *
******************************************************************/
int main(int argc, char *argv[]) 
{

    // Initialize kokkos
    #ifdef USE_KOKKOS
        /*int argc2 = 1;
        const char *argv2[10] = {NULL};
        argv2[0] = argv[0];
        #ifdef KOKKOS_HAVE_PTHREAD
            argv2[argc2] = "--kokkos-threads=4";
            argc2++;
        #endif
        Kokkos::initialize(argc2,(char**)argv2);*/
        Kokkos::initialize(argc,argv);
    #endif

    // Check the input arguments
    if ( argc != 2 ) {
        std::cerr << "CreateImage called with the wrong number of arguments\n";
        return -1;
    }

    // load the input file
    FILE *fid = fopen(argv[1],"rb");
    uint64_t N_bytes = 0;
    fread2(&N_bytes,sizeof(uint64_t),1,fid);
    char *data = new char[N_bytes];
    fread2(data,sizeof(char),N_bytes,fid);
    fclose(fid);
    
    // Create the image structure
    RayTrace::create_image_struct info;
    info.unpack(std::pair<char*,size_t>(data,N_bytes));
    delete [] data;  data = NULL;
    const double *image0 = info.image;
    const double *I_ang0 = info.I_ang;
    info.image = NULL;
    info.I_ang = NULL;

    // Get the list of methods to try
    std::vector<std::string> methods;
    methods.push_back("cpu");
    #if CXX_STD==11 || CXX_STD==14
        methods.push_back("threads");
    #endif
    #ifdef USE_CUDA
        methods.push_back("Cuda");
    #endif
    #ifdef USE_OPENACC
        methods.push_back("OpenAcc");
    #endif
    #ifdef USE_KOKKOS
        methods.push_back("Kokkos-Serial");
        #ifdef KOKKOS_HAVE_PTHREAD
            //methods.push_back("Kokkos-Thread");
        #endif
        #ifdef KOKKOS_HAVE_OPENMP
            methods.push_back("Kokkos-OpenMP");
        #endif
        #ifdef KOKKOS_HAVE_CUDA
            methods.push_back("Kokkos-Cuda");
        #endif
    #endif

    // Call create_image for each method
    int N_errors = 0;
    std::vector<double> time(methods.size());
    for (size_t i=0; i<methods.size(); i++) {
        printf("Running %s\n",methods[i].c_str());
        TIME_TYPE start, stop, f;
        get_frequency(&f);
        get_time(&start);
        RayTrace::create_image(&info,methods[i]);
        get_time(&stop);
        time[i] = get_diff(start,stop,f);
        // Check the results
        N_errors += check_ans( image0, I_ang0, info );
        free((void*)info.image);
        free((void*)info.I_ang);
        info.image = NULL;
        info.I_ang = NULL;
    }
    printf("\n      METHOD     TIME\n");
    for (size_t i=0; i<methods.size(); i++)
        printf("%14s   %0.3f\n",methods[i].c_str(),time[i]);

    // Free memory and return
    free((void*)image0);
    free((void*)I_ang0);
    delete info.euv_beam;
    delete info.seed_beam;
    delete [] info.gain;
    delete info.seed;
    if ( N_errors == 0 )
        std::cout << "All tests passed\n";
    else
        std::cout << "Some tests failed\n";
    #ifdef USE_KOKKOS
        Kokkos::finalize ();
    #endif
    return N_errors;
}


