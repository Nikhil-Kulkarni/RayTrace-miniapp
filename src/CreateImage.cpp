// This is a mini-application to mimics part of the full Ray-Trace code
// This miniapp mimics the behavior of create_image

#include "RayTrace.h"
#include "MPI_helpers.h"
#include "CreateImageHelpers.h"
#include "utilities/RayUtilities.h"

#include <cstring>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdint.h>
#include <string>
#include <algorithm>



#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#endif


// Load the input file
static RayTrace::create_image_struct* loadInput( const std::string& filename,
    double scale, double **image0=NULL, double **I_ang0=NULL )
{
    // Load the input file
    FILE *fid = fopen( filename.c_str(), "rb" );
    if ( fid == NULL ) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return NULL;
    }
    uint64_t N_bytes = 0;
    fread2( &N_bytes, sizeof( uint64_t ), 1, fid );
    char *data = new char[N_bytes];
    fread2( data, sizeof( char ), N_bytes, fid );
    fclose( fid );

    // Create the image structure
    auto info = new RayTrace::create_image_struct();
    info->unpack( std::pair<char *, size_t>( data, N_bytes ) );
    delete[] data;
    if ( image0 != NULL )
        *image0 = info->image;
    else
        delete[] info->image;
    info->image = NULL;
    if ( I_ang0 != NULL )
        *I_ang0 = info->I_ang;
    else
        delete[] info->I_ang;
    info->I_ang = NULL;
    if ( scale != 1.0 )
        scale_problem( *info, scale );
    return info;
}


// Free the structure
static inline void free2( RayTrace::create_image_struct* info )
{
    if ( info == NULL )
        return;
    if ( info->image !=NULL ) {
        free( (void *) info->image );
        info->image = NULL;
    }        
    if ( info->I_ang !=NULL ) {
        free( (void *) info->I_ang );
        info->I_ang = NULL;
    }        
    delete info->euv_beam;
    delete info->seed_beam;
    delete[] info->gain;
    delete info->seed;
    delete info;
}



// Run the tests for a single file
int run_tests( const std::string& filename, const Options& options )
{
    if ( rank() == 0 )
        printf( "\nRunning tests for %s\n\n", filename.c_str() );

    // Get the list of methods to try
    auto methods = options.methods;
    if ( methods.empty() ) {
        methods.push_back( "cpu" );
        methods.push_back( "threads" );
#ifdef USE_OPENMP
        methods.push_back( "OpenMP" );
#endif
#ifdef USE_CUDA
        methods.push_back( "Cuda" );
        methods.push_back( "Cuda-MultiGPU" );
#endif
#ifdef USE_OPENACC
        methods.push_back( "OpenAcc" );
#endif
#ifdef USE_KOKKOS
        methods.push_back( "Kokkos-Serial" );
#ifdef KOKKOS_HAVE_PTHREAD
//methods.push_back("Kokkos-Thread");
#endif
#ifdef KOKKOS_HAVE_OPENMP
        methods.push_back( "Kokkos-OpenMP" );
#endif
#ifdef KOKKOS_HAVE_CUDA
        methods.push_back( "Kokkos-Cuda" );
#endif
#endif
    }

    // Call a dummy CUDA/OpenAcc method to initialize the GPU (for more accurate times)
    static bool cudaInitialized = false;
    if ( !cudaInitialized ) {
        auto index = std::find(methods.begin(),methods.end(),"Cuda-MultiGPU");
        if ( index == methods.end() )
            index = std::find(methods.begin(),methods.end(),"Cuda");
        if ( index == methods.end() )
            index = find(methods.begin(),methods.end(),"OpenAcc");
        if ( index != methods.end() ) {
            auto info = loadInput( filename, 0.1 );
            RayTrace::create_image( info, *index );
            free2( info );
        }
        cudaInitialized = true;
    }

    // Load the image structure
    double *image0=NULL, *I_ang0=NULL;
    auto info = loadInput( filename, options.scale, &image0, &I_ang0 );
    if ( info == NULL )
        return -2;

    // Call create_image for each method
    int N_errors = 0;
    sleep_ms( 50 );
    std::vector<std::vector<double>> time( methods.size() );
    for ( size_t i = 0; i < methods.size(); i++ ) {
        if ( rank() == 0 )
            printf( "Running %s\n", methods[i].c_str() );
        double start = getTime();
        for ( int it = 0; it < options.iterations; it++ ) {
            RayTrace::create_image( info, methods[i] );
            double stop = getTime();
            time[i].push_back( stop - start );
            start = stop;
        }
        time[i] = gatherAll( time[i] );
        // Check the results
        if ( options.scale == 1.0 ) {
            bool pass = check_ans( image0, I_ang0, *info );
            if ( !pass )
                N_errors++;
        }
        free( (void *) info->image );
        free( (void *) info->I_ang );
        info->image = NULL;
        info->I_ang = NULL;
    }
    if ( rank() == 0 ) {
        printf( "\n        METHOD    Avg     Min     Max   Std Dev\n" );
        for ( size_t i = 0; i < methods.size(); i++ ) {
            double min = getMin( time[i] );
            double max = getMax( time[i] );
            double avg = getAvg( time[i] );
            double dev = getDev( time[i] );
            printf( "%14s %7.3f %7.3f %7.3f %7.3f\n", methods[i].c_str(), avg, min, max, dev );
            if ( dev/avg > 0.10 ) {
                printf( "   Standard deviation exceeded tolerance (10%%)\n");
                N_errors++;
            }
            if ( (max-avg)/avg > 0.15 ) {
                printf( "   Maximum runtime exceeded average by more than 15%%\n");
                N_errors++;
            }
        }
    }

    // Free memory and return
    free( (void *) image0 );
    free( (void *) I_ang0 );
    free2( info );
    return sumReduce(N_errors);
}


/******************************************************************
* Initialize/finalize kokkos                                      *
******************************************************************/
void KokkosInitialize( int argc, char *argv[] )
{
    NULL_USE(argc);
    NULL_USE(argv);
#ifdef USE_KOKKOS
#ifdef KOKKOS_HAVE_PTHREAD
    int argc2             = 1;
    const char *argv2[10] = { NULL };
    argv2[0]              = argv[0];
#ifdef KOKKOS_HAVE_PTHREAD
    argv2[argc2] = "--kokkos-threads=16";
    argc2++;
#endif
    Kokkos::initialize( argc2, (char **) argv2 );
#else
    Kokkos::initialize( argc, argv );
#endif
#endif
}
void KokkosFinalize()
{
#ifdef USE_KOKKOS
    Kokkos::finalize();
#endif
}


/******************************************************************
* The main program                                                *
******************************************************************/
int main( int argc, char *argv[] )
{
    // Start MPI (if used)
    startup( argc, argv );

    // Check the input arguments
    Options options;
    std::vector<std::string> filenames = options.read_cmd( argc, argv );
    if ( filenames.empty() )
        return -2;

    // Initialize kokkos
    KokkosInitialize( argc, argv );

    // Run the tests for all files
    int N_errors = 0;
    for (size_t i=0; i<filenames.size(); i++)
        N_errors += run_tests( filenames[i], options );

    if ( N_errors == 0 )
        std::cout << "\nAll tests passed\n";
    else
        std::cout << "\nSome tests failed\n";
    KokkosFinalize();
    shutdown();
    return N_errors;
}

