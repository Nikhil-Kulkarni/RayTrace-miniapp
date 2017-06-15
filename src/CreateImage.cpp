// This is a mini-application to mimics part of the full Ray-Trace code
// This miniapp mimics the behavior of create_image

#include "RayTrace.h"
#include "MPI_helpers.h"
#include "CreateImageHelpers.h"

#include <cstring>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdint.h>
#include <string>



#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#endif




// Run the tests for a single file
int run_tests( const std::string& filename, const Options& options )
{
    if ( rank() == 0 )
        printf( "\nRunning tests for %s\n\n", filename.c_str() );
    // load the input file
    FILE *fid = fopen( filename.c_str(), "rb" );
    if ( fid == NULL ) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return -2;
    }
    uint64_t N_bytes = 0;
    fread2( &N_bytes, sizeof( uint64_t ), 1, fid );
    char *data = new char[N_bytes];
    fread2( data, sizeof( char ), N_bytes, fid );
    fclose( fid );

    // Create the image structure
    double scale = options.scale;
    RayTrace::create_image_struct info;
    info.unpack( std::pair<char *, size_t>( data, N_bytes ) );
    delete[] data;
    data                 = NULL;
    const double *image0 = info.image;
    const double *I_ang0 = info.I_ang;
    info.image           = NULL;
    info.I_ang           = NULL;
    if ( scale != 1.0 ) {
        delete[] image0;
        image0 = NULL;
        delete[] I_ang0;
        I_ang0 = NULL;
        scale_problem( info, scale );
    }

    // Get the list of methods to try
    std::vector<std::string> methods = options.methods;
    if ( methods.empty() ) {
        methods.push_back( "cpu" );
#if CXX_STD == 11 || CXX_STD == 14
        methods.push_back( "threads" );
#endif
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
// methods.push_back("Kokkos-Thread");
#endif
#ifdef KOKKOS_HAVE_OPENMP
        methods.push_back( "Kokkos-OpenMP" );
#endif
#ifdef KOKKOS_HAVE_CUDA
        methods.push_back( "Kokkos-Cuda" );
#endif
#endif
    }

    // Call create_image for each method
    int N_errors = 0;
    sleep_ms( 50 );
    std::vector<std::vector<double>> time( methods.size() );
    for ( size_t i = 0; i < methods.size(); i++ ) {
        if ( rank() == 0 )
            printf( "Running %s\n", methods[i].c_str() );
        double start = getTime();
        for ( int it = 0; it < options.iterations; it++ ) {
            RayTrace::create_image( &info, methods[i] );
            double stop = getTime();
            time[i].push_back( stop - start );
            start = stop;
        }
        time[i] = gatherAll( time[i] );
        // Check the results
        if ( scale == 1.0 ) {
            bool pass = check_ans( image0, I_ang0, info );
            if ( !pass )
                N_errors++;
        }
        free( (void *) info.image );
        free( (void *) info.I_ang );
        info.image = NULL;
        info.I_ang = NULL;
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
    delete info.euv_beam;
    delete info.seed_beam;
    delete[] info.gain;
    delete info.seed;
    return sumReduce(N_errors);
}


/******************************************************************
* The main program                                                *
******************************************************************/
int main( int argc, char *argv[] )
{
    // Start MPI (if used)
    startup( argc, argv );

// Initialize kokkos
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

    // Check the input arguments
    Options options;
    std::vector<std::string> filenames = options.read_cmd( argc, argv );
    if ( filenames.empty() )
        return -2;

    // Run the tests for all files
    int N_errors = 0;
    for (size_t i=0; i<filenames.size(); i++)
        N_errors += run_tests( filenames[i], options );

    if ( N_errors == 0 )
        std::cout << "\nAll tests passed\n";
    else
        std::cout << "\nSome tests failed\n";
#ifdef USE_KOKKOS
    Kokkos::finalize();
#endif
    shutdown();
    return N_errors;
}
