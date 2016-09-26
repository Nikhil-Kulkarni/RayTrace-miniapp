// This is a mini-application to mimics part of the full Ray-Trace code
// This miniapp mimics the behavior of create_image

#include "RayTrace.h"
#include <cstring>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdint.h>
#include <string>


#if defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 )
#include <windows.h>
#define get_time( x ) QueryPerformanceCounter( x )
#define get_frequency( f ) QueryPerformanceFrequency( f )
#define TIME_TYPE LARGE_INTEGER
#define sleep_ms Sleep
inline double get_diff( TIME_TYPE start, TIME_TYPE end, TIME_TYPE f )
{
    return ( ( (double) ( end.QuadPart - start.QuadPart ) ) / ( (double) f.QuadPart ) );
}
#else
#include <sys/time.h>
#define get_time( x ) gettimeofday( x, NULL );
#define get_frequency( f ) ( *f = timeval() )
#define TIME_TYPE timeval
#define sleep_ms( X )                        \
    do {                                     \
        struct timespec ts;                  \
        ts.tv_sec  = X / 1000;               \
        ts.tv_nsec = ( X % 1000 ) * 1000000; \
        nanosleep( &ts, NULL );              \
    } while ( 0 )
inline double get_diff( TIME_TYPE start, TIME_TYPE end, TIME_TYPE )
{
    return ( (double) end.tv_sec - start.tv_sec ) + 1e-6 * ( (double) end.tv_usec - start.tv_usec );
}
#endif

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#endif


inline void fread2( void *ptr, size_t size, size_t count, FILE *fid )
{
    size_t N = fread( ptr, size, count, fid );
    if ( N != count ) {
        std::cerr << "Failed to read desired count\n";
        exit( -1 );
    }
}


// Check the answer
inline bool check_ans(
    const double *image0, const double *I_ang0, const RayTrace::create_image_struct &data )
{
    size_t N_image  = data.euv_beam->nx * data.euv_beam->ny * data.euv_beam->nv;
    size_t N_ang    = data.euv_beam->na * data.euv_beam->nb;
    double error[2] = { 0, 0 };
    double norm0[2] = { 0, 0 };
    double norm1[2] = { 0, 0 };
    for ( size_t i = 0; i < N_image; i++ ) {
        error[0] += ( image0[i] - data.image[i] ) * ( image0[i] - data.image[i] );
        norm0[0] += image0[i] * image0[i];
        norm1[0] += data.image[i] * data.image[i];
    }
    for ( size_t i = 0; i < N_ang; i++ ) {
        error[1] += ( I_ang0[i] - data.I_ang[i] ) * ( I_ang0[i] - data.I_ang[i] );
        norm0[1] += I_ang0[i] * I_ang0[i];
        norm1[1] += data.I_ang[i] * data.I_ang[i];
    }
    norm0[0] = sqrt( norm0[0] );
    norm0[1] = sqrt( norm0[1] );
    norm1[0] = sqrt( norm1[0] );
    norm1[1] = sqrt( norm1[1] );
    error[0] = sqrt( error[0] ) / norm0[0];
    error[1] = sqrt( error[1] ) / norm0[1];
    const double tol =
        5e-6; // RayTrace uses single precision for some calculations (may need to adjust to 1e-5)
    // bool pass = error[0]<=tol && error[1]<=tol;
    bool pass =
        ( norm0[0] - norm1[0] ) / norm0[0] <= tol && ( norm0[1] - norm1[1] ) / norm0[1] <= tol;
    if ( !pass ) {
        std::cerr << "  Answers do not match:" << std::endl;
        std::cerr << "    image: " << error[0] << " " << norm0[0] << " " << norm1[0] << std::endl;
        std::cerr << "    I_ang: " << error[1] << " " << norm0[1] << " " << norm1[1] << std::endl;
    }
    return pass;
}


// Scale the input problem
template <class TYPE>
void scale_beam( TYPE &beam, double scale )
{
    const double x[2] = { beam.x[0] - 0.5 * beam.dx, beam.x[beam.nx - 1] + 0.5 * beam.dx };
    const double y[2] = { beam.y[0] - 0.5 * beam.dy, beam.y[beam.ny - 1] + 0.5 * beam.dy };
    const double a[2] = { beam.a[0] - 0.5 * beam.da, beam.a[beam.na - 1] + 0.5 * beam.da };
    const double b[2] = { beam.b[0] - 0.5 * beam.db, beam.b[beam.nb - 1] + 0.5 * beam.db };
    int nx            = static_cast<int>( beam.nx * scale );
    int ny            = static_cast<int>( beam.ny * scale );
    int na            = static_cast<int>( beam.na * scale );
    int nb            = static_cast<int>( beam.nb * scale );
    delete[] beam.x;
    beam.x = new double[nx];
    delete[] beam.y;
    beam.y = new double[ny];
    delete[] beam.a;
    beam.a = new double[na];
    delete[] beam.b;
    beam.b  = new double[nb];
    beam.nx = nx;
    beam.dx = ( x[1] - x[0] ) / nx;
    beam.ny = ny;
    beam.dy = ( y[1] - y[0] ) / ny;
    beam.na = na;
    beam.da = ( a[1] - a[0] ) / na;
    beam.nb = nb;
    beam.db = ( b[1] - b[0] ) / nb;
    for ( int i = 0; i < nx; i++ ) {
        beam.x[i] = x[0] + ( 0.5 + i ) * beam.dx;
    }
    for ( int i = 0; i < ny; i++ ) {
        beam.y[i] = y[0] + ( 0.5 + i ) * beam.dy;
    }
    for ( int i = 0; i < na; i++ ) {
        beam.a[i] = a[0] + ( 0.5 + i ) * beam.da;
    }
    for ( int i = 0; i < nb; i++ ) {
        beam.b[i] = b[0] + ( 0.5 + i ) * beam.db;
    }
}
void scale_problem( RayTrace::create_image_struct &info, double scale )
{
    scale_beam( *const_cast<RayTrace::EUV_beam_struct *>( info.euv_beam ), pow( scale, 0.25 ) );
    if ( info.seed_beam != NULL )
        scale_beam(
            *const_cast<RayTrace::seed_beam_struct *>( info.seed_beam ), pow( scale, 0.25 ) );
}


// Clas to hold options
class Options {
public:
    Options() {};
    int iterations = 1;
    double scale   = 1.0;
    std::vector<std::string> methods;
    std::vector<std::string> read_cmd( int argc, char *argv[] )
    {
        const char *err_msg = "CreateImage called with the wrong number of arguments:\n"
            "  CreateImage <args> file.dat\n"
            "Optional arguments:\n"
            "  -methods=METHODS  Comma seperated list of methods to test.  Default is all availible methods\n"
            "                    cpu, threads, OpenMP, Cuda, OpenAcc, Kokkos-Serial, "
            "Kokkos-Thread, Kokkos-OpenMP, Kokkos-Cuda\n"
            "  -iterations=N     Number of iterations to run.  Time returned will be "
            "the average time/iteration.\n"
            "  -scale=factor     Increate the size of the problem by ~ this factor. "
            "(2.0 - twice as expensive)\n"
            "                    Note: this will disable checking the answer.\n"
            "                    Note: the scale factor is only approximate.\n";
        std::vector<std::string> filenames;
        for ( int i = 1; i < argc; i++ ) {
            if ( argv[i][0] == '-' ) {
                // Processing an argument
                if ( strncmp( argv[i], "-methods=", 9 ) == 0 ) {
                    std::stringstream ss( &argv[i][9] );
                    std::string token;
                    while ( std::getline( ss, token, ',' ) )
                        methods.push_back( token );
                } else if ( strncmp( argv[i], "-iterations=", 12 ) == 0 ) {
                    iterations = atoi( &argv[i][12] );
                } else if ( strncmp( argv[i], "-scale=", 7 ) == 0 ) {
                    scale = atof( &argv[i][7] );
                } else {
                    std::cerr << "Unknown option: " << argv[i] << std::endl;
                    return std::vector<std::string>();
                }
            } else {
                // Processing a filename
                filenames.push_back( argv[i] );
            }
        }
        if ( filenames.empty() )
            std::cerr << err_msg;
        return filenames;
    }
};


// Run the tests for a single file
int run_tests( const std::string& filename, const Options& options )
{
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
    std::vector<double> time( methods.size() );
    for ( size_t i = 0; i < methods.size(); i++ ) {
        printf( "Running %s\n", methods[i].c_str() );
        TIME_TYPE start, stop, f;
        get_frequency( &f );
        get_time( &start );
        for ( int it = 0; it < options.iterations; it++ )
            RayTrace::create_image( &info, methods[i] );
        get_time( &stop );
        time[i] = get_diff( start, stop, f );
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
    printf( "\n      METHOD     TIME\n" );
    for ( size_t i = 0; i < methods.size(); i++ )
        printf( "%14s   %0.3f\n", methods[i].c_str(), time[i] );

    // Free memory and return
    free( (void *) image0 );
    free( (void *) I_ang0 );
    delete info.euv_beam;
    delete info.seed_beam;
    delete[] info.gain;
    delete info.seed;
    return N_errors;
}


/******************************************************************
* The main program                                                *
******************************************************************/
int main( int argc, char *argv[] )
{

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
        std::cout << "All tests passed\n";
    else
        std::cout << "Some tests failed\n";
#ifdef USE_KOKKOS
    Kokkos::finalize();
#endif
    return N_errors;
}
