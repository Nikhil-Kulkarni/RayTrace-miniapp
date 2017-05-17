#include "CreateImageHelpers.h"

#include <math.h>


#if defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 )
#include <windows.h>
#define get_time( x ) QueryPerformanceCounter( x )
#define get_frequency( f ) QueryPerformanceFrequency( f )
#define TIME_TYPE LARGE_INTEGER
void sleep_ms( int N ) { Sleep(N) };
inline double get_diff( TIME_TYPE start, TIME_TYPE end, TIME_TYPE f )
{
    return ( ( (double) ( end.QuadPart - start.QuadPart ) ) / ( (double) f.QuadPart ) );
}
#else
#include <sys/time.h>
#define get_time( x ) gettimeofday( x, NULL );
#define get_frequency( f ) ( *f = timeval() )
#define TIME_TYPE timeval
void sleep_ms( int N ) {
    struct timespec ts;
    ts.tv_sec  = N / 1000;
    ts.tv_nsec = ( N % 1000 ) * 1000000;
    nanosleep( &ts, NULL );
}
inline double get_diff( TIME_TYPE start, TIME_TYPE end, TIME_TYPE )
{
    return ( (double) end.tv_sec - start.tv_sec ) + 1e-6 * ( (double) end.tv_usec - start.tv_usec );
}
#endif


// Read data
void fread2( void *ptr, size_t size, size_t count, FILE *fid )
{
    size_t N = fread( ptr, size, count, fid );
    if ( N != count ) {
        std::cerr << "Failed to read desired count\n";
        exit( -1 );
    }
}


// Get the time since startup
static inline TIME_TYPE getTime2( ) {
    TIME_TYPE t;
    get_time( &t );
    return t;
}
static inline TIME_TYPE getFrequency( ) {
    TIME_TYPE f;
    get_frequency( &f );
    return f;
}
static TIME_TYPE global_start = getTime2( );
static TIME_TYPE global_frequency = getFrequency( );
double getTime()
{
    TIME_TYPE stop = getTime2( );
    return get_diff( global_start, stop, global_frequency );
}


// Check the answer
bool check_ans( const double *image0, const double *I_ang0, const RayTrace::create_image_struct &data )
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


// Get the minimum value
double getMin( const std::vector<double>& x )
{
    double y = x[0];
    for (size_t i=1; i<x.size(); i++)
        y = std::min(y,x[i]);
    return y;
}


// Get the maximum value
double getMax( const std::vector<double>& x )
{
    double y = x[0];
    for (size_t i=1; i<x.size(); i++)
        y = std::max(y,x[i]);
    return y;
}


// Get the average value
double getAvg( const std::vector<double>& x )
{
    double y = 0;
    for (size_t i=0; i<x.size(); i++)
        y += x[i];
    return y/x.size();
}


// Get the standard deviation
double getDev( const std::vector<double>& x )
{
    double avg = getAvg( x );
    double y = 0;
    for (size_t i=0; i<x.size(); i++)
        y += (x[i]-avg)*(x[i]-avg);
    y = sqrt(y/x.size());
    return y;
}

