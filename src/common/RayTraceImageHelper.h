#ifndef RayTraceImageHelper
#define RayTraceImageHelper
// This file contains the source code for propagating the rays for a single thread
#define NOMINMAX
#include <cstring>
#include <math.h>
#include <stdint.h>

// RayTrace includes
#include "RayTrace.h"

#ifdef USE_OPENACC
// Include this file for OpenACC API
#ifdef _OPENACC
#include <openacc.h>
#endif
#endif
#if defined( __CUDA_ARCH__ )
#include <cuda.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

#if !defined( USE_OPENACC ) && !defined( USE_KOKKOS ) && !defined( USE_CUDA )
#define RAY_DEBUG
#endif

#define N_MAX 20        // Maximum number of length segments
#define K_MAX 100       // Maximum number of frequencies
#define N_SUB 3         // Number of sub lengths to use
#define N_FAILED_MAX 32 // Maximum number of failed rays to store/save


// Structure to hold a ray
struct ray_struct {
    float x;
    float y;
    float a;
    float b;
};


/**********************************************************************
* Some simple functions to set and check the Nth bit in an array      *
**********************************************************************/
inline void set_bit( size_t N, unsigned int &data )
{
    unsigned int mask = ( (unsigned int) 0x1 ) << N;
    data |= mask;
}
inline bool check_bit( size_t N, unsigned int data )
{
    unsigned int mask = ( (unsigned int) 0x1 ) << N;
    return ( data & mask ) != 0;
}


/**********************************************************************
* Helper function to normalize a vector (assuming the vector is ~1)   *
* Note: We cannot safely use the approximate version as s may drift   *
* from |s|=1                                                          *
**********************************************************************/
struct vec_struct {
    float x;
    float y;
    float z;
};
#ifdef USE_OPENACC
#pragma acc routine seq
#endif
HOST_DEVICE
inline void normalize_s( vec_struct &s )
{
#if 0
        // Approximate 1/|s| as 1-x*(0.5-0.375*x) where x = |s|^2-1
        float tmp = s.x*s.x+s.y*s.y+s.z*s.z-1;
        tmp = 1.0f - tmp*(0.5f-0.375f*tmp);     // 1/|s|
        s.x *= tmp;
        s.y *= tmp;
        s.z *= tmp;
#else
    float tmp = s.x * s.x + s.y * s.y + s.z * s.z;
    tmp       = 1.0 / sqrt( tmp );
    s.x *= tmp;
    s.y *= tmp;
    s.z *= tmp;
#endif
}


/**********************************************************************
* Find the first element in X which is greater than y using a simple  *
*   hashing technique.  If no values are greater it will return N.    *
* Note that the index returned will be 0<=i<=N.                       *
**********************************************************************/
#ifdef USE_OPENACC
#pragma acc routine seq
#endif
HOST_DEVICE
inline size_t findfirstsingle( const double *X, size_t size_X, double Y )
{
    if ( Y < X[0] )
        return 0;
    if ( Y > X[size_X - 1] )
        return size_X;
    size_t lower = 0;
    size_t upper = size_X - 1;
    while ( ( upper - lower ) != 1 ) {
        size_t value = ( upper + lower ) / 2;
        if ( X[value] >= Y )
            upper = value;
        else
            lower = value;
    }
    return upper;
}


/**********************************************************************
* Find the index for interpolation:                                   *
* Find the first element in X which is greater than y using a simple  *
*   hashing technique.  Note that the index returned will be 0<i<N.   *
* This function is similar to findfirstsingle except that it          *
*   assumes size_X >= 2 and will not return 0 or size_X.              *
**********************************************************************/
#ifdef USE_OPENACC
#pragma acc routine seq
#endif
HOST_DEVICE
inline uint32_t findindex( const double *X, uint32_t size_X, double Y )
{
    uint32_t lower = 0;
    uint32_t upper = size_X - 1;
    while ( ( upper - lower ) != 1 ) {
        uint32_t value = ( upper + lower ) / 2;
        if ( X[value] >= Y )
            upper = value;
        else
            lower = value;
    }
    return upper;
}


/******************************************************************
* Subroutine to perform bi-linear interpolation                   *
******************************************************************/
#ifdef USE_OPENACC
#pragma acc routine seq
#endif
HOST_DEVICE
inline float bilinear( float dx, float dy, float f1, float f2, float f3, float f4 )
{
    float dx2 = 1.0f - dx;
    float dy2 = 1.0f - dy;
    return ( dx * f2 + dx2 * f1 ) * dy2 + ( dx * f4 + dx2 * f3 ) * dy;
}


/******************************************************************
* Subroutine to perform cubic hermite interpolation               *
******************************************************************/
#ifdef USE_OPENACC
#pragma acc routine seq
#endif
HOST_DEVICE
inline double interp_pchip( size_t N, const double *xi, const double *yi, double x )
{
    if ( x <= xi[0] || N <= 2 ) {
        double dx = ( x - xi[0] ) / ( xi[1] - xi[0] );
        return ( 1.0 - dx ) * yi[0] + dx * yi[1];
    } else if ( x >= xi[N - 1] ) {
        double dx = ( x - xi[N - 2] ) / ( xi[N - 1] - xi[N - 2] );
        return ( 1.0 - dx ) * yi[N - 2] + dx * yi[N - 1];
    }
    size_t i  = findfirstsingle( xi, N, x );
    double f1 = yi[i - 1];
    double f2 = yi[i];
    double dx = ( x - xi[i - 1] ) / ( xi[i] - xi[i - 1] );
    // Compute the gradient in normalized coordinates [0,1]
    double g1 = 0, g2 = 0;
    if ( i <= 1 ) {
        g1 = f2 - f1;
    } else if ( ( f1 < f2 && f1 > yi[i - 2] ) || ( f1 > f2 && f1 < yi[i - 2] ) ) {
        // Compute the gradient by using a 3-point finite difference to f'(x)
        // Note: the real gradient is g1/(xi[i]-xi[i-1])
        double f0    = yi[i - 2];
        double dx1   = xi[i - 1] - xi[i - 2];
        double dx2   = xi[i] - xi[i - 1];
        double a1    = ( dx2 - dx1 ) / dx1;
        double a2    = dx1 / ( dx1 + dx2 );
        g1           = a1 * ( f1 - f0 ) + a2 * ( f2 - f0 );
        double fx1   = fabs( f1 - f0 ) / dx1;
        double fx2   = fabs( f2 - f1 ) / dx2;
        double g_max = 2 * dx2 * ( fx1 < fx2 ? fx1 : fx2 );
        g1           = ( ( g1 >= 0 ) ? 1 : -1 ) * ( fabs( g1 ) < g_max ? fabs( g1 ) : g_max );
    }
    if ( i >= N - 1 ) {
        g2 = f2 - f1;
    } else if ( ( f2 < f1 && f2 > yi[i + 1] ) || ( f2 > f1 && f2 < yi[i + 1] ) ) {
        // Compute the gradient by using a 3-point finite difference to f'(x)
        // Note: the real gradient is g2/(xi[i]-xi[i-1])
        double f0    = yi[i + 1];
        double dx1   = xi[i] - xi[i - 1];
        double dx2   = xi[i + 1] - xi[i];
        double a1    = -dx2 / ( dx1 + dx2 );
        double a2    = ( dx2 - dx1 ) / dx2;
        g2           = a1 * ( f1 - f0 ) + a2 * ( f2 - f0 );
        double fx1   = fabs( f2 - f1 ) / dx1;
        double fx2   = fabs( f0 - f2 ) / dx2;
        double g_max = 2 * dx1 * ( fx1 < fx2 ? fx1 : fx2 );
        g2           = ( ( g2 >= 0 ) ? 1 : -1 ) * ( fabs( g2 ) < g_max ? fabs( g2 ) : g_max );
    }
    // Perform the interpolation
    double dx2 = dx * dx;
    double f =
        f1 + dx2 * ( 2 * dx - 3 ) * ( f1 - f2 ) + dx * g1 - dx2 * ( g1 + ( 1 - dx ) * ( g1 + g2 ) );
    return f;
}


/******************************************************************
*  Function to calculate the seed intensity at a point            *
******************************************************************/
#ifdef USE_OPENACC
#pragma acc routine seq
#endif
HOST_DEVICE
inline void calc_seed_inline(
    const RayTrace::ray_seed_struct &seed, double x, double y, double a, double b, double *Iv )
{
    double f = 0.0;
    // Check to see if the point is within the grid given
    if ( x >= seed.x[0][0] && x <= seed.x[0][seed.dim[0] - 1] && y >= seed.x[1][0] &&
         y <= seed.x[1][seed.dim[1] - 1] && a >= seed.x[2][0] && a <= seed.x[2][seed.dim[2] - 1] &&
         b >= seed.x[3][0] && b <= seed.x[3][seed.dim[3] - 1] ) {
        double fx = interp_pchip( seed.dim[0], seed.x[0], seed.f[0], x );
        double fy = interp_pchip( seed.dim[1], seed.x[1], seed.f[1], y );
        double fa = interp_pchip( seed.dim[2], seed.x[2], seed.f[2], a );
        double fb = interp_pchip( seed.dim[3], seed.x[3], seed.f[3], b );
        f         = seed.f0 * fx * fy * fa * fb;
        f         = f < 0.0 ? 0.0 : f; // max(f,0);
    }
    for ( int i = 0; i < seed.dim[4]; i++ )
        Iv[i]   = f * seed.f[4][i];
}


/**********************************************************************
* Inline function to propagate a ray over a small step over which we  *
*    can assume n(x,y) = n0 + a*x + b*y                               *
* Inputs/Outputs:                                                     *
*    r      - Position vector                                         *
*    s      - Ray vector                                              *
*    dx - Maximum distance to propagate                               *
*    dx_sum - Distance moved                                          *
*    dn_dx  - Gradient of n in the x-direction                        *
*    dn_dy  - Gradient of n in the y-direction                        *
*    c - Safety factor (must be < 1)                                  *
* Return value - Path length traversed                                *
**********************************************************************/
// Check if all(x<y)
#define CHECK_DIST( x, y ) x[0] < y[0] && x[1] < y[1] && x[2] < y[2]
// Propagate
#ifdef USE_OPENACC
#pragma acc routine seq
#endif
HOST_DEVICE
inline float propagate(
    vec_struct &r, vec_struct &s, float n0, float dn_dx, float dn_dy, float dx[3], float c )
{
    float sum    = 0.0f;
    float dz_max = c * 1.00001f * dx[2];
    r.x          = 0;
    r.y          = 0;
    r.z          = 0;
    float n      = n0;
    while ( fabs( r.x ) < dx[0] && fabs( r.y ) < dx[1] && fabs( r.z ) < dx[2] &&
            fabs( n - n0 ) < 0.05 ) {
        // Compute some basic variables
        n          = n0 + r.x * dn_dx + r.y * dn_dy;             // n(r)
        float t    = ( s.x * dn_dx + s.y * dn_dy + 1e-12f ) / n; // 1/n*dot(s,grad(n))
        float f[3] = { dn_dx / n - s.x * t, dn_dy / n - s.y * t,
            -s.z * t }; // 1/n*(grad(n)-dot(s,grad(n)))
        // Compute the step length
        // Note: we are assuming n, grad(n) and dot(s,grad(n)) are const
        float step  = c * 0.1f / fabs( t );          // step length based on step*t
        step        = step < dz_max ? step : dz_max; // step length based on dz_max
        float step2 = 1.0001f * ( dx[2] - fabs( r.z ) ) / fabs( s.z ); // step length based on dz-z
        float step3 = c * 0.05f * ( fabs( s.x ) + 5e-4f ) /
                      ( fabs( f[0] ) + 1e-8f ); // step length based on change of sx
        float step4 = c * 0.05f * ( fabs( s.y ) + 5e-4f ) /
                      ( fabs( f[1] ) + 1e-8f ); // step length based on change of sy
        step     = step < step2 ? step : step2;
        step     = step < step3 ? step : step3;
        step     = step < step4 ? step : step4;
        float st = step * t;
        // Update r
        float c1 = 0.5f * step * step * ( 1.0f - st / 3.0f + st * st / 12.0f );
        r.x += s.x * step + c1 * f[0];
        r.y += s.y * step + c1 * f[1];
        r.z += s.z * step + c1 * f[2];
        // Update s
        float c2 = step * ( 1.0f - 0.5f * st + st * st / 6.0f );
        s.x += c2 * f[0];
        s.y += c2 * f[1];
        s.z += c2 * f[2];
        normalize_s( s );
        sum += step;
    }
    return sum;
}
#ifdef USE_OPENACC
#pragma acc routine seq
#endif
HOST_DEVICE
inline float propagate2( vec_struct &pos, vec_struct &s, float dz, const double x[2],
    const double y[2], const float range[4], const double n[4], bool abs_y, float c )
{
    float z        = 0.0f;
    float ds_sum   = 0.0f;
    const float dx = (float) ( x[1] - x[0] );
    const float dy = (float) ( y[1] - y[0] );
    float y2       = abs_y ? fabs( pos.y ) : pos.y;
    while (
        pos.x > range[0] && pos.x < range[1] && y2 > range[2] && y2 < range[3] && z < 0.999 * dz ) {
        // Interpolate the necessary quantities
        y2          = abs_y ? fabs( pos.y ) : pos.y;
        float dxi   = (float) ( ( pos.x - x[0] ) / dx );
        float dyi   = (float) ( ( y2 - y[0] ) / dy );
        float n0    = bilinear( dxi, dyi, (float) n[0], (float) n[1], (float) n[2], (float) n[3] );
        float dn_dx = (float) ( ( 1.0 - dyi ) * ( n[1] - n[0] ) / dx + dyi * ( n[3] - n[2] ) / dx );
        float dn_dy = (float) ( ( 1.0 - dxi ) * ( n[2] - n[0] ) / dy + dxi * ( n[3] - n[1] ) / dy );
        if ( abs_y && pos.y < 0 )
            dn_dy = -dn_dy;
        // Propagate
        vec_struct r;
        r.x             = 0;
        r.y             = 0;
        r.z             = 0;
        float dx_max[3] = { 0.1f * dx, 0.1f * dy, dz - z };
        ds_sum += propagate( r, s, n0, dn_dx, dn_dy, dx_max, c );
        pos.x += r.x;
        pos.y += r.y;
        pos.z += r.z;
        z += fabs( r.z );
        y2 = abs_y ? fabs( pos.y ) : pos.y;
    }
    return ds_sum;
}


/**********************************************************************
* This is a subfunction to follow the path of an individual ray, and  *
*    its resulting gain and intensity.                                *
* Input variables:                                                    *
*    ray    - The ray information                                     *
*    N      - The number of lengths                                   *
*    dz     - The length grid spacing                                 *
*    gain   - The gain structure                                      *
*    seed   - The seed properties                                     *
*    K      - The number of frequencies                               *
*    method - The propagation method                                  *
*    c      - Optional safety factor for calculating the step length  *
*    dv     - Optional frequency grid spacing (used if debug!=NULL)   *
*    debug  - Optional variable to store debug info                   *
* Output variables:                                                   *
*    Iv     - The output intensity (1xK)                              *
*    ray2   - The output ray properties                               *
* Notes:                                                              *
*    This routine uses mixed precision to minimize memory movement    *
*    while maintaining necessary accuracy.                            *
**********************************************************************/
#ifdef USE_OPENACC
#pragma acc routine seq
#endif
HOST_DEVICE
inline int RayTrace_calc_ray( const ray_struct &ray, const int N, const float dz0,
    const RayTrace::ray_gain_struct *gain, const RayTrace::ray_seed_struct *seed, int K, int method,
    double *Iv, ray_struct &ray2, const float c = 0.5, const double *dv = NULL,
    float *debug = NULL )
{

    // Initialize gvl, evl, ikl, Iv
    float gvl[N_MAX][N_SUB];
    float evl[N_MAX][N_SUB];
    int ivl[N_MAX][N_SUB];
    // For both CCE and PGI compilers (with OpenACC), initialization lists
    // result in internal compiler error, so the below loops implement instead.
    for ( int i = 0; i < N_MAX; ++i ) {
        for ( int is = 0; is < N_SUB; is++ ) {
            gvl[i][is] = 0.0f;
            evl[i][is] = 0.0f;
            ivl[i][is] = 0;
        }
    }
    for ( int k = 0; k < K; ++k )
        Iv[k]   = 0.0;

    // Check if we are using the emissivity
    bool use_emis = gain->E0 != NULL && seed == NULL;

    // Calculate starting point and the direction vector s
    vec_struct s, pos;
    pos.x = ray.x;
    pos.y = ray.y;
    pos.z = 0.0f;
    s.x   = tan( 1e-3f * ray.a );
    s.y   = tan( 1e-3f * ray.b );
    s.z   = 1.0f;
    if ( method == 1 ) {
        // Propagate backward
        s.x = -s.x;
        s.y = -s.y;
        s.z = -s.z;
    }
    normalize_s( s );
#ifdef RAY_DEBUG
    if ( dv != NULL && debug != NULL ) {
        short int ii = method == 1 ? ( N - 1 ) * N_SUB : 0;
        memset( debug, 0, 3 * ( N_SUB * ( N - 1 ) + 1 ) * sizeof( float ) );
        debug[3 * ii + 0] = pos.x;
        debug[3 * ii + 1] = pos.y;
    }
#endif

    // Propagate though the plasma, saving g(z)
    bool escaped = false;
    for ( short int i = 0; i < N - 1 && !escaped; i++ ) {
        // Get the index to the current length that is used for the gain
        // Note: we always use the length segment on the high energy side to
        //     limit the error from saturation at high intensities
        short int ii = 0;
        if ( method == 1 ) {
            // Propagate backward
            ii = N - i - 1;
        } else {
            // Propagate forward
            ii = i + 1;
        }
        uint32_t Nx = gain[ii].Nx;
        uint32_t Ny = gain[ii].Ny;
        float range[4];
        range[0]   = static_cast<float>( gain[ii].x[0] );
        range[1]   = static_cast<float>( gain[ii].x[Nx - 1] );
        range[2]   = static_cast<float>( gain[ii].y[0] );
        range[3]   = static_cast<float>( gain[ii].y[Ny - 1] );
        bool abs_y = false;
        if ( range[2] >= 0 ) {
            range[2] = -range[3];
            abs_y    = true;
        }
        const double *ptr_x = gain[ii].x;
        const double *ptr_y = gain[ii].y;
        const double *ptr_n = gain[ii].n;
        const float *ptr_g0 = gain[ii].g0;
        const float *ptr_E0 = gain[ii].E0;
        float z             = 0.0f;
        for ( short int iz = 0; iz < N_SUB; iz++ ) {
            short int is = method == 1 ? N_SUB - iz - 1 : iz;
            float z_stop = ( dz0 * ( iz + 1.0f ) / N_SUB );
            while ( z < 0.995f * z_stop ) {
                // Check if the ray escaped the plasma
                if ( pos.x < range[0] || pos.x > range[1] || pos.y < range[2] || pos.y > range[3] ||
                     s.z * s.z < 0.01 ) {
                    escaped = true;
                    break;
                }
                // Ray is in plasma
                float y2    = abs_y ? fabs( pos.y ) : pos.y;
                uint32_t k1 = findindex( ptr_x, Nx, pos.x );
                uint32_t k2 = findindex( ptr_y, Ny, y2 );
                uint32_t i1 = ( k1 - 1 ) + ( k2 - 1 ) * Nx;
                uint32_t i2 = k1 + ( k2 - 1 ) * Nx;
                uint32_t i3 = ( k1 - 1 ) + k2 * Nx;
                uint32_t i4 = k1 + k2 * Nx;
                double x[2] = { ptr_x[k1 - 1], ptr_x[k1] };
                double y[2] = { ptr_y[k2 - 1], ptr_y[k2] };
                double n[4] = { ptr_n[i1], ptr_n[i2], ptr_n[i3], ptr_n[i4] };
                // Interpolate g0 and E0
                float dxi = (float) ( ( pos.x - ptr_x[k1 - 1] ) / ( ptr_x[k1] - ptr_x[k1 - 1] ) );
                float dyi = (float) ( ( y2 - ptr_y[k2 - 1] ) / ( ptr_y[k2] - ptr_y[k2 - 1] ) );
                float g0  = bilinear( dxi, dyi, ptr_g0[i1], ptr_g0[i2], ptr_g0[i3], ptr_g0[i4] );
                float E0  = use_emis ? bilinear( dxi, dyi, ptr_E0[i1], ptr_E0[i2], ptr_E0[i3],
                                          ptr_E0[i4] ) :
                                      0.0f;
                // Update x and s
                pos.z          = 0.0f;
                float range[4] = { (float) ( x[0] - 0.1 * ( ptr_x[k1] - ptr_x[k1 - 1] ) ),
                    (float) ( x[1] + 0.1 * ( ptr_x[k1] - ptr_x[k1 - 1] ) ),
                    (float) ( y[0] - 0.1 * ( ptr_y[k2] - ptr_y[k2 - 1] ) ),
                    (float) ( y[1] + 0.1 * ( ptr_y[k2] - ptr_y[k2 - 1] ) ) };
                if ( abs_y && k2 <= 1 )
                    range[2] = -range[3];
                float ds_sum = propagate2( pos, s, z_stop - z, x, y, range, n, abs_y, c );
                z += fabs( pos.z );
                // Find g(v)(x,y,z)*ds, e(v)(x,y,z)*ds
                gvl[ii - 1][is] += g0 * ds_sum;
                evl[ii - 1][is] += E0 * ds_sum;
                ivl[ii - 1][is] = i1;
            }
#ifdef RAY_DEBUG
            if ( dv != NULL && debug != NULL ) {
                int index            = N_SUB * ( ii - 1 ) + is + ( method == 1 ? 0 : 1 );
                debug[3 * index + 0] = pos.x;
                debug[3 * index + 1] = pos.y;
            }
#endif
        }
    }
    // Check if the ray is essentially travelling perpendicular to the z-direction
    if ( s.z * s.z < 0.01 )
        return -1;
    // Calculate the output ray
    ray2.x = pos.x;
    ray2.y = pos.y;
    ray2.a = atan( s.x / s.z ) * 1e3f;
    ray2.b = atan( s.y / s.z ) * 1e3f;
    // Calculate the initial intensity
    if ( seed == NULL || escaped ) {
        // No seed beam or the ray escaped the plasma column
    } else if ( method == 1 ) {
        // We are propagating backward
        double a = ray2.a;
        double b = ray2.b;
        calc_seed_inline( *seed, pos.x, pos.y, a, b, Iv );
    } else if ( method == 2 ) {
        // We are propagating forward
        calc_seed_inline( *seed, ray.x, ray.y, ray.a, ray.b, Iv );
    }
// Calculate output intensity & gain:  dI(x)/dx = j(x) + g(x)*I(x)
// If j and g are constant:  I(x) = j/g*(exp(g*x)-1)+I(0)*exp(g*x)
#ifdef RAY_DEBUG
    if ( dv != NULL && debug != NULL ) {
        debug[2] = 0.0f;
        for ( int k = 0; k < K; k++ )
            debug[2] += (float) ( 2 * Iv[k] * dv[k] );
    }
#endif
    if ( use_emis || debug != NULL ) {
        // Calculate the ray intensity with both gain and spontaneous emission (average g/e over dx)
        for ( int i = 0; i < N - 1; i++ ) {
            for ( int is = 0; is < N_SUB; is++ ) {
                const float *gv = &gain[i + 1].gv[ivl[i][is] * K];
                for ( int k = 0; k < K; k++ ) {
                    double gl = gvl[i][is] * gv[k];
                    double el = evl[i][is] * gv[k];
                    if ( fabs( gl ) < 1e-3 ) {
                        Iv[k] = el * ( 1.0 + 0.5 * gl * ( 1.0 + 0.3333333333 * gl ) ) +
                                Iv[k] * ( 1.0 + gl * ( 1.0 + 0.5 * gl ) );
                    } else {
                        double exp_gl = exp( gl );
                        Iv[k]         = el / gl * ( exp_gl - 1.0 ) + Iv[k] * exp_gl;
                    }
                }
#ifdef RAY_DEBUG
                if ( dv != NULL && debug != NULL ) {
                    int index    = 3 * ( N_SUB * i + is + 1 ) + 2;
                    debug[index] = 0.0f;
                    for ( int k = 0; k < K; k++ )
                        debug[index] += (float) ( 2 * Iv[k] * dv[k] );
                }
#endif
            }
        }
    } else {
        // Calculate the ray intensity with gain only (increases the performance)
        for ( int k = 0; k < K; k++ ) {
            double gl = 0;
            for ( int i = 0; i < N - 1; i++ ) {
                for ( int is = 0; is < N_SUB; is++ ) {
                    double gv = gain[i + 1].gv[k + ivl[i][is] * K];
                    gl += gvl[i][is] * gv;
                }
            }
            Iv[k] *= exp( gl );
        }
    }
    bool neg  = false;
    bool nans = false;
    for ( int jj = 0; jj < K; jj++ ) {
        neg  = neg || Iv[jj] < 0.0;
        nans = nans || Iv[jj] != Iv[jj];
    }
    int error = 0;
    if ( neg ) {
        error = -2;
    } else if ( nans ) {
        error = -3;
    }
    return error;
}


#endif
