#undef USE_OPENACC
#undef USE_KOKKOS
#undef USE_CUDA

#include "common/RayTraceImageHelper.h"

#if CXX_STD == 11 || CXX_STD == 14
#include <atomic>
#include <thread>
#endif


void RayTraceImageCPULoop( int N, int nx, int ny, int na, int nb, int nv, const double *x,
    const double *y, const double *a, const double *b, double dx, double dy, double dz, double da,
    double db, const double *dv, const RayTrace::ray_gain_struct *gain,
    const RayTrace::ray_seed_struct *seed, int method, const std::vector<ray_struct> &rays,
    double scale, double *image, double *I_ang, unsigned int &failure_code,
    std::vector<ray_struct> &failed_rays )
{
    failure_code = 0;

    // Loop through the rays
    for ( size_t it = 0; it < rays.size(); ++it ) {
        const ray_struct ray = rays[it];
        double Iv[K_MAX];
        ray_struct ray2;
        int error = RayTrace_calc_ray( ray, N, (float) dz, gain, seed, nv, method, Iv, ray2 );
        if ( error != 0 ) {
            failed_rays.push_back( ray );
            set_bit( -error, failure_code );
            continue;
        }
        if ( method == 1 ) {
            // We are propagating backward, use ray for the cell updates
            ray2 = ray;
        } else {
            // We are propagating forward, use ray2 for the cell updates
            // Note: The sign of the angle is reversed with respect to the euv_beam
            ray2.a = -ray2.a;
            ray2.b = -ray2.b;
            if ( ray2.y < 0.0 && y[0] >= 0.0 ) {
                // We need to change the sign of y
                ray2.y = -ray2.y;
            }
        }
        // Get the indicies to the cells in image and I_ang
        // Note: do not replace these lines with findindex, we need to be able to return 0 for the
        // index
        int i1 = static_cast<int>( findfirstsingle( x, nx, ray2.x - 0.5 * dx ) );
        int i2 = static_cast<int>( findfirstsingle( y, ny, ray2.y - 0.5 * dy ) );
        int i3 = static_cast<int>( findfirstsingle( a, na, ray2.a - 0.5 * da ) );
        int i4 = static_cast<int>( findfirstsingle( b, nb, ray2.b - 0.5 * db ) );
        if ( ray2.x < x[0] - 0.5 * dx || ray2.x > x[nx - 1] + 0.5 * dx )
            i1 = -1; // The ray's z position is out of the range of image
        if ( ray2.y < y[0] - 0.5 * dy || ray2.y > y[ny - 1] + 0.5 * dy )
            i2 = -1; // The ray's y position is out of the range of image
        if ( -ray2.a < a[0] - 0.5 * da || -ray2.a > a[na - 1] + 0.5 * da )
            i3 = -1; // The ray's z angle is out of the range of I_ang
        if ( -ray2.b < b[0] - 0.5 * db || -ray2.b > b[nb - 1] + 0.5 * db )
            i4 = -1; // The ray's y angle is out of the range of I_ang
        // Copy I_out into image
        if ( i1 >= 0 && i2 >= 0 ) {
            double *Iv2 = &image[nv * ( i1 + i2 * nx )];
            for ( int iv = 0; iv < nv; iv++ ) {
                Iv2[iv] += Iv[iv] * scale;
            }
        }
        // Copy I_out into I_ang
        if ( i3 >= 0 && i4 >= 0 ) {
            double tmp = 0.0;
            for ( int iv = 0; iv < nv; iv++ )
                tmp += 2.0 * dv[iv] * Iv[iv];
            I_ang[i3 + i4 * na] += tmp;
        }
    }
}


#ifdef USE_OPENMP
void RayTraceImageOpenMPLoop( int N, int nx, int ny, int na, int nb, int nv, const double *x,
    const double *y, const double *a, const double *b, double dx, double dy, double dz, double da,
    double db, const double *dv, const RayTrace::ray_gain_struct *gain,
    const RayTrace::ray_seed_struct *seed, int method, const std::vector<ray_struct> &rays,
    double scale, double *image, double *I_ang, unsigned int &failure_code,
    std::vector<ray_struct> &failed_rays )
{
    failure_code = 0;

// Loop through the rays
#pragma omp parallel for
    for ( int it = 0; it < (int) rays.size(); ++it ) {
        const ray_struct ray = rays[it];
        double Iv[K_MAX];
        ray_struct ray2;
        int error = RayTrace_calc_ray( ray, N, (float) dz, gain, seed, nv, method, Iv, ray2 );
        if ( error != 0 ) {
            failed_rays.push_back( ray );
            set_bit( -error, failure_code );
            continue;
        }
        if ( method == 1 ) {
            // We are propagating backward, use ray for the cell updates
            ray2 = ray;
        } else {
            // We are propagating forward, use ray2 for the cell updates
            // Note: The sign of the angle is reversed with respect to the euv_beam
            ray2.a = -ray2.a;
            ray2.b = -ray2.b;
            if ( ray2.y < 0.0 && y[0] >= 0.0 ) {
                // We need to change the sign of y
                ray2.y = -ray2.y;
            }
        }
        // Get the indicies to the cells in image and I_ang
        // Note: do not replace these lines with findindex, we need to be able to return 0 for the
        // index
        int i1 = static_cast<int>( findfirstsingle( x, nx, ray2.x - 0.5 * dx ) );
        int i2 = static_cast<int>( findfirstsingle( y, ny, ray2.y - 0.5 * dy ) );
        int i3 = static_cast<int>( findfirstsingle( a, na, ray2.a - 0.5 * da ) );
        int i4 = static_cast<int>( findfirstsingle( b, nb, ray2.b - 0.5 * db ) );
        if ( ray2.x < x[0] - 0.5 * dx || ray2.x > x[nx - 1] + 0.5 * dx )
            i1 = -1; // The ray's z position is out of the range of image
        if ( ray2.y < y[0] - 0.5 * dy || ray2.y > y[ny - 1] + 0.5 * dy )
            i2 = -1; // The ray's y position is out of the range of image
        if ( -ray2.a < a[0] - 0.5 * da || -ray2.a > a[na - 1] + 0.5 * da )
            i3 = -1; // The ray's z angle is out of the range of I_ang
        if ( -ray2.b < b[0] - 0.5 * db || -ray2.b > b[nb - 1] + 0.5 * db )
            i4 = -1; // The ray's y angle is out of the range of I_ang
        // Copy I_out into image
        if ( i1 >= 0 && i2 >= 0 ) {
            double *Iv2 = &image[nv * ( i1 + i2 * nx )];
            for ( int iv = 0; iv < nv; iv++ ) {
#pragma omp atomic
                Iv2[iv] += Iv[iv] * scale;
            }
        }
        // Copy I_out into I_ang
        if ( i3 >= 0 && i4 >= 0 ) {
            double tmp = 0.0;
            for ( int iv = 0; iv < nv; iv++ )
                tmp += 2.0 * dv[iv] * Iv[iv];
#pragma omp atomic
            I_ang[i3 + i4 * na] += tmp;
        }
    }
}
#endif


#if CXX_STD == 11 || CXX_STD == 14
void RayTraceImageThreadLoop( int N, int nx, int ny, int na, int nb, int nv, const double *x,
    const double *y, const double *a, const double *b, double dx, double dy, double dz, double da,
    double db, const double *dv, const RayTrace::ray_gain_struct *gain,
    const RayTrace::ray_seed_struct *seed, int method, const std::vector<ray_struct> &rays,
    double scale, double *image, double *I_ang, unsigned int &failure_code,
    std::vector<ray_struct> &failed_rays )
{
    size_t N_threads = std::thread::hardware_concurrency();
    std::vector<std::vector<ray_struct>> rays2( N_threads );
    std::vector<std::vector<ray_struct>> failed_rays2( N_threads );
    std::vector<unsigned int> failure_code2( N_threads, 0 );
    std::vector<double *> image2( N_threads, NULL );
    std::vector<double *> I_ang2( N_threads, NULL );
    std::vector<std::thread> threads;
    for ( size_t i = 0, j = 0; i < N_threads; i++ ) {
        size_t N2 = rays.size() / N_threads + 1;
        if ( N2 == 0 ) {
            N2 = 8;
        }
        rays2.reserve( N2 );
        for ( size_t k = 0; k < N2 && j < rays.size(); k++, j++ )
            rays2[i].push_back( rays[j] );
        image2[i] = (double *) calloc( nx * ny * nv, sizeof( double ) );
        I_ang2[i] = (double *) calloc( na * nb, sizeof( double ) );
        threads.push_back( std::thread( RayTraceImageCPULoop, N, nx, ny, na, nb, nv, x, y, a, b, dx,
            dy, dz, da, db, dv, gain, seed, method, std::ref( rays2[i] ), scale, image2[i],
            I_ang2[i], std::ref( failure_code2[i] ), std::ref( failed_rays2[i] ) ) );
    }
    for ( size_t i = 0; i < N_threads; i++ ) {
        threads[i].join();
        for ( int j = 0; j < nx * ny * nv; j++ )
            image[j] += image2[i][j];
        for ( int j = 0; j < na * nb; j++ )
            I_ang[j] += I_ang2[i][j];
        failure_code = failure_code | failure_code2[i];
        for ( size_t j = 0; j < failed_rays2[i].size(); j++ )
            failed_rays.push_back( failed_rays2[i][j] );
        free( image2[i] );
        free( I_ang2[i] );
        rays2[i].clear();
    }
}
#endif
