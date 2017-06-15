#undef USE_OPENACC
#undef USE_KOKKOS
#undef USE_CUDA

#include "common/RayTraceImageHelper.h"

#include <atomic>
#include <thread>


inline int getIndex( int n, const double *x, double dx, double y )
{
    if ( y < x[0] - 0.5 * dx || y > x[n-1] + 0.5 * dx )
        return -1;
    return findfirstsingle( x, n, y - 0.5 * dx );
}


void RayTraceImageCPULoop( int N, const RayTrace::EUV_beam_struct& beam,
    const RayTrace::ray_gain_struct *gain, const RayTrace::ray_seed_struct *seed,
    int method, const std::vector<ray_struct> &rays, double scale, double *image,
    double *I_ang, unsigned int &failure_code, std::vector<ray_struct> &failed_rays )
{
    failure_code = 0;

    // Loop through the rays
    for ( size_t it = 0; it < rays.size(); ++it ) {
        const ray_struct ray = rays[it];
        double Iv[K_MAX];
        ray_struct ray2;
        int error = RayTrace_calc_ray( ray, N, beam.dz, gain, seed, beam.nv, method, Iv, ray2 );
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
            if ( ray2.y < 0.0 && beam.y[0] >= 0.0 ) {
                // We need to change the sign of y
                ray2.y = -ray2.y;
            }
        }
        // Get the indicies to the cells in image and I_ang
        int i1 = getIndex( beam.nx, beam.x, beam.dx, ray2.x );
        int i2 = getIndex( beam.ny, beam.y, beam.dy, ray2.y );
        int i3 = getIndex( beam.na, beam.a, beam.da, ray2.a );
        int i4 = getIndex( beam.nb, beam.b, beam.db, ray2.b );
        // Copy I_out into image
        if ( i1 >= 0 && i2 >= 0 ) {
            double *Iv2 = &image[beam.nv * ( i1 + i2 * beam.nx )];
            for ( int iv = 0; iv < beam.nv; iv++ ) {
                Iv2[iv] += Iv[iv] * scale;
            }
        }
        // Copy I_out into I_ang
        if ( i3 >= 0 && i4 >= 0 ) {
            double tmp = 0.0;
            for ( int iv = 0; iv < beam.nv; iv++ )
                tmp += 2.0 * beam.dv[iv] * Iv[iv];
            I_ang[i3 + i4 * beam.na] += tmp;
        }
    }
}


#ifdef USE_OPENMP
void RayTraceImageOpenMPLoop( int N, const RayTrace::EUV_beam_struct& beam,
    const RayTrace::ray_gain_struct *gain, const RayTrace::ray_seed_struct *seed,
    int method, const std::vector<ray_struct> &rays, double scale, double *image,
    double *I_ang, unsigned int &failure_code, std::vector<ray_struct> &failed_rays )
{
    failure_code = 0;

// Loop through the rays
#pragma omp parallel for
    for ( int it = 0; it < (int) rays.size(); ++it ) {
        const ray_struct ray = rays[it];
        double Iv[K_MAX];
        ray_struct ray2;
        int error = RayTrace_calc_ray( ray, N, beam.dz, gain, seed, beam.nv, method, Iv, ray2 );
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
            if ( ray2.y < 0.0 && beam.y[0] >= 0.0 ) {
                // We need to change the sign of y
                ray2.y = -ray2.y;
            }
        }
        // Get the indicies to the cells in image and I_ang
        int i1 = getIndex( beam.nx, beam.x, beam.dx, ray2.x );
        int i2 = getIndex( beam.ny, beam.y, beam.dy, ray2.y );
        int i3 = getIndex( beam.na, beam.a, beam.da, ray2.a );
        int i4 = getIndex( beam.nb, beam.b, beam.db, ray2.b );
        // Copy I_out into image
        if ( i1 >= 0 && i2 >= 0 ) {
            double *Iv2 = &image[beam.nv * ( i1 + i2 * beam.nx )];
            for ( int iv = 0; iv < beam.nv; iv++ ) {
#pragma omp atomic
                Iv2[iv] += Iv[iv] * scale;
            }
        }
        // Copy I_out into I_ang
        if ( i3 >= 0 && i4 >= 0 ) {
            double tmp = 0.0;
            for ( int iv = 0; iv < beam.nv; iv++ )
                tmp += 2.0 * beam.dv[iv] * Iv[iv];
#pragma omp atomic
            I_ang[i3 + i4 * beam.na] += tmp;
        }
    }
}
#endif


