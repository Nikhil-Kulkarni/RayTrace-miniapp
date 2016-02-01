#include "common/RayTraceImageHelper.h"


void RayTraceImageOpenAccLoop( int N, int nx, int ny, int na, int nb, int nv, const double *x,
    const double *y, const double *a, const double *b, double dx, double dy, double dz, double da,
    double db, const double *dv, const RayTrace::ray_gain_struct *gain_in,
    const RayTrace::ray_seed_struct *seed_in, int method, const std::vector<ray_struct> &rays,
    double scale, double *image, double *I_ang, unsigned int &failure_code,
    std::vector<ray_struct> &failed_rays )
{
    int N_rays              = rays.size();
    const ray_struct *rays2 = &rays[0];
    failure_code            = 0;

    // place the ray gain and seed structures on the device
    const RayTrace::ray_gain_struct *gain = RayTrace::ray_gain_struct::copy_device( N, gain_in );
    const RayTrace::ray_seed_struct *seed = NULL;
    if ( seed_in != NULL )
        seed = seed_in->copy_device();

#pragma acc data copyin( x[0 : nx],                                                          \
    y[0 : ny], a[0 : na], b[0 : nb], dv[0 : nv], rays2[0 : N_rays] ) deviceptr( gain, seed ) \
                                             copyout( image[0 : nx *ny *nv], I_ang[0 : na *nb] )
    {

// Initialize device images
#pragma acc parallel loop
        for ( int i  = 0; i < nx * ny * nv; ++i )
            image[i] = 0;
#pragma acc parallel loop
        for ( int i  = 0; i < na * nb; ++i )
            I_ang[i] = 0;
// Loop through y, x, b, a
#pragma acc parallel loop gang vector vector_length( 32 )
        for ( int it = 0; it < N_rays; ++it ) {
            const ray_struct ray = rays2[it];
            double Iv[K_MAX];
            ray_struct ray2;
            int error = RayTrace_calc_ray( ray, N, dz, gain, seed, nv, method, Iv, ray2 );
            if ( error != 0 ) {
                // failed_rays.push_back(ray);
                // set_bit(-error,failure_code);
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
            // Note: do not replace these lines with findindex, we need to be able to return 0 for
            // the index
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
#pragma acc atomic update
                    Iv2[iv] += Iv[iv] * scale;
                }
            }
            // Copy I_out into I_ang
            if ( i3 >= 0 && i4 >= 0 ) {
                double tmp = 0.0;
                for ( int iv = 0; iv < nv; iv++ )
                    tmp += 2.0 * dv[iv] * Iv[iv];
#pragma acc atomic update
                I_ang[i3 + i4 * na] += tmp;
            }
        }

    } // pragma acc data region scope

    // Free device pointers
    RayTrace::ray_gain_struct::free_device( N, gain_in, gain );
    RayTrace::ray_seed_struct::free_device( seed_in, seed );
}
