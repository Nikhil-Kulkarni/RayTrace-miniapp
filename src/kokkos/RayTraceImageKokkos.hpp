#include "Kokkos_Core.hpp"
#include "common/RayTraceImageHelper.h"

template <class Device>
struct RayTraceImageKokkosKernel {
    // Typedefs
    typedef const Kokkos::View<const double *, Device> ConstDblVec;

    // Member variables
    const int N, nx, ny, na, nb, nv;
    ConstDblVec x0, y0, a0, b0;
    const double dx, dy, dz, da, db;
    ConstDblVec dv0;
    const RayTrace::ray_gain_struct *gain;
    const RayTrace::ray_seed_struct *seed;
    int method;
    const Kokkos::View<const ray_struct *, Device> rays;
    const double scale;
    Kokkos::View<double *, Device> image, I_ang;

    // Constructor
    RayTraceImageKokkosKernel( int N_, int nx_, int ny_, int na_, int nb_, int nv_, ConstDblVec x_,
        ConstDblVec y_, ConstDblVec a_, ConstDblVec b_, const double dx_, const double dy_,
        const double dz_, const double da_, const double db_, ConstDblVec dv_,
        const RayTrace::ray_gain_struct *gain_, const RayTrace::ray_seed_struct *seed_, int method_,
        const Kokkos::View<const ray_struct *, Device> rays_, double scale_,
        Kokkos::View<double *, Device> image_, Kokkos::View<double *, Device> I_ang_ )
        : N( N_ ),
          nx( nx_ ),
          ny( ny_ ),
          na( na_ ),
          nb( nb_ ),
          nv( nv_ ),
          x0( x_ ),
          y0( y_ ),
          a0( a_ ),
          b0( b_ ),
          dx( dx_ ),
          dy( dy_ ),
          dz( dz_ ),
          da( da_ ),
          db( db_ ),
          dv0( dv_ ),
          gain( gain_ ),
          seed( seed_ ),
          method( method_ ),
          rays( rays_ ),
          scale( scale_ ),
          image( image_ ),
          I_ang( I_ang_ )
    {
    }

    // Operator
    KOKKOS_INLINE_FUNCTION
    void operator()( const int it ) const
    {
        const ray_struct ray = rays( it );
        const double *x      = &x0( 0 );
        const double *y      = &y0( 0 );
        const double *a      = &a0( 0 );
        const double *b      = &b0( 0 );
        const double *dv     = &dv0( 0 );
        double Iv[K_MAX];
        ray_struct ray2;
        int error = RayTrace_calc_ray( ray, N, dz, gain, seed, nv, method, Iv, ray2 );
        if ( error != 0 ) {
            // failed_rays.push_back(ray);
            // set_bit(-error,failure_code);
            return;
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
                Kokkos::atomic_fetch_add( &Iv2[iv], Iv[iv] * scale );
            }
        }
        // Copy I_out into I_ang
        if ( i3 >= 0 && i4 >= 0 ) {
            double tmp = 0.0;
            for ( int iv = 0; iv < nv; iv++ )
                tmp += 2.0 * dv[iv] * Iv[iv];
            Kokkos::atomic_fetch_add( &I_ang[i3 + i4 * na], tmp );
        }
    }
};


template <class Device>
const RayTrace::ray_gain_struct *copy_gain_device( int, const RayTrace::ray_gain_struct *gain )
{
    return gain;
}
template <class Device>
const RayTrace::ray_seed_struct *copy_seed_device( const RayTrace::ray_seed_struct *seed )
{
    return seed;
}
template <class Device>
void free_gain_device( int, const RayTrace::ray_gain_struct * )
{
}
template <class Device>
void free_seed_device( const RayTrace::ray_seed_struct * )
{
}
#if defined( KOKKOS_HAVE_CUDA )
template <>
const RayTrace::ray_gain_struct *copy_gain_device<Kokkos::Cuda>(
    int N, const RayTrace::ray_gain_struct *gain )
{
    return RayTrace::ray_gain_struct::copy_device( N, gain );
}
template <>
const RayTrace::ray_seed_struct *copy_seed_device<Kokkos::Cuda>(
    const RayTrace::ray_seed_struct *seed_in )
{
    const RayTrace::ray_seed_struct *seed = NULL;
    if ( seed_in != NULL )
        seed = seed_in->copy_device();
    return seed;
}
template <>
void free_gain_device<Kokkos::Cuda>( int N, const RayTrace::ray_gain_struct *gain )
{
    RayTrace::ray_gain_struct::free_device( N, NULL, gain );
}
template <>
void free_seed_device<Kokkos::Cuda>( const RayTrace::ray_seed_struct *seed )
{
    RayTrace::ray_seed_struct::free_device( NULL, seed );
}
#endif




template <class Device>
void RayTraceImageKokkosLoop( int N, const RayTrace::EUV_beam_struct& beam,
    const RayTrace::ray_gain_struct *gain_in,
    const RayTrace::ray_seed_struct *seed_in, int method, const std::vector<ray_struct> &rays,
    double scale, double *image, double *I_ang, unsigned int &failure_code,
    std::vector<ray_struct> &failed_rays )
{
    failure_code = 0;
    const int nx = beam.nx;
    const int ny = beam.ny;
    const int na = beam.na;
    const int nb = beam.nb;
    const int nv = beam.nv;
    const double dx = beam.dx;
    const double dy = beam.dy;
    const double dz = beam.dz;
    const double da = beam.da;
    const double db = beam.db;
    // Get gain and seed on the device
    const RayTrace::ray_gain_struct *gain = copy_gain_device<Device>( N, gain_in );
    const RayTrace::ray_seed_struct *seed = copy_seed_device<Device>( seed_in );
    // Copy the data to the device
    typedef Kokkos::View<const double *, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> KokkosConstHostDouble;
    typedef Kokkos::View<const ray_struct *, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> KokkosConstHostRay;
    typedef Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> KokkosHostDouble;
    typedef Kokkos::View<double *, Kokkos::LayoutRight, Device> KokkosDeviceDouble;
    typedef Kokkos::View<ray_struct *, Kokkos::LayoutRight, Device> KokkosDeviceRay;
    KokkosConstHostDouble x_host( beam.x, nx );
    KokkosConstHostDouble y_host( beam.y, ny );
    KokkosConstHostDouble a_host( beam.a, na );
    KokkosConstHostDouble b_host( beam.b, nb );
    KokkosConstHostDouble dv_host( beam.dv, nv );
    KokkosConstHostRay rays_host( &rays[0], rays.size() );
    KokkosHostDouble image_host( image, nx * ny * nv );
    KokkosHostDouble I_ang_host( I_ang, na * nb );
    KokkosDeviceDouble x_dev( "x", nx );
    KokkosDeviceDouble y_dev( "y", ny );
    KokkosDeviceDouble a_dev( "a", na );
    KokkosDeviceDouble b_dev( "b", nb );
    KokkosDeviceDouble dv_dev( "dv", nv );
    KokkosDeviceRay rays_dev( "rays", rays.size() );
    KokkosDeviceDouble image_dev( "image", nx * ny * nv );
    KokkosDeviceDouble I_ang_dev( "I_ang", na * nb );
    Kokkos::deep_copy( x_dev, x_host );
    Kokkos::deep_copy( y_dev, y_host );
    Kokkos::deep_copy( a_dev, a_host );
    Kokkos::deep_copy( b_dev, b_host );
    Kokkos::deep_copy( dv_dev, dv_host );
    Kokkos::deep_copy( rays_dev, rays_host );
    Kokkos::deep_copy( image_dev, image_host );
    Kokkos::deep_copy( I_ang_dev, I_ang_host );
    // Call the parallel for loop
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Device>( 0, rays.size() ),
        RayTraceImageKokkosKernel<Device>( N, nx, ny, na, nb, nv, x_dev, y_dev, a_dev, b_dev, dx,
            dy, dz, da, db, dv_dev, gain, seed, method, rays_dev, scale, image_dev, I_ang_dev ) );
    // Copy the results back to the CPU
    Kokkos::deep_copy( image_host, image_dev );
    Kokkos::deep_copy( I_ang_host, I_ang_dev );
    // Free gain and seed on the device
    free_gain_device<Device>( N, gain );
    free_seed_device<Device>( seed );
}
