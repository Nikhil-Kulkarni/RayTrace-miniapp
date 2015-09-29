// This file contains the source code for propagating the rays for a single thread

#define NOMINMAX
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <algorithm>

// Timer include
#include "ProfilerApp.h"

// AtomicModel includes
#ifndef USE_OPENACC
    #include "interp.h"
#endif
#ifndef DISABLE_WRITE_FAILED_RAYS
    #include "file_utilities.h"
#endif

// RayTrace includes
#include "RayTrace.h"
#include "utilities/RayUtilities.h"

// Include helper files
// Note: we need to undefine macros due to issues with OpenACC
#ifdef USE_OPENACC
    #define ENABLE_OPENACC
    #undef USE_OPENACC
#endif
#ifdef USE_KOKKOS
    #define ENABLE_KOKKOS
    #undef USE_KOKKOS
#endif
#ifdef USE_CUDA
    #define ENABLE_CUDA
    #undef USE_CUDA
#endif
#include "RayTraceImageHelper.h"

// Define the external loop interfaces
extern void RayTraceImageCPULoop( int N, int nx, int ny, int na, int nb, int nv, 
    const double *x, const double *y, const double *a, const double *b,
    double dx, double dy, double dz, double da, double db, const double *dv,
    const RayTrace::ray_gain_struct* gain, const RayTrace::ray_seed_struct* seed,
    int method, const std::vector<ray_struct>& rays,  double scale,
    double *image, double *I_ang, 
    unsigned int& failure_code, std::vector<ray_struct>& failed_rays );
#if CXX_STD==11 || CXX_STD==14
    extern void RayTraceImageThreadLoop( int N, int nx, int ny, int na, int nb, int nv, 
        const double *x, const double *y, const double *a, const double *b,
        double dx, double dy, double dz, double da, double db, const double *dv,
        const RayTrace::ray_gain_struct* gain, const RayTrace::ray_seed_struct* seed,
        int method, const std::vector<ray_struct>& rays,  double scale,
        double *image, double *I_ang, 
        unsigned int& failure_code, std::vector<ray_struct>& failed_rays );
#endif
#if defined(ENABLE_OPENACC)
    extern void RayTraceImageOpenAccLoop( int N, int nx, int ny, int na, int nb, int nv, 
        const double *x, const double *y, const double *a, const double *b,
        double dx, double dy, double dz, double da, double db, const double *dv,
        const RayTrace::ray_gain_struct* gain, const RayTrace::ray_seed_struct* seed,
        int method, const std::vector<ray_struct>& rays,  double scale,
        double *image, double *I_ang, 
        unsigned int& failure_code, std::vector<ray_struct>& failed_rays );
#endif
#if defined(ENABLE_KOKKOS)
    #include "kokkos/RayTraceImageKokkos.hpp"
    /*extern void RayTraceImageKokkosLoop( int N, int nx, int ny, int na, int nb, int nv, 
        const double *x, const double *y, const double *a, const double *b,
        double dx, double dy, double dz, double da, double db, const double *dv,
        const RayTrace::ray_gain_struct* gain, const RayTrace::ray_seed_struct* seed,
        int method, const std::vector<ray_struct>& rays,  double scale,
        double *image, double *I_ang, 
        unsigned int& failure_code, std::vector<ray_struct>& failed_rays );*/
#endif
#if defined(ENABLE_CUDA)
    extern void RayTraceImageCudaLoop( int N, int nx, int ny, int na, int nb, int nv, 
        const double *x, const double *y, const double *a, const double *b,
        double dx, double dy, double dz, double da, double db, const double *dv,
        const RayTrace::ray_gain_struct* gain, const RayTrace::ray_seed_struct* seed,
        int method, const std::vector<ray_struct>& rays,  double scale,
        double *image, double *I_ang, 
        unsigned int& failure_code, std::vector<ray_struct>& failed_rays );
#endif


/**********************************************************************
* Write the failed rays to a file                                     *
**********************************************************************/
static void write_failures( unsigned int failure_code, const std::vector<ray_struct>& failed_rays, 
    int method, int N, double dz, const RayTrace::ray_gain_struct *gain )
{
    if ( check_bit(1,failure_code) )
        perr << "Invalid ray detected\n";
    if ( check_bit(2,failure_code) )
        perr << "Negitive intensity detected\n";
    if ( check_bit(3,failure_code) )
        perr << "NaNs detected in intensity\n";
    #ifndef DISABLE_WRITE_FAILED_RAYS
        FILE *fid = fopen("Failed_RayTrace_rays.dat","wb");
        if ( fid != NULL ) {
            // Write the failed rays
            double *tmp = new double[failed_rays.size()*4];
            for (size_t i=0; i<failed_rays.size(); i++) {
                tmp[4*i+0] = failed_rays[i].x;
                tmp[4*i+1] = failed_rays[i].y;
                tmp[4*i+2] = failed_rays[i].a;
                tmp[4*i+3] = failed_rays[i].b;
            }
            file_utilities::write_variable_double( fid, "rays", failed_rays.size()*4, tmp, false );
            delete [] tmp;
            // Write some basic data used for the ray trace
            file_utilities::write_scalar_int( fid, "method", method );
            file_utilities::write_scalar_int( fid, "N", N );
            file_utilities::write_scalar_double( fid, "dz", dz );
            // Write the gain structure
            for (int i=0; i<N; i++) {
                char prefix[20];
                sprintf(prefix,"gain[%i].",i);
                gain[i].writeData( fid, prefix );
            }
            // Finished writing debug data
            perr << "Failed rays written to Failed_RayTrace_rays.dat\n";
            fclose(fid);
        }
    #else
        NULL_USE(failed_rays);
        NULL_USE(method);
        NULL_USE(N);
        NULL_USE(dz);
        NULL_USE(gain);
    #endif
}


/**********************************************************************
* Call calc_ray                                                       *
**********************************************************************/
int RayTrace::calc_ray( const double ray_in[4], const int N, const double dz0, const ray_gain_struct *gain, 
    const ray_seed_struct *seed, int K, int method, double *Iv, double *ray_out ) 
{
    ray_struct ray, ray2;
    ray.x = (float) ray_in[0];
    ray.y = (float) ray_in[1];
    ray.a = (float) ray_in[2];
    ray.b = (float) ray_in[3];
    int error = RayTrace_calc_ray( ray, N, (float) dz0, gain, seed, K, method, Iv, ray2 );
    ray_out[0] = ray2.x;
    ray_out[1] = ray2.y;
    ray_out[2] = ray2.a;
    ray_out[3] = ray2.b;
    return error;
}


/**********************************************************************
* Call calc_seed                                                      *
**********************************************************************/
void RayTrace::calc_seed( const ray_seed_struct& seed, 
    double x, double y, double a, double b, double *Iv ) 
{
    calc_seed_inline(seed,x,y,a,b,Iv);
}


/**********************************************************************
* Create the image                                                    *
**********************************************************************/
inline bool check_grid( int N, double dx, const double *x ) {
    bool error = false;
    for (int i=1; i<N; i++)
        error = error || (fabs((x[i]-x[i-1])-dx)>1e-12*dx);
    return error;
}
void RayTrace::create_image( create_image_struct *info, std::string compute_method ) 
{
    if ( info->N > N_MAX )
        RAY_ERROR("Exceeded maximum number of length segments");
    if ( info->euv_beam->nv >= K_MAX )
        RAY_ERROR("Exceeded maximum number of frequencies");
    PROFILE_START("create_image");

    // Copy some variables
    const int N = info->N;
    const int nx = info->euv_beam->nx;
    const int ny = info->euv_beam->ny;
    const int na = info->euv_beam->na;
    const int nb = info->euv_beam->nb;
    const int nv = info->euv_beam->nv;
    const double *x = info->euv_beam->x;
    const double *y = info->euv_beam->y;
    const double *a = info->euv_beam->a;
    const double *b = info->euv_beam->b;
    const double dx = info->euv_beam->dx;
    const double dy = info->euv_beam->dy;
    const double dz = info->dz;
    const double da = info->euv_beam->da;
    const double db = info->euv_beam->db;
    const double *dv = info->euv_beam->dv;
    
    // Check the euv_beam grid
    bool grid_error = false;
    grid_error = grid_error || check_grid(nx,dx,x);
    grid_error = grid_error || check_grid(ny,dy,y);
    grid_error = grid_error || check_grid(na,da,a);
    grid_error = grid_error || check_grid(nb,db,b);
    if ( grid_error )
        RAY_ERROR("Only uniform grid spacings are currently supported (euv_beam)");
    if ( dz!=info->euv_beam->dz )
        RAY_ERROR("info.dz != euv_beam.dz");

    // Check the seed_beam grid
    if ( info->seed_beam!=NULL ) {
        const seed_beam_struct *seed_beam = info->seed_beam;
        grid_error = false;
        grid_error = grid_error || check_grid(seed_beam->nx,seed_beam->dx,seed_beam->x);
        grid_error = grid_error || check_grid(seed_beam->ny,seed_beam->dy,seed_beam->y);
        grid_error = grid_error || check_grid(seed_beam->na,seed_beam->da,seed_beam->a);
        grid_error = grid_error || check_grid(seed_beam->nb,seed_beam->db,seed_beam->b);
        if ( grid_error )
            RAY_ERROR("Only uniform grid spacings are currently supported (seed_beam)");
        if ( (info->euv_beam->y[0]>=0.0) != (seed_beam->y[0]>=0.0) )
            RAY_ERROR("Negitive y positions in seed_beam or euv_beam, but not both");
    }
	
    // Create image and I_ang 
    // Note: calloc will initialize the memory to zero but may delay the paging until access
    //    which may give better performance than new followed by memset(x,0,N*sizeof(double))
    // Note: Technically using memset (or calloc) to initialize double may not be correct,
    //    but is seems to work everywhere I've tried
    double *image = (double*) calloc(nx*ny*nv,sizeof(double));
    double *I_ang = (double*) calloc(na*nb,sizeof(double));
    info->image = image;
    info->I_ang = I_ang;

    
    // Create a list of rays to propagate
    std::vector<ray_struct> rays;
    double scale = 0;
    int method = 0;
    int N2[4] = {nx,ny,na,nb};
    std::string timer_name;
    if ( info->seed!=NULL ) {
        method = 2;
        N2[0] = info->seed_beam->nx;
        N2[1] = info->seed_beam->ny;
        N2[2] = info->seed_beam->na;
        N2[3] = info->seed_beam->nb;
        double seed_dx = info->seed_beam->dx;
        double seed_dy = info->seed_beam->dy;
        double seed_da = info->seed_beam->da;
        double seed_db = info->seed_beam->db;
        scale = (seed_dx*seed_dy*seed_da*seed_db)/(dx*dy);
        timer_name = "propagate_seed";
    } else {
        method = 1;
        scale = 1.0;
        timer_name = "propagate_ASE";
    }
    const int skip = info->N_parallel;
    const int offset = info->N_start;
    int Nt = N2[0]*N2[1]*N2[2]*N2[3];
    rays.reserve((Nt/skip)+1);
    for (int it=0; it<(Nt/skip)+1; ++it) {
        int ijkm = offset + it*skip;
        if ( ijkm >= Nt ) { continue; }
        // Get the indicies
        int m =  ijkm%N2[3];                // b
        int k = (ijkm/N2[3])%N2[2];         // a
        int j = (ijkm/(N2[2]*N2[3]))%N2[1]; // y
        int i =  ijkm/(N2[1]*N2[2]*N2[3]);  // x
        // Create ray information
        ray_struct ray;
        if ( info->seed_beam==NULL ) {
            ray.x = (float) x[i];
            ray.y = (float) y[j];
            ray.a = (float) a[k];
            ray.b = (float) b[m];
        } else {
            ray.x = (float) info->seed_beam->x[i];
            ray.y = (float) info->seed_beam->y[j];
            ray.a = (float) info->seed_beam->a[k];
            ray.b = (float) info->seed_beam->b[m];
        }
        rays.push_back(ray);
    }

    // Perform the ray-trace
    std::vector<ray_struct> failed_rays;
    unsigned int failure_code = 0;
    std::transform(compute_method.begin(),compute_method.end(),compute_method.begin(),::tolower);
    if ( compute_method=="auto" ) {
        #if defined(ENABLE_OPENACC)
            compute_method="openacc";
        #elif defined(ENABLE_KOKKOS)
            compute_method="kokkos";
        #elif defined(ENABLE_CUDA)
            compute_method="cuda";
        #else
            compute_method="cpu";
        #endif
    }
    timer_name += "-" + compute_method;
    PROFILE_START(timer_name);
    if ( compute_method=="openacc" ) {
        #if defined(ENABLE_OPENACC)
            RayTraceImageOpenAccLoop( N, nx, ny, na, nb, nv, x, y, a, b,
                dx, dy, dz, da, db, dv, info->gain, info->seed, method, rays, scale,
                image, I_ang, failure_code, failed_rays );
        #else
            RAY_ERROR("OpenAcc is not availible");
        #endif
    } else if ( compute_method.substr(0,7)=="kokkos-" ) {
        #if defined(ENABLE_KOKKOS)
            if ( compute_method=="kokkos-serial" ) {
                RayTraceImageKokkosLoop<Kokkos::Serial>( N, nx, ny, na, nb, nv, x, y, a, b,
                    dx, dy, dz, da, db, dv, info->gain, info->seed, method, rays, scale,
                    image, I_ang, failure_code, failed_rays );
            } else if ( compute_method=="kokkos-openmp" ) {
                #ifdef KOKKOS_HAVE_OPENMP
                    RayTraceImageKokkosLoop<Kokkos::OpenMP>( N, nx, ny, na, nb, nv, x, y, a, b,
                        dx, dy, dz, da, db, dv, info->gain, info->seed, method, rays, scale,
                        image, I_ang, failure_code, failed_rays );
                #else
                    RAY_ERROR("Kokkos compiled without OpenMP");
                #endif
            } else if ( compute_method=="kokkos-thread" ) {
                #ifdef KOKKOS_HAVE_PTHREAD
                    RayTraceImageKokkosLoop<Kokkos::Threads>( N, nx, ny, na, nb, nv, x, y, a, b,
                        dx, dy, dz, da, db, dv, info->gain, info->seed, method, rays, scale,
                        image, I_ang, failure_code, failed_rays );
                #else
                    RAY_ERROR("Kokkos compiled without OpenMP");
                #endif
            } else if ( compute_method=="kokkos-cuda" ) {
                #ifdef KOKKOS_HAVE_CUDA
                    RayTraceImageKokkosLoop<Kokkos::Cuda>( N, nx, ny, na, nb, nv, x, y, a, b,
                        dx, dy, dz, da, db, dv, info->gain, info->seed, method, rays, scale,
                        image, I_ang, failure_code, failed_rays );
                #else
                    RAY_ERROR("Kokkos compiled without OpenMP");
                #endif
            } else {
                RAY_ERROR("Unknown kokkos method");
            }
        #else
            RAY_ERROR("Kokkos is not availible");
        #endif
    } else if (compute_method=="cuda" ) {
        #if defined(ENABLE_CUDA)
            RayTraceImageCudaLoop( N, nx, ny, na, nb, nv, x, y, a, b,
                dx, dy, dz, da, db, dv, info->gain, info->seed, method, rays, scale,
                image, I_ang, failure_code, failed_rays );
        #else
            RAY_ERROR("Cuda is not availible");
        #endif
    } else if ( compute_method=="cpu" ) {
        RayTraceImageCPULoop( N, nx, ny, na, nb, nv, x, y, a, b,
            dx, dy, dz, da, db, dv, info->gain, info->seed, method, rays, scale,
            image, I_ang, failure_code, failed_rays );
    } else if ( compute_method=="threads" ) {
        #if CXX_STD==11 || CXX_STD==14
            RayTraceImageThreadLoop( N, nx, ny, na, nb, nv, x, y, a, b,
                dx, dy, dz, da, db, dv, info->gain, info->seed, method, rays, scale,
                image, I_ang, failure_code, failed_rays );
        #else
            RAY_ERROR("Threaded version requires C++11");
        #endif
    } else {
        RAY_ERROR("Unknown method: "+compute_method);
    }
    PROFILE_STOP(timer_name);

    // Save failed rays
    if ( failure_code != 0 ) {
        write_failures( failure_code, failed_rays, method, N, dz, info->gain );
        RAY_ERROR("Some rays failed");
    }

    // Finished
    PROFILE_STOP("create_image");
}


/**********************************************************************
* Calculate the ray paths                                             *
**********************************************************************/
int RayTrace::calc_ray_path( int Nx, int Ny, int Na, int Nb, const double *x, const double *y, 
    const double *a, const double *b, const int N, const double dz, 
    const ray_gain_struct *gain, const ray_seed_struct *seed, 
    int K, const double *dv, int method, double c,
    std::vector<float>& xr, std::vector<float>& yr, std::vector<float>& Ir )
{
    int N2 = N_SUB*(N-1)+1;
    xr.resize(N2*Nx*Ny*Na*Nb);
    yr.resize(N2*Nx*Ny*Na*Nb);
    Ir.resize(N2*Nx*Ny*Na*Nb);
    int N_errors = 0;
    float *tmp = new float[3*N2];
    for (int i=0; i<Nx; i++) {
        for (int j=0; j<Ny; j++) {
            for (int k=0; k<Na; k++) {
                for (int m=0; m<Nb; m++) {
                    ray_struct ray, ray2;
                    ray.x = (float) x[i];
                    ray.y = (float) y[j];
                    ray.a = (float) a[k];
                    ray.b = (float) b[m];
                    double Iv[K_MAX];
                    int error = RayTrace_calc_ray( ray, N, (float) dz, gain, seed, K, method, Iv, ray2, (float) c, dv, tmp );
                    int index0 = N2*(i+j*Nx+k*Nx*Ny+m*Nx*Ny*Na);
                    for (int n=0; n<N2; n++) {
                        xr[n+index0] = tmp[3*n+0];
                        yr[n+index0] = tmp[3*n+1];
                        Ir[n+index0] = tmp[3*n+2];
                    }
                    if ( error ) 
                        N_errors++;
                }
            }
        }
    }
    delete [] tmp;
    return N_errors;
}


