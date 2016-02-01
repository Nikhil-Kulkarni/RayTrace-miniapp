#include "RayTraceStructures.h"
#include "common/RayTraceImageHelper.h"
#include "utilities/RayUtilities.h"
#include <cuda.h>


// Check for errors
#define CUDA_CHECK()                                    \
    do {                                                \
        if ( cudaPeekAtLastError() != cudaSuccess ) {   \
            cudaError_t error = cudaGetLastError();     \
            printf("cuda error: %i\n",error);           \
            printf("   %s\n",cudaGetErrorString(error)); \
            printf("   line: %i\n",(int)__LINE__);      \
            printf("   file: %s\n",__FILE__);           \
            exit(-1);                                   \
        }                                               \
    } while (0)


#define CUDA_PRINT_FUNCTION( fun )                      \
    do {                                                \
        cudaFuncAttributes attr;                        \
        cudaFuncGetAttributes(&attr,fun);               \
        printf("%s:\n",#fun);                           \
        printf("  version = %i\n",attr.binaryVersion);  \
        printf("  ptx = %i\n",attr.ptxVersion);         \
        printf("  constSize = %i\n",attr.constSizeBytes); \
        printf("  localSize = %i\n",attr.localSizeBytes); \
        printf("  sharedSize = %i\n",attr.sharedSizeBytes); \
        printf("  maxThreads = %i\n",attr.maxThreadsPerBlock); \
        printf("  numRegs = %i\n",attr.numRegs);        \
    } while (0)


// Atomic add operation for double
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


// Get the globally unique thread id
__device__ int getGlobalIdx3D()
{
	int blockId = blockIdx.x 
			 + blockIdx.y * gridDim.x 
			 + gridDim.x * gridDim.y * blockIdx.z; 
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
			  + (threadIdx.z * (blockDim.x * blockDim.y))
			  + (threadIdx.y * blockDim.x)
			  + threadIdx.x;
	return threadId;
}


// Kernel that executes on the CUDA device
__global__
__launch_bounds__(128,8)    // Set bounds to limit the number of registers
void RayTraceImageCudaKernel( int N, int nx, int ny, int na, int nb, int nv, 
    const double *x, const double *y, const double *a, const double *b,
    double dx, double dy, double dz, double da, double db, const double *dv,
    const RayTrace::ray_gain_struct* gain, const RayTrace::ray_seed_struct* seed,
    int method, int N_rays, const ray_struct *rays, double scale,
    double *image, double *I_ang ) 
{
    int idx = getGlobalIdx3D();
    if ( idx<N_rays ) {
        const ray_struct ray = rays[idx];
        double Iv[K_MAX];
        ray_struct ray2;
        int error = RayTrace_calc_ray( ray, N, dz, gain, seed, nv, method, Iv, ray2 );
        if ( error!=0 ) {
            //failed_rays.push_back(ray);
            //set_bit(-error,failure_code);
        } else {
            if ( method == 1 ) {
                // We are propagating backward, use ray for the cell updates
                ray2 = ray;
            } else {
                // We are propagating forward, use ray2 for the cell updates
                // Note: The sign of the angle is reversed with respect to the euv_beam
                ray2.a = -ray2.a;
                ray2.b = -ray2.b;
                if ( ray2.y<0.0 && y[0]>=0.0 ) {
                    // We need to change the sign of y
                    ray2.y = -ray2.y;
                }
            }
            // Get the indicies to the cells in image and I_ang
            // Note: do not replace these lines with findindex, we need to be able to return 0 for the index
            int i1 = static_cast<int>( findfirstsingle( x, nx, ray2.x-0.5*dx ) );
            int i2 = static_cast<int>( findfirstsingle( y, ny, ray2.y-0.5*dy ) );
            int i3 = static_cast<int>( findfirstsingle( a, na, ray2.a-0.5*da ) );
            int i4 = static_cast<int>( findfirstsingle( b, nb, ray2.b-0.5*db ) );
            if ( ray2.x<x[0]-0.5*dx || ray2.x>x[nx-1]+0.5*dx )
                i1 = -1;        // The ray's z position is out of the range of image 
            if ( ray2.y<y[0]-0.5*dy || ray2.y>y[ny-1]+0.5*dy )
                i2 = -1;        // The ray's y position is out of the range of image 
            if ( -ray2.a<a[0]-0.5*da || -ray2.a>a[na-1]+0.5*da )
                i3 = -1;        // The ray's z angle is out of the range of I_ang 
            if ( -ray2.b<b[0]-0.5*db || -ray2.b>b[nb-1]+0.5*db )
                i4 = -1;        // The ray's y angle is out of the range of I_ang
            // Copy I_out into image 
            if (i1>=0 && i2>=0){ 
                double *Iv2 = &image[nv*(i1+i2*nx)];
                for (int iv=0; iv<nv; iv++)
                    atomicAdd(&Iv2[iv],Iv[iv]*scale);
            }
            // Copy I_out into I_ang 
            if (i3>=0 && i4>=0) {    
                double tmp = 0.0;
                for (int iv=0; iv<nv; iv++)
                    tmp += 2.0*dv[iv]*Iv[iv];
                atomicAdd(&I_ang[i3+i4*na],tmp);
            }
        }
    }
}


// Compute the block size to use
inline dim3 calcBlockSize( size_t N_blocks )
{
    dim3 block_size;
    if ( N_blocks < 65535 ) {
        block_size.x = N_blocks;
    } else {
        block_size.y = N_blocks/32768;
        block_size.x = N_blocks/block_size.y + (N_blocks%block_size.y == 0 ? 0:1);
    }
    return block_size;
}


// Create the image and call the cuda kernel
void RayTraceImageCudaLoop( int N, int nx, int ny, int na, int nb, int nv, 
    const double *x, const double *y, const double *a, const double *b,
    double dx, double dy, double dz, double da, double db, const double *dv,
    const RayTrace::ray_gain_struct* gain_in, const RayTrace::ray_seed_struct* seed_in,
    int method, const std::vector<ray_struct>& rays, double scale,
    double *image, double *I_ang, 
    unsigned int& failure_code, std::vector<ray_struct>& failed_rays ) 
{
    failure_code = 0;   // Need to track failures on GPU
    // Get device properties
    static int maxThreadsPerBlock = 0;
    if ( maxThreadsPerBlock == 0 ) {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr,RayTraceImageCudaKernel);
        maxThreadsPerBlock = attr.maxThreadsPerBlock;
        RAY_ASSERT(maxThreadsPerBlock>0);
        //CUDA_PRINT_FUNCTION(RayTraceImageCudaKernel);
    }    
    // place the ray gain and seed structures on the device
    const RayTrace::ray_gain_struct* gain = RayTrace::ray_gain_struct::copy_device( N, gain_in );
    const RayTrace::ray_seed_struct* seed = NULL;
    if ( seed_in!=NULL )
        seed = seed_in->copy_device();
    // Allocate device memory
    size_t N_rays = rays.size();
    double *x2, *y2, *a2, *b2, *dv2, *image2, *I_ang2;
    ray_struct *rays2;
    cudaMalloc(&x2,nx*sizeof(double));
    cudaMalloc(&y2,ny*sizeof(double));
    cudaMalloc(&a2,na*sizeof(double));
    cudaMalloc(&b2,nb*sizeof(double));
    cudaMalloc(&dv2,nv*sizeof(double));
    cudaMalloc(&image2,nx*ny*nv*sizeof(double));
    cudaMalloc(&I_ang2,na*nb*sizeof(double));
    cudaMalloc(&rays2,N_rays*sizeof(ray_struct));
    cudaMemcpy(x2,x,nx*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(y2,y,ny*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(a2,a,na*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(b2,b,nb*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dv2,dv,nv*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(rays2,&rays[0],N_rays*sizeof(ray_struct),cudaMemcpyHostToDevice);
    cudaMemset(image2,0,nx*ny*nv*sizeof(double));
    cudaMemset(I_ang2,0,na*nb*sizeof(double));
    CUDA_CHECK();
    // Do calculation on device:
    size_t threads = maxThreadsPerBlock;
    size_t N_blocks = N_rays/threads + (N_rays%threads == 0 ? 0:1);
    dim3 block_size = calcBlockSize(N_blocks);
    block_size.x = N_rays/threads + (N_rays%threads == 0 ? 0:1);
    RayTraceImageCudaKernel <<< block_size,threads >>> (N,nx,ny,na,nb,nv,x2,y2,a2,b2,
        dx,dy,dz,da,db,dv2,gain,seed,method,N_rays,rays2,scale,image2,I_ang2);
    CUDA_CHECK();
    // Retrieve result from device and store it in host array
    cudaMemcpy(image,image2,nx*ny*nv*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(I_ang,I_ang2,na*nb*sizeof(double),cudaMemcpyDeviceToHost);
    CUDA_CHECK();
    // Cleanup
    cudaFree(x2);
    cudaFree(y2);
    cudaFree(a2);
    cudaFree(b2);
    cudaFree(dv2);
    cudaFree(rays2);
    cudaFree(image2);
    cudaFree(I_ang2);
    CUDA_CHECK();
    RayTrace::ray_gain_struct::free_device( N, gain_in, gain );
    RayTrace::ray_seed_struct::free_device( seed_in, seed );
}


// Copy ray_gain_struct to GPU
const RayTrace::ray_gain_struct* ray_gain_struct_copy_device_cuda( size_t N, const RayTrace::ray_gain_struct* arr )
{
    RayTrace::ray_gain_struct* host_ptr = new RayTrace::ray_gain_struct[N];
    for (size_t i=0; i<N; i++) {
        host_ptr[i].Nx = arr[i].Nx;
        host_ptr[i].Ny = arr[i].Ny;
        host_ptr[i].Nv = arr[i].Nv;
        cudaMalloc(&host_ptr[i].x,arr[i].Nx*sizeof(double));
        cudaMalloc(&host_ptr[i].y,arr[i].Ny*sizeof(double));
        cudaMalloc(&host_ptr[i].n,arr[i].Nx*arr[i].Ny*sizeof(double));
        cudaMalloc(&host_ptr[i].g0,arr[i].Nx*arr[i].Ny*sizeof(float));
        cudaMalloc(&host_ptr[i].E0,arr[i].Nx*arr[i].Ny*sizeof(float));
        cudaMalloc(&host_ptr[i].gv,arr[i].Nx*arr[i].Ny*arr[i].Nv*sizeof(float));
        cudaMalloc(&host_ptr[i].gv0,arr[i].Nx*arr[i].Ny*sizeof(float));
        cudaMemcpy(host_ptr[i].x,arr[i].x,arr[i].Nx*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(host_ptr[i].y,arr[i].y,arr[i].Ny*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(host_ptr[i].n,arr[i].n,arr[i].Nx*arr[i].Ny*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(host_ptr[i].g0,arr[i].g0,arr[i].Nx*arr[i].Ny*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(host_ptr[i].E0,arr[i].E0,arr[i].Nx*arr[i].Ny*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(host_ptr[i].gv,arr[i].gv,arr[i].Nx*arr[i].Ny*arr[i].Nv*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(host_ptr[i].gv0,arr[i].gv0,arr[i].Nx*arr[i].Ny*sizeof(float),cudaMemcpyHostToDevice);
    }
    RayTrace::ray_gain_struct* dev_ptr=NULL;
    cudaMalloc(&dev_ptr,N*sizeof(RayTrace::ray_gain_struct));
    cudaMemcpy(dev_ptr,host_ptr,N*sizeof(RayTrace::ray_gain_struct),cudaMemcpyHostToDevice);
    for (size_t i=0; i<N; i++) {
        host_ptr[i].x = NULL;
        host_ptr[i].y = NULL;
        host_ptr[i].n = NULL;
        host_ptr[i].g0 = NULL;
        host_ptr[i].E0 = NULL;
        host_ptr[i].gv = NULL;
        host_ptr[i].gv0 = NULL;
    }
    delete [] host_ptr;
    CUDA_CHECK();
    return dev_ptr;
}
// Free ray_gain_struct from GPU
void ray_gain_struct_free_device_cuda( size_t N, const RayTrace::ray_gain_struct* dev_ptr )
{
    RayTrace::ray_gain_struct* host_ptr = new RayTrace::ray_gain_struct[N];
    cudaMemcpy(host_ptr,dev_ptr,N*sizeof(RayTrace::ray_gain_struct),cudaMemcpyDeviceToHost);
    for (size_t i=0; i<N; i++) {
        cudaFree(host_ptr[i].x);
        cudaFree(host_ptr[i].y);
        cudaFree(host_ptr[i].n);
        cudaFree(host_ptr[i].g0);
        cudaFree(host_ptr[i].E0);
        cudaFree(host_ptr[i].gv);
        cudaFree(host_ptr[i].gv0);
        host_ptr[i].x = NULL;
        host_ptr[i].y = NULL;
        host_ptr[i].n = NULL;
        host_ptr[i].g0 = NULL;
        host_ptr[i].E0 = NULL;
        host_ptr[i].gv = NULL;
        host_ptr[i].gv0 = NULL;
    }
    cudaFree((void*)dev_ptr);
    delete [] host_ptr;
    CUDA_CHECK();
}


// Copy ray_seed_struct to GPU
const RayTrace::ray_seed_struct* ray_seed_struct_copy_device_cuda( const RayTrace::ray_seed_struct& seed )
{
    RayTrace::ray_seed_struct* host_ptr = new RayTrace::ray_seed_struct();
    host_ptr->f0 = seed.f0;
    for (size_t i=0; i<5; i++) {
        host_ptr->dim[i] = seed.dim[i];
        cudaMalloc(&host_ptr->x[i],seed.dim[i]*sizeof(double));
        cudaMalloc(&host_ptr->f[i],seed.dim[i]*sizeof(double));
        cudaMemcpy(host_ptr->x[i],seed.x[i],seed.dim[i]*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(host_ptr->f[i],seed.f[i],seed.dim[i]*sizeof(double),cudaMemcpyHostToDevice);
    }
    RayTrace::ray_seed_struct* dev_ptr=NULL;
    cudaMalloc(&dev_ptr,sizeof(RayTrace::ray_seed_struct));
    cudaMemcpy(dev_ptr,host_ptr,sizeof(RayTrace::ray_seed_struct),cudaMemcpyHostToDevice);
    for (size_t i=0; i<5; i++) {
        host_ptr->x[i] = NULL;
        host_ptr->f[i] = NULL;
    }
    delete host_ptr;
    CUDA_CHECK();
    return dev_ptr;
}
// Free ray_seed_struct from GPU
void ray_seed_struct_free_device_cuda( const RayTrace::ray_seed_struct* dev_ptr )
{
    RayTrace::ray_seed_struct* host_ptr = new RayTrace::ray_seed_struct;
    cudaMemcpy(host_ptr,dev_ptr,sizeof(RayTrace::ray_seed_struct),cudaMemcpyDeviceToHost);
    for (size_t i=0; i<5; i++) {
        cudaFree(host_ptr->x[i]);
        cudaFree(host_ptr->f[i]);
        host_ptr->x[i] = NULL;
        host_ptr->f[i] = NULL;
    }
    cudaFree((void*)dev_ptr);
    delete host_ptr;
    CUDA_CHECK();
}


