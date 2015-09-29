// This is a restricted namespace that contains some of the ray trace functions
#ifndef included_RayTrace
#define included_RayTrace

#include "RayTraceStructures.h"

#include <stdio.h>
#include <stdlib.h>
#include <ostream>


/*! \namespace RayTrace
    \brief A Ray Trace namespace
    This namespace provides routines for the Ray Trace model
*/
namespace RayTrace {


    /*!
     * @brief  Trace a single ray path
     * @details  This function follows the path of an individual ray, and calculates
     * its resulting gain and intensity.
     * This function returns 0 if sucessful.
     *
     * @param[in]  ray          The initial ray position (x,y,a,b)
     * @param[in]  N            The number of lengths
     * @param[in]  dz           The length grid spacing (cm)
     * @param[in]  gain         The gain structure (N+1)
     * @param[in]  seed         The seed properties (may be NULL)
     * @param[in]  K            The number of frequencies
     * @param[in]  method       The propagation method
     *                          1 - Propagate backward through the plasma
     *                          2 - Propagate forward through the plasma
     * @param[out] Iv           The output intensity (1xK)
     * @param[out] ray2         The output ray properties (x,y,a,b)
     */
    int calc_ray( const double ray[4], const int N, const double dz, 
        const ray_gain_struct *gain, const ray_seed_struct *seed, 
        int K, int method, double *Iv, double *ray2 );


    /*!
     * @brief  Trace a set of rays, returning their paths and intensity
     * @details  This function calls calc_ray for each ray, storing the
     *   the ray path and intensity as it propagates
     * This function returns 0 if sucessful.
     *
     * @param[in]  Nx           The number of grid points for x
     * @param[in]  Ny           The number of grid points for y
     * @param[in]  Na           The number of grid points for a
     * @param[in]  Nb           The number of grid points for b
     * @param[in]  x            The grid points for x
     * @param[in]  y            The grid points for y
     * @param[in]  a            The grid points for a
     * @param[in]  b            The grid points for b
     * @param[in]  N            The number of lengths
     * @param[in]  dz           The length grid spacing (cm)
     * @param[in]  gain         The gain structure (N+1)
     * @param[in]  seed         The seed properties (may be NULL)
     * @param[in]  K            The number of frequencies
     * @param[in]  dv           The frequency grid spacing
     * @param[in]  method       The propagation method
     *                          1 - Propagate backward through the plasma
     *                          2 - Propagate forward through the plasma
     * @param[in]  c            Safety factor for step length (should be < 1)
     * @param[out] xr           The x-coordinates for the ray paths ( N x Nx x Ny x Na x Nb )
     * @param[out] yr           The y-coordinates for the ray paths ( N x Nx x Ny x Na x Nb )
     * @param[out] Ir           The intensity vs length for the ray paths ( N x Nx x Ny x Na x Nb )
     */
    int calc_ray_path( int Nx, int Ny, int Na, int Nb, const double *x, const double *y, 
        const double *a, const double *b, const int N, const double dz, 
        const ray_gain_struct *gain, const ray_seed_struct *seed, 
        int K, const double *dv, int method, double c,
        std::vector<float>& xr, std::vector<float>& yr, std::vector<float>& Ir );


    /*!
     * @brief  Calculate the seed intensity
     * @details  This function calculates the seed intensity for a single point in space.
     * @param seed          The seed properties  
     * @param x             The x coordinate
     * @param y             The y coordinate
     * @param a             The x angular coordinate
     * @param b             The y angular coordinate
     * @param Iv            The ouput seed intensity (as a function of frequency)
     */
    void calc_seed(const ray_seed_struct& seed, double x, double y, double a, double b, double *Iv);


    /*!
     * @brief  Function to create the EUV beam images
     * @details  This function calculates the near and far field beam images.  This function is 
     * thread-safe provided unique arrays for mem, image, I_ang are provided in the info struct.
     * @param info          Data structure containing the information needed to create the EUV beam images
     */
    void create_image (create_image_struct *info, std::string method="auto" );


}


#endif

