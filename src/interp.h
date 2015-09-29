#ifndef included_interp
#define included_interp

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "CompilerFeatures.h"
#ifdef ENABLE_STD_FUNCTION
    #include <functional>
#endif


/*! \namespace interp
 *  \brief  An interpolation namespace
 *  \details  This namespace provides routines for linear an multi-dimensional linear interpolation of data.
 *     It also provides useful routines for sorting vectors and determining the width of a function.
*/
namespace interp {

    /*! 
     * @brief    Subroutine for a fast approximate x^y
     * @details  This function performs a fast approximation to z=x^y.  
     * @param[in] x         The value for x
     * @param[in] y         The value for y
     */
    double fast_pow( double x, double y );


    /*! 
     * @brief    Subroutine for a fast approximate exponential sum
     * @details  This function performs a fast approximation to exp(sum(ai*log(xi)))
     * @param[in] N         The number of points
     * @param[in] ai        The weights to use 
     * @param[in] xi        The values to sum
     */
    double fast_exp_avg( int N, const double *ai, const double *xi );


    /*! 
     * @brief    Subroutine to get the ratio for interpolation
     * @details  This function returns the interpolation ratio.  On a linear scale
     * this is dx = (x-x0)/(x1-x0).  On a log scale this is dx = log(x/x0)/log(x1/x0).
     * @param[in] x0        The left point
     * @param[in] x1        The right point
     * @param[in] x         The interpolation point
     * @param[in] extrap    Extrapolate beyond [0-1] (true) or limit to [0,1] (false)
     * @param[in] use_log   Use a log scale (true) or linear scale (false)
     */
    double get_interp_ratio( double x0, double x1, double x, bool use_log, bool extrap );


    /*! 
     * @brief    Subroutine to perform linear interpolation
     * @details  This function returns f(x) interpolated between x1 and x2
     * @param dx        (x-x1)/(x2-x1)
     * @param f1        f(x1)
     * @param f2        f(x2)
     */
    inline double linear(double dx, double f1, double f2);


    /*! 
     * @brief    Subroutine to perform bi-linear interpolation (2D)
     * @details  This function returns f(x,y) interpolated between x1-x2 and y1-y2
     * @param dx        (x-x1)/(x2-x1)
     * @param dy        (y-y1)/(y2-y1)
     * @param f1        f(x1,y1)
     * @param f2        f(x2,y1)
     * @param f3        f(x1,y2)
     * @param f4        f(x2,y2)
     */
    inline double bilinear(double dx, double dy, double f1, double f2, double f3, double f4);


    /*!
     * @brief    Subroutine to perform tri-linear interpolation (3D)
     * @details  This function returns f(x,y,z) interpolated between x1-x2, y1-y2, and z1-z2
     * @param dx        (x-x1)/(x2-x1)
     * @param dy        (y-y1)/(y2-y1)
     * @param dz        (z-z1)/(z2-z1)
     * @param f1        f(x1,y1,z1)
     * @param f2        f(x2,y1,z1)
     * @param f3        f(x1,y2,z1)
     * @param f4        f(x2,y2,z1)
     * @param f5        f(x1,y1,z2)
     * @param f6        f(x2,y1,z2)
     * @param f7        f(x1,y2,z2)
     * @param f8        f(x2,y2,z2)
     */
    inline double trilinear(double dx, double dy, double dz, double f1, double f2, 
        double f3, double f4, double f5, double f6, double f7, double f8);


    /*!
     * @brief    Subroutine to perform N-D-linear interpolation
     * @details  This function returns f(x,y,z,..) interpolated between x1-x2, y1-y2, and z1-z2
     * The data should be stored in the following order:
     *   fn[0] = f(x1,y1,z1,...),
     *   fn[1] = f(x2,y1,z1,...),
     *   fn[2] = f(x1,y2,z1,...),
     *   fn[3] = f(x2,y2,z1,...),
     *   fn[4] = f(x1,y1,z2,...), ...
     * Note: f contains 2^N points, dx contains N values, where N is the number of dimensions.
     * Note: to keep this function general, it modifies the input values fn. 
     *   This could be easily modified by coping the info to a local or temporary variable.
     * @param N         The number of dimensions
     * @param dx        (x-x1)/(x2-x1)
     * @param fn        The function values evaluated at each grid point
     */
    inline double n_linear(size_t N, double *dx, double *fn);


    /*! 
     * @brief    Subroutine to perform linear interpolation
     * @details  This function returns f(x) interpolated between x1 and x2
     * @param N         Number of points in xi
     * @param xi        Sorted grid to use for interpolation
     * @param yi        Function values
     * @param x         Point to be interpolated
     */
    double interp_linear(size_t N, const double *xi, const double *yi, double x);


    /*! 
     * @brief    Subroutine to perform cubic hermite
     * @details  This function returns f(x) interpolated between x1 and x2.
     * It does so using a monotonic preserving piecewise cubic hermite polynomial.
     * Ouside of the domain it will perform linear interpolation.
     * @param N         Number of points in xi
     * @param xi        Sorted grid to use for interpolation
     * @param yi        Function values
     * @param x         Point to be interpolated
     */
    double interp_pchip(size_t N, const double *xi, const double *yi, double x);


    /*! 
     * @brief    Subroutine to perform bi-linear interpolation (2D)
     * @details  This function returns f(x,y) interpolated from a rectangular grid
     * @param x         x-value of the interpolate point
     * @param y         y-value of the interpolate point
     * @param N         Number of points in the x-grid
     * @param M         Number of points in the y-grid
     * @param xgrid     x-values of the grid
     * @param ygrid     y-values of the grid
     * @param fgrid     function values evaluated at the grid points
     */
    double bilinear(double x, double y, size_t N, size_t M, const double *xgrid, const double *ygrid, const double *fgrid);


    /*! 
     * @brief    Subroutine to perform tri-linear interpolation (2D)
     * @details  This function returns f(x,y,z) interpolated from a rectangular grid
     * @param x         x-value of the interpolate point
     * @param y         y-value of the interpolate point
     * @param z         z-value of the interpolate point
     * @param N         Number of points in the x-grid
     * @param M         Number of points in the y-grid
     * @param K         Number of points in the z-grid
     * @param xgrid     x-values of the grid
     * @param ygrid     y-values of the grid
     * @param zgrid     z-values of the grid
     * @param fgrid     function values evaluated at the grid points
     */
    double trilinear(double x, double y, double z, size_t N, size_t M, size_t K, 
        const double *xgrid, const double *ygrid, const double *zgrid, const double *fgrid);


    /*!
     * @brief    Subroutine to check if a vector is in ascending order
     * @details  This function checks if the values in an array are in ascending order
     * @param N         Number of values in the array
     * @param X         Data values
     */
    inline bool check_ascending(size_t N, const double *X);


    /*!
     * @brief    Subroutine to find the first element in X which is greater than Y
     * @details  This function finds the first element in X which is greater than Y
     *    using 2 for loops.  This is the slowest and simplist method, but
     *    doesn't require the vector X to be in ascending order.
     *    Returns 0 if no value is larger.
     * @param X         X
     * @param Y         Y
     * @param INDEX     The index of the value of X that is greater than Y
     * @param size_X    The number of values in X
     * @param size_Y    The number of values in Y
     */
    template <class type>
    inline void findfirstloop(const type *X, const type *Y, size_t *INDEX, size_t size_X, size_t size_Y);


    /*!
     * @brief    Subroutine to find the first element in X which is greater than Y
     * @details  This function finds the first element in X which is greater than Y
     *    using a simple hashing technique.  This is the a faster method, but
     *    requires the vector X to be in ascending order.
     *    Returns the index to the largest (last) element if no value is larger.
     * @param X         X
     * @param Y         Y
     * @param INDEX     The index of the value of X that is greater than Y
     * @param size_X    The number of values in X
     * @param size_Y    The number of values in Y
     */
    template <class type>
    inline void findfirsthash(const type *X, const type *Y, size_t *INDEX, size_t size_X, size_t size_Y);


    /*!
     * @brief    Subroutine to find the first element in X which is greater than Y
     * @details  This function finds the first element in X which is greater than Y
     *    using a simple hashing technique.  This is the a faster method, but
     *    requires the vector X to be in ascending order.
     *    This works with an initial guess, and only one value.
     * @param X         X
     * @param size_X    The number of values in X
     * @param Y         Y
     */
    template <class type>
    inline size_t findfirstsingle( const type *X, size_t size_X, type Y );


    /*!
     * @brief    Subroutine to sort the elements in X
     * @details  This function sorts the values in X using quicksort
     * @param n         The number of values in X
     * @param X         Input/Output: Points to sort
     */
    template <class type>
    void quicksort(size_t n, type *X);


    /*!
     * @brief    Subroutine to sort the elements in X
     * @details  This function sorts the values in X
     * @param n         The number of values in X
     * @param X         Input/Output: Points to sort
     * @param Y         Input/Output: Extra values to be sorted with X
     */
    template <class type1, class type2>
    void quicksort(size_t n, type1 *X, type2 *Y);


    /*!
     * @brief    Subroutine to sort the elements in X
     * @details  This function sorts the values in X, storing them in Y
     * @param n         The number of values in X
     * @param X         Points to sort
     * @param Y         Output: Points to sort
     */
    inline void sort(size_t n, const double *X, double *Y);


    /*!
     * @brief    Subroutine to sort the elements in X
     * @details  This function sorts the values in X, storing them in Y.  
     *    The vector I is also returned such that Y[j] = X[I[j]]
     * @param n         The number of values in X
     * @param X         Points to sort
     * @param Y         Output: Points to sort
     * @param I         Output: Indicies of the sort (indexing starting at 0)
     */
    inline void sort(size_t n, const double *X, double *Y, size_t *I);


    /*!
     * @brief    Subroutine to perform the unique operation on the elements in X
     * @details  This function performs the unique operation on the values in X
     * @param n         Input/Output: The number of values in X
     * @param X         Input/Output: Points to sort
     */
    inline void unique(size_t *n, double *X);


    /*!
     * @brief    Subroutine to perform the unique operation on the elements in X
     * @details  This function performs the unique operation on the values in X storing them in Y
     * @param nx        The number of values in X
     * @param X         Points to sort
     * @param ny        Output: The number of values in X
     * @param Y         Output: Sorted points
     */
    inline void unique(size_t nx, const double *X, size_t *ny, double *Y);


    /*!
     * @brief    Subroutine to perform the unique operation on the elements in X
     * @details  This function performs the unique operation on the values in X storing them in Y.
     *    It also returns the index vectors I and J such that Y[k] = X[I[k]] and X[k] = Y[J[k]].  
     * @param nx        The number of values in X
     * @param X         Points to sort (nx)
     * @param ny        Output: The number of values in X
     * @param Y         Output: Sorted points (ny)
     * @param I         Output: The index vector I (ny)
     * @param J         Output: The index vector J (nx)
     */
    inline void unique(size_t nx, const double *X, size_t *ny, double *Y, size_t *I, size_t *J);


    /*!
     * @brief    Function to calculate the "FWHM" of a function
     * @details  This function calculates the "FWHM" based on the narrowest region 
     *    that contains 76% of the energy.  This matches the definition of the FWHM 
     *    for a gaussian profile.  If errors are encountered -1 is returned.  
     * @param n         The number of points in the function
     * @param x         The coordinates of the points
     * @param y         The function values (must be > 0 for each value)
     */
    double calc_width(size_t n, const double *x, const double *y);


#ifdef ENABLE_STD_FUNCTION
    /*!
     * @brief    Function to solve for the zero of f(x)
     * @details  This function solves for the solution of f(x)=0 using a modified bisection method.  
     * @param fun       The function to solve of the form f(x,options), where 
     *                  options is a pointer to any additional data needed by the function
     * @param lb        The lower bound to search
     * @param ub        The upper bound to search
     * @param tol1      The tolerance of the function evaluation to stop: abs(f(x)) < tol1
     * @param tol2      The tolerance of x to stop: abs(x1-x2) < tol2
     * @param options   Any additional options that are needed by the function
     */
    template<class ... Args>
    double bisection( std::function<double(double,Args...)>, double lb, double ub, double tol1, double tol2, Args... options );
#endif


    /*!
     * @brief    Function to compute the next guess for a modified bisection method
     * @details  This function solves computes the next guess for a bisection method 
     *    given the previous guesses and function (residual) evaluations.  
     *    This method uses a piecewise cubic interpolation to compute the guess,
     *    then modifies it to preserve a minimum search region for the bisection method.
     *    It does not require the revious guesses to be in sorted order. 
     *    It does require that there are two values with differnt size.
     * @param[in] N         The number of previous guesses
     * @param[in] x         The previous guesses
     * @param[in] r         The function or residual evaluations at x
     * @param[out] range    Optional output with the current bounds that contain the solution [lb ub]
     * @return              The next guess to use
     */
    double bisection_coeff( size_t N, const double* x, const double* r, double* range=NULL );

}

#include "interp.hpp"

#endif

