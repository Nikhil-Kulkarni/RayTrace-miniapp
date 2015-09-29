#ifndef included_interp_hpp
#define included_interp_hpp

#include <stdlib.h>
#include <stdio.h>
#include <stdexcept>
#include <iostream>
#include <limits> 
#include "interp.h"


// Subroutine to perform linear interpolation
inline double interp::linear(double dx, double f1, double f2) 
{
    double f, dx2;
    dx2 = 1.0-dx;
    f = dx2*f1 + dx*f2;
    return(f);
}


// Subroutine to perform bi-linear interpolation
inline double interp::bilinear(double dx, double dy, double f1, double f2, double f3, double f4) 
{
    double f, dx2, dy2;
    dx2 = 1.0-dx;
    dy2 = 1.0-dy;
    f   = (dx*f2 + dx2*f1)*dy2 + (dx*f4 + dx2*f3)*dy;
    return(f);
}


// Subroutine to perform tri-linear interpolation 
inline double interp::trilinear(double dx, double dy, double dz, double f1, double f2, 
    double f3, double f4, double f5, double f6, double f7, double f8) 
{
    double f, dx2, dy2, dz2, h0, h1;
    dx2 = 1.0-dx;
    dy2 = 1.0-dy;
    dz2 = 1.0-dz;
    h0  = (dx*f2 + dx2*f1)*dy2 + (dx*f4 + dx2*f3)*dy;
    h1  = (dx*f6 + dx2*f5)*dy2 + (dx*f8 + dx2*f7)*dy;
    f   = h0*dz2 + h1*dz;
    return(f);
}


// Subroutine to perform N-D-linear interpolation
inline double interp::n_linear(size_t N, double *dx, double *fn) 
{
    /* We need to perform a set of linear interpolations */
    for (size_t i=0; i<N; i++) {
        double dy = dx[i];
        double dy2 = 1.0-dy;
        size_t k = 2;
        for (size_t j=0; j<N-i-1; j++)
            k *= 2;
        for (size_t j=0; j<k; j+=2) {
            fn[j/2] = dy2*fn[j] + dy*fn[j+1];
        }
    }
    return(fn[0]);
}


// Subroutine to check if a vector is in ascending order.
inline bool interp::check_ascending(size_t N, const double *X)
{
    for (size_t i=1; i<N; i++) {
        if ( X[i] <= X[i-1] )
            return false;
    }
    return true;
}


// Subroutine to find the first element in X which is greater than Y using 2 for loops
template<class TYPE>
inline void interp::findfirstloop(const TYPE *X, const TYPE *Y, size_t *INDEX, size_t size_X, size_t size_Y)
{
    for (size_t j=0; j<size_Y; j++) {
        INDEX[j] = size_X;
        for (size_t i=0; i<size_X; i++) {
            if (X[i] >= Y[j]) {
                INDEX[j] = i;
                break;
            }
        }
    }
}


// Subroutine to find the first element in X which is greater than Y  using a simple hashing technique
template<class TYPE>
inline void interp::findfirsthash(const TYPE *X, const TYPE *Y, size_t *INDEX, size_t size_X, size_t size_Y)
{
    for (size_t j=0; j<size_Y; j++) {
        if (X[0] >= Y[j]) {
            INDEX[j] = 0;
        } else if (X[size_X-1] < Y[j]) {
            INDEX[j] = size_X-1;
        } else {
            size_t lower = 0;
            size_t upper = size_X-1;
            while ( (upper-lower) != 1 ) {
                size_t value = (upper+lower)/2;
                if ( X[value] >= Y[j] )
                    upper = value;
                else
                    lower = value;
            }
            INDEX[j] = upper;
        }
    }
}



// Subroutine to find the first element in X which is greater than y using a simple hashing technique
template<class TYPE>
inline size_t interp::findfirstsingle( const TYPE *X, size_t size_X, TYPE Y )
{
    if ( size_X == 0 )
        return 0;
    if ( X[0] >= Y )
        return 0;
    if ( X[size_X-1] < Y )
        return size_X;
    size_t lower = 0;
    size_t upper = size_X-1;
    while ( (upper-lower) != 1 ) {
        size_t value = (upper+lower)/2;
        if ( X[value] >= Y )
            upper = value;
        else
            lower = value;
    }
    return upper;
}


// Subroutine to perform a quicksort
template <class T>
void interp::quicksort(size_t a_size, T *arr)
{
    long int n = static_cast<long int>(a_size);
    if ( n <= 1 )
        return;
    bool test;
    long int i, ir, j, jstack, k, l, istack[100];
    T a, tmp_a;
    jstack = 0;
    l = 0;
    ir = n-1;
    while (1) {
        if ( ir-l < 7 ) {             // Insertion sort when subarray small enough.
            for ( j=l+1; j<=ir; j++ ) {
                a = arr[j];
                test = true;
                for (i=j-1; i>=0; i--) {
                    if ( arr[i] < a ) {
                        arr[i+1] = a;
                        test = false;
                        break;
                    }
                    arr[i+1] = arr[i];
                }
                if ( test ) {
                    i = l-1;
                    arr[i+1] = a;
                }
            }
            if ( jstack==0 )
                return;
            ir = istack[jstack];    // Pop stack and begin a new round of partitioning.
            l = istack[jstack-1];
            jstack -= 2;
        } else {
            k = (l+ir)/2;           // Choose median of left, center and right elements as partitioning
                                    // element a. Also rearrange so that a(l) < a(l+1) < a(ir).
            tmp_a = arr[k];
            arr[k] = arr[l+1];
            arr[l+1] = tmp_a;
            if ( arr[l]>arr[ir] ) {
                tmp_a = arr[l];
                arr[l] = arr[ir];
                arr[ir] = tmp_a;
            }
            if ( arr[l+1] > arr[ir] ) {
                tmp_a = arr[l+1];
                arr[l+1] = arr[ir];
                arr[ir] = tmp_a;
            }
            if ( arr[l] > arr[l+1] ) {
                tmp_a = arr[l];
                arr[l] = arr[l+1];
                arr[l+1] = tmp_a;
            }
            // Scan up to find element > a
            j = ir;
            a = arr[l+1];           // Partitioning element.
            for (i=l+2; i<=ir; i++) { 
                if ( arr[i]<a ) 
                    continue;
                while ( arr[j]>a )  // Scan down to find element < a.
                    j--;
                if ( j < i )
                    break;          // Pointers crossed. Exit with partitioning complete.
                tmp_a = arr[i];     // Exchange elements of both arrays.
                arr[i] = arr[j];
                arr[j] = tmp_a;
            }
            arr[l+1] = arr[j];      // Insert partitioning element in both arrays.
            arr[j] = a;
            jstack += 2;
            // Push pointers to larger subarray on stack, process smaller subarray immediately.
            if ( ir-i+1 >= j-l ) {
                istack[jstack] = ir;
                istack[jstack-1] = i;
                ir = j-1;
            } else {
                istack[jstack] = j-1;
                istack[jstack-1] = l;
                l = i;
            }
        }
    }
}


// Subroutine to perform a quicksort
template <class T1, class T2>
void interp::quicksort(size_t size, T1 *arr, T2 *brr)
{
    long int n = static_cast<long int>(size);
    if ( n <= 1 )
        return;
    bool test;
    long int i, ir, j, jstack, k, l, istack[100];
    T1 a, tmp_a;
    T2 b, tmp_b;
    jstack = 0;
    l = 0;
    ir = n-1;
    while (1) {
        if ( ir-l < 7 ) {             // Insertion sort when subarray small enough.
            for ( j=l+1; j<=ir; j++ ) {
                a = arr[j];
                b = brr[j];
                test = true;
                for (i=j-1; i>=0; i--) {
                    if ( arr[i] < a ) {
                        arr[i+1] = a;
                        brr[i+1] = b;
                        test = false;
                        break;
                    }
                    arr[i+1] = arr[i];
                    brr[i+1] = brr[i];
                }
                if ( test ) {
                    i = l-1;
                    arr[i+1] = a;
                    brr[i+1] = b;
                }
            }
            if ( jstack==0 )
                return;
            ir = istack[jstack];    // Pop stack and begin a new round of partitioning.
            l = istack[jstack-1];
            jstack -= 2;
        } else {
            k = (l+ir)/2;           // Choose median of left, center and right elements as partitioning
                                    // element a. Also rearrange so that a(l) ? a(l+1) ? a(ir).
            tmp_a = arr[k];
            arr[k] = arr[l+1];
            arr[l+1] = tmp_a;
            tmp_b = brr[k];
            brr[k] = brr[l+1];
            brr[l+1] = tmp_b;
            if ( arr[l]>arr[ir] ) {
                tmp_a = arr[l];
                arr[l] = arr[ir];
                arr[ir] = tmp_a;
                tmp_b = brr[l];
                brr[l] = brr[ir];
                brr[ir] = tmp_b;
            }
            if ( arr[l+1] > arr[ir] ) {
                tmp_a = arr[l+1];
                arr[l+1] = arr[ir];
                arr[ir] = tmp_a;
                tmp_b = brr[l+1];
                brr[l+1] = brr[ir];
                brr[ir] = tmp_b;
            }
            if ( arr[l] > arr[l+1] ) {
                tmp_a = arr[l];
                arr[l] = arr[l+1];
                arr[l+1] = tmp_a;
                tmp_b = brr[l];
                brr[l] = brr[l+1];
                brr[l+1] = tmp_b;
            }
            // Scan up to find element > a
            j = ir;
            a = arr[l+1];           // Partitioning element.
            b = brr[l+1];
            for (i=l+2; i<=ir; i++) { 
                if ( arr[i]<a ) 
                    continue;
                while ( arr[j]>a )  // Scan down to find element < a.
                    j--;
                if ( j < i )
                    break;          // Pointers crossed. Exit with partitioning complete.
                tmp_a = arr[i];     // Exchange elements of both arrays.
                arr[i] = arr[j];
                arr[j] = tmp_a;
                tmp_b = brr[i];
                brr[i] = brr[j];
                brr[j] = tmp_b;
            }
            arr[l+1] = arr[j];      // Insert partitioning element in both arrays.
            arr[j] = a;
            brr[l+1] = brr[j];
            brr[j] = b;
            jstack += 2;
            // Push pointers to larger subarray on stack, process smaller subarray immediately.
            if ( ir-i+1 >= j-l ) {
                istack[jstack] = ir;
                istack[jstack-1] = i;
                ir = j-1;
            } else {
                istack[jstack] = j-1;
                istack[jstack-1] = l;
                l = i;
            }
        }
    }
}


inline void interp::sort(size_t n, const double *X, double *Y) {
    // Copy the values of X
    for (size_t i=0; i<n; i++)
        Y[i] = X[i];
    // Sort the values
    quicksort(n,Y);
}



inline void interp::sort(size_t n, const double *X, double *Y, size_t *I) {
    // Copy the values of X and create index array
    for (size_t i=0; i<n; i++) {
        Y[i] = X[i];
        I[i] = i;
    }
    // Sort the values
    quicksort(n,Y,I);
}


// Subroutine to find the unique elements in a list 
inline void interp::unique(size_t *n, double *x)
{
    if ( (*n)<=1 )
        return;
    // First perform a quicksort
    quicksort(*n,x);
    // Next remove duplicate entries
    size_t pos = 1;
    for (size_t i=1; i<*n; i++) {
        if ( x[i] != x[pos-1] ) {
            x[pos] = x[i];
            pos++;
        }
    }
    *n = pos;
}


// Subroutine to find the unique elements in a list 
inline void interp::unique( size_t n, const double *X, size_t *ny, double *Y)
{
    *ny = n;
    for (size_t i=0; i<n; i++)
        Y[i] = X[i];
    interp::unique(ny,Y);
}


inline void interp::unique(size_t nX, const double *X, size_t *nY, double *Y, size_t *I, size_t *J) 
{
    // Copy the values of X and initialize the index vector I
    for (size_t i=0; i<nX; i++) {
        Y[i] = X[i];
        I[i] = i;
    }
    // Sort the values
    quicksort(nX,Y,I);
    // Delete duplicate entries
    size_t i = 0;
    J[I[0]] = 0;
    for (size_t j=1; j<nX; j++) {
        if ( Y[i] == Y[j] ) {
            J[I[j]] = J[I[i]];
        } else {
            Y[i+1] = Y[j];
            I[i+1] = I[j];
            J[I[i+1]] = i+1;
            i++;
        }
    }
    *nY= i+1;
}


// Modified bisection root finding
#ifdef ENABLE_STD_FUNCTION
template<class ... Args>
double interp::bisection( std::function<double(double,Args...)> fun, 
    double lb, double ub, double tol1, double tol2, Args... options )
{
    if ( ub <= lb )
        throw std::logic_error("Error: lb <= ub");
    const int N_max = 500;
    double range[2] = {lb, ub};
    double x[N_max], f[N_max];
    x[0] = lb;
    x[1] = ub;
    f[0] = fun(x[0],options...);
    f[1] = fun(x[1],options...);
    if ( fabs(f[0])<tol1 || fabs(f[1])<tol1 )
        return fabs(f[0])<tol1 ? x[0]:x[1];
    if ( ( f[0]<0 && f[1]<0 ) || ( f[0]>0 && f[1]>0 ) )
        throw std::logic_error("Error: sign(f(lb)) == sign(f(ub))");
    size_t i = 2;
    while ( (range[1]-range[0]) > tol2 ) {
        // Get the next guess
        x[i] = bisection_coeff(i,x,f,range);
        // Evaluate f
        f[i] = fun(x[i],options...);
        i++;
        if ( fabs(f[i-1]) < tol1 )
            break;
        if ( i > N_max )
            throw std::logic_error("Excessive number of iterations");
    }
    return x[i-1];
}
#endif


#endif

