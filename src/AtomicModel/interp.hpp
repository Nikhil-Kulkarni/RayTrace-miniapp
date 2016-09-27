#ifndef included_interp_hpp
#define included_interp_hpp

#include "AtomicModel/interp.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>


// Subroutine to perform linear interpolation
inline double interp::linear( double dx, double f1, double f2 )
{
    double f, dx2;
    dx2 = 1.0 - dx;
    f   = dx2 * f1 + dx * f2;
    return ( f );
}


// Subroutine to perform bi-linear interpolation
inline double interp::bilinear( double dx, double dy, double f1, double f2, double f3, double f4 )
{
    double f, dx2, dy2;
    dx2 = 1.0 - dx;
    dy2 = 1.0 - dy;
    f   = ( dx * f2 + dx2 * f1 ) * dy2 + ( dx * f4 + dx2 * f3 ) * dy;
    return ( f );
}


// Subroutine to perform tri-linear interpolation
inline double interp::trilinear( double dx, double dy, double dz, double f1, double f2, double f3,
    double f4, double f5, double f6, double f7, double f8 )
{
    double f, dx2, dy2, dz2, h0, h1;
    dx2 = 1.0 - dx;
    dy2 = 1.0 - dy;
    dz2 = 1.0 - dz;
    h0  = ( dx * f2 + dx2 * f1 ) * dy2 + ( dx * f4 + dx2 * f3 ) * dy;
    h1  = ( dx * f6 + dx2 * f5 ) * dy2 + ( dx * f8 + dx2 * f7 ) * dy;
    f   = h0 * dz2 + h1 * dz;
    return ( f );
}


// Subroutine to perform N-D-linear interpolation
inline double interp::n_linear( size_t N, double *dx, double *fn )
{
    /* We need to perform a set of linear interpolations */
    for ( size_t i = 0; i < N; i++ ) {
        double dy  = dx[i];
        double dy2 = 1.0 - dy;
        size_t k   = 2;
        for ( size_t j = 0; j < N - i - 1; j++ )
            k *= 2;
        for ( size_t j = 0; j < k; j += 2 ) {
            fn[j / 2] = dy2 * fn[j] + dy * fn[j + 1];
        }
    }
    return ( fn[0] );
}


// Subroutine to check if a vector is in ascending order.
inline bool interp::check_ascending( size_t N, const double *X )
{
    for ( size_t i = 1; i < N; i++ ) {
        if ( X[i] <= X[i - 1] )
            return false;
    }
    return true;
}


// Subroutine to find the first element in X which is greater than Y using 2 for loops
template <class TYPE>
HOST_DEVICE inline void interp::findfirstloop(
    const TYPE *X, const TYPE *Y, size_t *INDEX, size_t size_X, size_t size_Y )
{
    for ( size_t j = 0; j < size_Y; j++ ) {
        INDEX[j] = size_X;
        for ( size_t i = 0; i < size_X; i++ ) {
            if ( X[i] >= Y[j] ) {
                INDEX[j] = i;
                break;
            }
        }
    }
}


// Subroutine to find the first element in X which is greater than Y  using a simple hashing
// technique
template <class TYPE>
HOST_DEVICE inline void interp::findfirsthash(
    const TYPE *X, const TYPE *Y, size_t *INDEX, size_t size_X, size_t size_Y )
{
    for ( size_t j = 0; j < size_Y; j++ ) {
        if ( X[0] >= Y[j] ) {
            INDEX[j] = 0;
        } else if ( X[size_X - 1] < Y[j] ) {
            INDEX[j] = size_X - 1;
        } else {
            size_t lower = 0;
            size_t upper = size_X - 1;
            while ( ( upper - lower ) != 1 ) {
                size_t value = ( upper + lower ) / 2;
                if ( X[value] >= Y[j] )
                    upper = value;
                else
                    lower = value;
            }
            INDEX[j] = upper;
        }
    }
}


// Subroutine to find the first element in X which is greater than y using a simple hashing
// technique
template <class TYPE>
HOST_DEVICE inline size_t interp::findfirstsingle( const TYPE *X, size_t size_X, TYPE Y )
{
    if ( size_X == 0 )
        return 0;
    if ( X[0] >= Y )
        return 0;
    if ( X[size_X - 1] < Y )
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


// Subroutine to perform a quicksort
template <class T>
void interp::quicksort( size_t a_size, T *arr )
{
    long int n = static_cast<long int>( a_size );
    if ( n <= 1 )
        return;
    bool test;
    long int i, ir, j, jstack, k, l, istack[100];
    T a, tmp_a;
    jstack = 0;
    l      = 0;
    ir     = n - 1;
    while ( 1 ) {
        if ( ir - l < 7 ) { // Insertion sort when subarray small enough.
            for ( j = l + 1; j <= ir; j++ ) {
                a    = arr[j];
                test = true;
                for ( i = j - 1; i >= 0; i-- ) {
                    if ( arr[i] < a ) {
                        arr[i + 1] = a;
                        test       = false;
                        break;
                    }
                    arr[i + 1] = arr[i];
                }
                if ( test ) {
                    i          = l - 1;
                    arr[i + 1] = a;
                }
            }
            if ( jstack == 0 )
                return;
            ir = istack[jstack]; // Pop stack and begin a new round of partitioning.
            l  = istack[jstack - 1];
            jstack -= 2;
        } else {
            k = ( l + ir ) / 2; // Choose median of left, center and right elements as partitioning
                                // element a. Also rearrange so that a(l) < a(l+1) < a(ir).
            tmp_a      = arr[k];
            arr[k]     = arr[l + 1];
            arr[l + 1] = tmp_a;
            if ( arr[l] > arr[ir] ) {
                tmp_a   = arr[l];
                arr[l]  = arr[ir];
                arr[ir] = tmp_a;
            }
            if ( arr[l + 1] > arr[ir] ) {
                tmp_a      = arr[l + 1];
                arr[l + 1] = arr[ir];
                arr[ir]    = tmp_a;
            }
            if ( arr[l] > arr[l + 1] ) {
                tmp_a      = arr[l];
                arr[l]     = arr[l + 1];
                arr[l + 1] = tmp_a;
            }
            // Scan up to find element > a
            j = ir;
            a = arr[l + 1]; // Partitioning element.
            for ( i = l + 2; i <= ir; i++ ) {
                if ( arr[i] < a )
                    continue;
                while ( arr[j] > a ) // Scan down to find element < a.
                    j--;
                if ( j < i )
                    break;       // Pointers crossed. Exit with partitioning complete.
                tmp_a  = arr[i]; // Exchange elements of both arrays.
                arr[i] = arr[j];
                arr[j] = tmp_a;
            }
            arr[l + 1] = arr[j]; // Insert partitioning element in both arrays.
            arr[j]     = a;
            jstack += 2;
            // Push pointers to larger subarray on stack, process smaller subarray immediately.
            if ( ir - i + 1 >= j - l ) {
                istack[jstack]     = ir;
                istack[jstack - 1] = i;
                ir                 = j - 1;
            } else {
                istack[jstack]     = j - 1;
                istack[jstack - 1] = l;
                l                  = i;
            }
        }
    }
}


// Subroutine to perform a quicksort
template <class T1, class T2>
void interp::quicksort( size_t size, T1 *arr, T2 *brr )
{
    long int n = static_cast<long int>( size );
    if ( n <= 1 )
        return;
    bool test;
    long int i, ir, j, jstack, k, l, istack[100];
    T1 a, tmp_a;
    T2 b, tmp_b;
    jstack = 0;
    l      = 0;
    ir     = n - 1;
    while ( 1 ) {
        if ( ir - l < 7 ) { // Insertion sort when subarray small enough.
            for ( j = l + 1; j <= ir; j++ ) {
                a    = arr[j];
                b    = brr[j];
                test = true;
                for ( i = j - 1; i >= 0; i-- ) {
                    if ( arr[i] < a ) {
                        arr[i + 1] = a;
                        brr[i + 1] = b;
                        test       = false;
                        break;
                    }
                    arr[i + 1] = arr[i];
                    brr[i + 1] = brr[i];
                }
                if ( test ) {
                    i          = l - 1;
                    arr[i + 1] = a;
                    brr[i + 1] = b;
                }
            }
            if ( jstack == 0 )
                return;
            ir = istack[jstack]; // Pop stack and begin a new round of partitioning.
            l  = istack[jstack - 1];
            jstack -= 2;
        } else {
            k = ( l + ir ) / 2; // Choose median of left, center and right elements as partitioning
                                // element a. Also rearrange so that a(l) ? a(l+1) ? a(ir).
            tmp_a      = arr[k];
            arr[k]     = arr[l + 1];
            arr[l + 1] = tmp_a;
            tmp_b      = brr[k];
            brr[k]     = brr[l + 1];
            brr[l + 1] = tmp_b;
            if ( arr[l] > arr[ir] ) {
                tmp_a   = arr[l];
                arr[l]  = arr[ir];
                arr[ir] = tmp_a;
                tmp_b   = brr[l];
                brr[l]  = brr[ir];
                brr[ir] = tmp_b;
            }
            if ( arr[l + 1] > arr[ir] ) {
                tmp_a      = arr[l + 1];
                arr[l + 1] = arr[ir];
                arr[ir]    = tmp_a;
                tmp_b      = brr[l + 1];
                brr[l + 1] = brr[ir];
                brr[ir]    = tmp_b;
            }
            if ( arr[l] > arr[l + 1] ) {
                tmp_a      = arr[l];
                arr[l]     = arr[l + 1];
                arr[l + 1] = tmp_a;
                tmp_b      = brr[l];
                brr[l]     = brr[l + 1];
                brr[l + 1] = tmp_b;
            }
            // Scan up to find element > a
            j = ir;
            a = arr[l + 1]; // Partitioning element.
            b = brr[l + 1];
            for ( i = l + 2; i <= ir; i++ ) {
                if ( arr[i] < a )
                    continue;
                while ( arr[j] > a ) // Scan down to find element < a.
                    j--;
                if ( j < i )
                    break;       // Pointers crossed. Exit with partitioning complete.
                tmp_a  = arr[i]; // Exchange elements of both arrays.
                arr[i] = arr[j];
                arr[j] = tmp_a;
                tmp_b  = brr[i];
                brr[i] = brr[j];
                brr[j] = tmp_b;
            }
            arr[l + 1] = arr[j]; // Insert partitioning element in both arrays.
            arr[j]     = a;
            brr[l + 1] = brr[j];
            brr[j]     = b;
            jstack += 2;
            // Push pointers to larger subarray on stack, process smaller subarray immediately.
            if ( ir - i + 1 >= j - l ) {
                istack[jstack]     = ir;
                istack[jstack - 1] = i;
                ir                 = j - 1;
            } else {
                istack[jstack]     = j - 1;
                istack[jstack - 1] = l;
                l                  = i;
            }
        }
    }
}


inline void interp::sort( size_t n, const double *X, double *Y )
{
    // Copy the values of X
    for ( size_t i = 0; i < n; i++ )
        Y[i]       = X[i];
    // Sort the values
    quicksort( n, Y );
}


inline void interp::sort( size_t n, const double *X, double *Y, size_t *I )
{
    // Copy the values of X and create index array
    for ( size_t i = 0; i < n; i++ ) {
        Y[i] = X[i];
        I[i] = i;
    }
    // Sort the values
    quicksort( n, Y, I );
}


// Subroutine to find the unique elements in a list
inline void interp::unique( size_t *n, double *x )
{
    if ( ( *n ) <= 1 )
        return;
    // First perform a quicksort
    quicksort( *n, x );
    // Next remove duplicate entries
    size_t pos = 1;
    for ( size_t i = 1; i < *n; i++ ) {
        if ( x[i] != x[pos - 1] ) {
            x[pos] = x[i];
            pos++;
        }
    }
    *n = pos;
}


// Subroutine to find the unique elements in a list
inline void interp::unique( size_t n, const double *X, size_t *ny, double *Y )
{
    *ny = n;
    for ( size_t i = 0; i < n; i++ )
        Y[i]       = X[i];
    interp::unique( ny, Y );
}


inline void interp::unique(
    size_t nX, const double *X, size_t *nY, double *Y, size_t *I, size_t *J )
{
    // Copy the values of X and initialize the index vector I
    for ( size_t i = 0; i < nX; i++ ) {
        Y[i] = X[i];
        I[i] = i;
    }
    // Sort the values
    quicksort( nX, Y, I );
    // Delete duplicate entries
    size_t i = 0;
    J[I[0]]  = 0;
    for ( size_t j = 1; j < nX; j++ ) {
        if ( Y[i] == Y[j] ) {
            J[I[j]] = J[I[i]];
        } else {
            Y[i + 1]    = Y[j];
            I[i + 1]    = I[j];
            J[I[i + 1]] = i + 1;
            i++;
        }
    }
    *nY = i + 1;
}


// Modified bisection root finding
#ifdef ENABLE_STD_FUNCTION
template <class... Args>
double interp::bisection( std::function<double( double, Args... )> fun, double lb, double ub,
    double tol1, double tol2, Args... options )
{
    if ( ub <= lb )
        throw std::logic_error( "Error: lb <= ub" );
    double range[2] = { lb, ub };
    double x[500] = { 0 }, f[500] = { 0 };
    x[0] = lb;
    x[1] = ub;
    f[0] = fun( x[0], options... );
    f[1] = fun( x[1], options... );
    if ( fabs( f[0] ) < tol1 || fabs( f[1] ) < tol1 )
        return fabs( f[0] ) < tol1 ? x[0] : x[1];
    if ( ( f[0] < 0 && f[1] < 0 ) || ( f[0] > 0 && f[1] > 0 ) )
        throw std::logic_error( "Error: sign(f(lb)) == sign(f(ub))" );
    size_t i = 2;
    while ( ( range[1] - range[0] ) > tol2 ) {
        // Get the next guess
        x[i] = bisection_coeff( i, x, f, range );
        // Evaluate f
        f[i] = fun( x[i], options... );
        i++;
        if ( fabs( f[i - 1] ) < tol1 )
            break;
        if ( i > 500 )
            throw std::logic_error( "Excessive number of iterations" );
    }
    return x[i - 1];
}
#endif


// Fast approximate x^y
HOST_DEVICE inline double interp::fast_pow( double x, double y )
{
    union {
        double d;
        unsigned long long i;
    } v               = { x };
    const bool x_zero = v.i == 0;
    int li2           = ( v.i >> 52 ) & 0x7FF;
    li2 -= 1023; // Compute log2 of the exponent
    double m =
        2.220446049250313e-16 * static_cast<double>( v.i & 0xFFFFFFFFFFFFF ); // Compute fraction
    v.d =
        li2 +
        m * ( 1.420864533971306 +
                m * ( 0.156386111143355 * m - 0.577250645114661 ) ); // Add the log2( 1 + fraction )
    v.d *= y;                                                        // Compute y*log2(x)
    int w     = (int) ( ( v.d < 0 ) ? v.d - 1 : v.d );               // Get the integer power
    double f  = v.d - w;                                             // Get the fraction
    double f2 = 1.0 +
                f * ( 0.693147180559945 +
                        f * ( 0.230508889200065 + 0.076343930239989 * f ) ); // Compute 2^fraction
    v.i = static_cast<unsigned long long>( w + 1023 ) << 52;                 // Compute 2^w
    return ( x_zero || ( w < -1022 ) ) ? 0.0 : f2 * v.d; // Compute 2^(y*log2(x))
}


// Subroutine for a fast approximate exponential sum
// Note: scaling by the maximum ensures that we have no error for a single value
//   and we do not exceed the peak value, but increases runtime by ~50%
HOST_DEVICE inline double interp::fast_exp_avg( int N, const double *ai, const double *xi )
{
    union {
        double d;
        unsigned long long i;
    } y = { 0 };
    for ( int i = 0; i < N; i++ ) {
        union {
            double d;
            unsigned long long i;
        } v     = { xi[i] };
        int li2 = ( v.i >> 52 ) & 0x7FF;
        li2 -= 1023; // Compute log2 of the exponent
        double m = 2.220446049250313e-16 *
                   static_cast<double>( v.i & 0xFFFFFFFFFFFFF ); // Compute fraction
        v.d = li2 +
              m * ( 1.420864533971306 +
                      m * ( 0.156386111143355 * m -
                              0.577250645114661 ) ); // Add the log2( 1 + fraction )
        y.d += ai[i] * v.d;
    }
    int w     = (int) ( ( y.d < 0 ) ? y.d - 1 : y.d ); // Get the integer power
    double f  = y.d - w;                               // Get the fraction
    double f2 = 1.0 +
                f * ( 0.693147180559945 +
                        f * ( 0.230508889200065 + 0.076343930239989 * f ) ); // Compute 2^fraction
    y.i = static_cast<unsigned long long>( w + 1023 ) << 52;                 // Compute 2^w
    return ( w < -1022 ) ? 0.0 : f2 * y.d; // Compute 2^(y*log2(x))
}


// Subroutine to get the ratio for interpolation
HOST_DEVICE inline double interp::get_interp_ratio(
    double x0, double x1, double x, bool use_log, bool extrap )
{
    double y = 0;
    if ( !use_log ) {
        y = ( x - x0 ) / ( x1 - x0 );
    } else {
        // y = log(x/x0)/log(x1/x0);
        double m;
        union {
            double d;
            unsigned long long i;
        } v1 = { x / x0 };
        union {
            double d;
            unsigned long long i;
        } v2    = { x1 / x0 };
        int li1 = ( v1.i >> 52 ) & 0x7FF;
        int li2 = ( v2.i >> 52 ) & 0x7FF;
        li1 -= 1023; // Compute log2 of the exponent
        li2 -= 1023; // Compute log2 of the exponent
        m = 2.220446049250313e-16 *
            static_cast<double>( v1.i & 0xFFFFFFFFFFFFF ); // Compute fraction
        v1.d = li1 +
               m * ( 1.420864533971306 +
                       m * ( 0.156386111143355 * m -
                               0.577250645114661 ) ); // Add the log2( 1 + fraction )
        m = 2.220446049250313e-16 *
            static_cast<double>( v2.i & 0xFFFFFFFFFFFFF ); // Compute fraction
        v2.d = li2 +
               m * ( 1.420864533971306 +
                       m * ( 0.156386111143355 * m -
                               0.577250645114661 ) ); // Add the log2( 1 + fraction )
        y = v1.d / v2.d;
    }
    if ( !extrap ) {
        y = ( y > 0.0 ) ? y : 0.0;
        y = ( y < 1.0 ) ? y : 1.0;
    }
    return y;
}


#ifdef ENABLE_STD_FUNCTION


// Integrate the function using the midpoints
template <class T1, class T2>
HOST_DEVICE inline T1 interp::integrate_midpoint(
    const std::function<T1( T2 )> &f, const std::array<T2, 2> &range, int N )
{
    T2 dx = ( range[1] - range[0] ) / N;
    T1 y  = 0;
    for ( int i = 0; i < N; i++ )
        y += f( range[0] + ( i + 0.5 ) * dx );
    y *= dx;
    return y;
}


// Integrate the function using Simpson's rule
template <class T1, class T2>
HOST_DEVICE inline T1 interp::integrate_simpson(
    const std::function<T1( T2 )> &f, const std::array<T2, 2> &range, int N )
{
    if ( N <= 2 )
        return ( range[1] - range[0] ) / 6 *
               ( f( range[0] ) + 4 * f( ( range[0] + range[1] ) / 2 ) + f( range[1] ) );
    if ( N % 2 != 0 )
        throw std::logic_error( "Error: N must be even" );
    T2 dx = ( range[1] - range[0] ) / N;
    T1 y  = f( range[0] ) + f( range[1] ) + 4 * f( range[0] + dx );
    for ( int i = 1; i < N / 2; i++ ) {
        y += 2 * f( range[0] + 2 * i * dx );
        y += 4 * f( range[0] + 2 * i * dx + dx );
    }
    y *= ( dx / 3 );
    return y;
}


// Integrate the function using adaptive Simpson's rule
template <class T1, class T2>
HOST_DEVICE inline T1 adaptiveSimpsonsAux( const std::function<T1( T2 )> &fun, T2 a, T2 b, T2 tol,
    T1 S, T1 fa, T1 fb, T1 fc, int bottom, int &N_eval, const std::function<T2( T1 )> &norm )
{
    T2 c = ( a + b ) / 2, h = b - a;
    T2 d = ( a + c ) / 2, e = ( c + b ) / 2;
    T1 fd = fun( d ), fe = fun( e );
    T1 Sleft  = ( h / 12 ) * ( fa + 4 * fd + fc );
    T1 Sright = ( h / 12 ) * ( fc + 4 * fe + fb );
    T1 S2     = Sleft + Sright;
    N_eval += 2;
    T2 err = norm( S2 - S );
    if ( bottom <= 0 || err <= 15 * tol ) // magic 15 comes from error analysis
        return S2 + 0.066666666666667 * ( S2 - S );
    return adaptiveSimpsonsAux<T1, T2>(
               fun, a, c, tol / 2, Sleft, fa, fc, fd, bottom - 1, N_eval, norm ) +
           adaptiveSimpsonsAux<T1, T2>(
               fun, c, b, tol / 2, Sright, fc, fb, fe, bottom - 1, N_eval, norm );
}
template <class T1, class T2>
HOST_DEVICE inline T1 interp::integrate( const std::function<T1( T2 )> &fun,
    const std::array<T2, 2> &range, T2 tol, int *N_eval_out, const std::function<T2( T1 )> &norm )
{
    T2 h       = range[1] - range[0];
    T1 fa      = fun( range[0] );
    T1 fb      = fun( range[1] );
    T1 fc      = fun( range[0] + 0.5 * h );
    T1 S       = ( h / 6 ) * ( fa + 4 * fc + fb );
    int N_eval = 2;
    S          = adaptiveSimpsonsAux<T1, T2>(
        fun, range[0], range[1], tol, S, fa, fb, fc, 100, N_eval, norm );
    if ( N_eval_out != NULL )
        *N_eval_out = N_eval;
    return S;
}
template <class T1, class T2>
HOST_DEVICE inline T1 interp::integrate( const std::function<T1( T2, T2 )> &fun,
    const std::array<T2, 4> &range, T2 tol, int *N_eval_out, const std::function<T2( T1 )> &norm )
{
    std::array<T2, 2> range1 = { range[0], range[1] };
    std::array<T2, 2> range2 = { range[2], range[3] };
    int N_eval      = 0;
    int *N_eval_ptr = &N_eval;
    auto fun2       = [fun, range1, tol, N_eval_ptr, norm]( T2 y ) {
        auto fun3 = [fun, y, N_eval_ptr]( T2 x ) {
            ( *N_eval_ptr )++;
            return fun( x, y );
        };
        return integrate<T1, T2>( fun3, range1, tol, NULL, norm );
    };
    auto S = integrate<T1, T2>( fun2, range2, tol, NULL, norm );
    if ( N_eval_out != NULL )
        *N_eval_out = N_eval;
    return S;
}
template <class T1, class T2>
HOST_DEVICE inline T1 interp::integrate( const std::function<T1( T2, T2, T2 )> &fun,
    const std::array<T2, 6> &range, T2 tol, int *N_eval_out, const std::function<T2( T1 )> &norm )
{
    std::array<T2, 4> range1 = { range[0], range[1], range[2], range[3] };
    std::array<T2, 2> range2 = { range[4], range[5] };
    int N_eval      = 0;
    int *N_eval_ptr = &N_eval;
    auto fun2       = [fun, range1, tol, N_eval_ptr, norm]( T2 z ) {
        auto fun3 = [fun, z, N_eval_ptr]( T2 x, T2 y ) {
            ( *N_eval_ptr )++;
            return fun( x, y, z );
        };
        return integrate<T1, T2>( fun3, range1, tol, NULL, norm );
    };
    auto S = integrate<T1, T2>( fun2, range2, tol, NULL, norm );
    if ( N_eval_out != NULL )
        *N_eval_out = N_eval;
    return S;
}


#endif

#endif
