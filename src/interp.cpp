#include <stdexcept>
#include <string.h>
#include <algorithm>

#include "interp.h"


// Fast approximate x^y
double interp::fast_pow( double x, double y )
{
    union { double d; unsigned long long i; } v = { x };
    const bool x_zero = v.i==0;
    int li2 = (v.i>>52)&0x7FF;
    li2 -= 1023;                                                                    // Compute log2 of the exponent
    double m = 2.220446049250313e-16*static_cast<double>(v.i&0xFFFFFFFFFFFFF);      // Compute fraction
    v.d = li2 + m*(1.420864533971306+m*(0.156386111143355*m-0.577250645114661));    // Add the log2( 1 + fraction )
    v.d *= y;                                                                       // Compute y*log2(x)
    int w = (int)( (v.d < 0) ? v.d-1 : v.d );                                       // Get the integer power
    double f = v.d - w;                                                             // Get the fraction
    double f2 = 1.0+f*(0.693147180559945+f*(0.230508889200065+0.076343930239989*f)); // Compute 2^fraction
    v.i = static_cast<unsigned long long>(w+1023) << 52;                            // Compute 2^w
    return (x_zero||(w<-1022)) ? 0.0 : f2*v.d;                                      // Compute 2^(y*log2(x))
}


// Subroutine for a fast approximate exponential sum
// Note: scaling by the maximum ensures that we have no error for a single value
//   and we do not exceed the peak value, but increases runtime by ~50%
double interp::fast_exp_avg( int N, const double *ai, const double *xi )
{
    union { double d; unsigned long long i; } y = { 0 };
    for (int i=0; i<N; i++) {
        union { double d; unsigned long long i; } v = { xi[i] };
        int li2 = (v.i>>52)&0x7FF;
        li2 -= 1023;                                                                // Compute log2 of the exponent
        double m = 2.220446049250313e-16*static_cast<double>(v.i&0xFFFFFFFFFFFFF);  // Compute fraction
        v.d = li2 + m*(1.420864533971306+m*(0.156386111143355*m-0.577250645114661)); // Add the log2( 1 + fraction )
        y.d += ai[i]*v.d;
    }    
    int w = (int)( (y.d < 0) ? y.d-1 : y.d );                                       // Get the integer power
    double f = y.d - w;                                                             // Get the fraction
    double f2 = 1.0+f*(0.693147180559945+f*(0.230508889200065+0.076343930239989*f)); // Compute 2^fraction
    y.i = static_cast<unsigned long long>(w+1023) << 52;                            // Compute 2^w
    return (w<-1022) ? 0.0 : f2*y.d;                                                // Compute 2^(y*log2(x))
}


// Subroutine to get the ratio for interpolation
double interp::get_interp_ratio( double x0, double x1, double x, bool use_log, bool extrap )
{
    double y = 0;
    if ( !use_log ) {
        y = (x-x0)/(x1-x0);
    } else {
        // y = log(x/x0)/log(x1/x0);
        double m;
        union { double d; unsigned long long i; } v1 = { x/x0 };
        union { double d; unsigned long long i; } v2 = { x1/x0 };
        int li1 = (v1.i>>52)&0x7FF;
        int li2 = (v2.i>>52)&0x7FF;
        li1 -= 1023;                                                                    // Compute log2 of the exponent
        li2 -= 1023;                                                                    // Compute log2 of the exponent
        m = 2.220446049250313e-16*static_cast<double>(v1.i&0xFFFFFFFFFFFFF);            // Compute fraction
        v1.d = li1 + m*(1.420864533971306+m*(0.156386111143355*m-0.577250645114661));   // Add the log2( 1 + fraction )
        m = 2.220446049250313e-16*static_cast<double>(v2.i&0xFFFFFFFFFFFFF);            // Compute fraction
        v2.d = li2 + m*(1.420864533971306+m*(0.156386111143355*m-0.577250645114661));   // Add the log2( 1 + fraction )
        y = v1.d/v2.d;
    }
    if ( !extrap ) {
        y = (y>0.0) ? y:0.0;
        y = (y<1.0) ? y:1.0;
    }
    return y;
}


// Subroutine to perform linear interpolation
double interp::interp_linear(size_t N, const double *xi, const double *yi, double x) {
    size_t i = findfirstsingle( xi, N, x );
    if ( i==0 ) { i=1; }
    if ( i==N ) { i--; }
    double dx = (x-xi[i-1])/(xi[i]-xi[i-1]);
    double dx2 = 1.0-dx;
    double y = dx2*yi[i-1] + dx*yi[i];
    return(y);
}


// Subroutine to perform bi-linear interpolation using the full grid
double interp::bilinear(double x, double y, size_t N, size_t M, const double *xgrid, const double *ygrid, const double *f_grid) {
    size_t i, j;
    double f1, f2, f3, f4, dx, dy, f, dx2, dy2;
    i = findfirstsingle( xgrid, N, x );
    j = findfirstsingle( ygrid, M, y );
    if ( i == 0 ) { i = 1; }
    if ( j == 0 ) { j = 1; }
    if ( i==N ) { i--; }
    if ( j==M ) { j--; }
    dx = ( x - xgrid[i-1] ) / ( xgrid[i] - xgrid[i-1] );
    dy = ( y - ygrid[j-1] ) / ( ygrid[j] - ygrid[j-1] );
    f1 = f_grid[i-1+(j-1)*N];
    f2 = f_grid[i+(j-1)*N];
    f3 = f_grid[i-1+j*N];
    f4 = f_grid[i+j*N];
    dx2 = 1.0-dx;
    dy2 = 1.0-dy;
    f   = (dx*f2 + dx2*f1)*dy2 + (dx*f4 + dx2*f3)*dy;
    return(f);
}


// Subroutine to perform tri-linear interpolation using the full grid
double interp::trilinear(double x, double y, double z, size_t Nx, size_t Ny, size_t Nz, 
    const double *xgrid, const double *ygrid, const double *zgrid, const double *f_grid)
{
    size_t i, j, k;
    double f1, f2, f3, f4, f5,f6,f7,f8, dx, dy, dz, f, dx2, dy2, dz2;
    i = findfirstsingle( xgrid, Nx, x );
    j = findfirstsingle( ygrid, Ny, y );
    k = findfirstsingle( zgrid, Nz, z );
    if ( i == 0 ) { i = 1; }
    if ( j == 0 ) { j = 1; }
    if ( k == 0 ) { k = 1; }
    dx = ( x - xgrid[i-1] ) / ( xgrid[i] - xgrid[i-1] );
    dy = ( y - ygrid[j-1] ) / ( ygrid[j] - ygrid[j-1] );
    dz = ( z - zgrid[k-1] ) / ( zgrid[k] - zgrid[k-1] );
    f1 = f_grid[i-1+(j-1)*Nx+(k-1)*Nx*Ny];
    f2 = f_grid[i+(j-1)*Nx+(k-1)*Nx*Ny];
    f3 = f_grid[i-1+j*Nx+(k-1)*Nx*Ny];
    f4 = f_grid[i+j*Nx+(k-1)*Nx*Ny];
    f5 = f_grid[i-1+(j-1)*Nx+(k)*Nx*Ny];
    f6 = f_grid[i+(j-1)*Nx+(k)*Nx*Ny];
    f7 = f_grid[i-1+j*Nx+(k)*Nx*Ny];
    f8 = f_grid[i+j*Nx+(k)*Nx*Ny];
    dx2 = 1.0-dx;
    dy2 = 1.0-dy;
    dz2 = 1.0-dz;
    f   = ((dx*f2 + dx2*f1)*dy2 + (dx*f4 + dx2*f3)*dy)*dz2 +((dx*f6 + dx2*f5)*dy2 + (dx*f8 + dx2*f7)*dy)*dz;
    return(f);
}


// Subroutine to perform cubic hermite interpolation
double interp::interp_pchip(size_t N, const double *xi, const double *yi, double x)
{
    if ( x <= xi[0] || N<=2 ) {
        double dx = (x-xi[0])/(xi[1]-xi[0]);
        return (1.0-dx)*yi[0] + dx*yi[1];
    } else if ( x >= xi[N-1] ) {
        double dx = (x-xi[N-2])/(xi[N-1]-xi[N-2]);
        return (1.0-dx)*yi[N-2] + dx*yi[N-1];
    }
    size_t i = findfirstsingle( xi, N, x );
    double f1 = yi[i-1];
    double f2 = yi[i];
    double dx = (x-xi[i-1])/(xi[i]-xi[i-1]);
    // Compute the gradient in normalized coordinates [0,1]
    double g1=0, g2=0;
    if ( i<=1 ) {
        g1 = f2-f1;
    } else if ( ( f1<f2 && f1>yi[i-2] ) || ( f1>f2 && f1<yi[i-2] ) ) {
        // Compute the gradient by using a 3-point finite difference to f'(x)
        // Note: the real gradient is g1/(xi[i]-xi[i-1])
        double f0 = yi[i-2];
        double dx1 = xi[i-1]-xi[i-2];
        double dx2 = xi[i]-xi[i-1];
        double a1 = (dx2-dx1)/dx1;
        double a2 = dx1/(dx1+dx2);
        g1 = a1*(f1-f0)+a2*(f2-f0);
        double g_max = 2*dx2*std::min(fabs(f1-f0)/dx1,fabs(f2-f1)/dx2);
        g1 = ((g1>=0)?1:-1)*std::min(fabs(g1),g_max);
    }
    if ( i>=N-1 ) {
        g2 = f2-f1;
    } else if ( ( f2<f1 && f2>yi[i+1] ) || ( f2>f1 && f2<yi[i+1] ) ) {
        // Compute the gradient by using a 3-point finite difference to f'(x)
        // Note: the real gradient is g2/(xi[i]-xi[i-1])
        double f0 = yi[i+1];
        double dx1 = xi[i]-xi[i-1];
        double dx2 = xi[i+1]-xi[i];
        double a1 = -dx2/(dx1+dx2);
        double a2 = (dx2-dx1)/dx2;
        g2 = a1*(f1-f0)+a2*(f2-f0);
        double g_max = 2*dx1*std::min(fabs(f2-f1)/dx1,fabs(f0-f2)/dx2);
        g2 = ((g2>=0)?1:-1)*std::min(fabs(g2),g_max);
    }
    // Perform the interpolation
    double dx2 = dx*dx;
    double f = f1 + dx2*(2*dx-3)*(f1-f2) + dx*g1 - dx2*(g1+(1-dx)*(g1+g2));
    return f;
}


// This function calculates the FWHM by finding the narrowest region that contains 76% of the energy
double interp::calc_width(size_t n, const double *x, const double *y) 
{
    // Check the inputs
    if ( n < 2 ) {
        std::cerr << "There must be at least two points\n";
        return -1;
    }
    for (size_t i=0; i<n; i++) {
        if ( y[i] < 0.0 ) {
            std::cerr << "Negitive values in y detected\n";
            return -1;
        }
    }
    for (size_t i=1; i<n; i++) {
        if ( x[i] <= x[i-1] ) {
            std::cerr << "x must be sorted and unique\n";
            return -1;
        }
    }
    // Calculate the normalized cumulative sum (x may not be uniformly spaced)
    double *ys = new double[n];
    ys[0] = 0.0;
    for (size_t i=1; i<n; i++)
        ys[i] = ys[i-1] + (x[i]-x[i-1])*0.5*(y[i]+y[i-1]);  // ys = int(y*dx)
    if ( ys[n-1] == 0.0 ) {
        std::cerr << "y is all zeros\n";
        delete [] ys;
        return -1;
    }
    double tmp = 1.0/ys[n-1];
    for (size_t i=0; i<n; i++)
        ys[i] *= tmp;
    // Find the width that contains ~76% of the energy for each starting position
    const double f = 0.760968108550488;     // erf(sqrt(log(2)))
    double FWHM = x[n-1]-x[0];
    for (size_t i=0; i<n; i++) {
        if ( ys[i] > 1-f )
            break;
        double x2 = interp_linear(n,ys,x,ys[i]+f);
        if ( x2-x[i] < FWHM )
            FWHM = x2-x[i];
    }
    delete [] ys;
    return FWHM;
}


// Calculate coefficients for the bisection method
double interp::bisection_coeff( size_t N, const double* x_in, const double* r_in, double* range_out )
{
    if ( N < 2 )
        throw std::logic_error("Error: N<2");
    // Copy and sort x,r by r
    double *x = new double[N];
    double *r = new double[N];
    memcpy(x,x_in,N*sizeof(double));
    memcpy(r,r_in,N*sizeof(double));
    quicksort(N,r,x);
    // Check that we have two signs for r
    if ( r[0]>0.0 || r[N-1]<0.0 ) {
        delete [] x;
        delete [] r;
        throw std::logic_error("r does not have two different signs");
    }
    // Find r > 0
    size_t i = findfirstsingle(r,N,0.0);
    double range[2] = { std::min(x[i-1],x[i]), std::max(x[i-1],x[i]) };
    double y = 0;
    if ( N < 5 ) {
        // Use simple bisection for the first couple of iterations
        y = 0.5*(range[0]+range[1]);
    } else if ( i==1 || i==N-1 ) {
        // We are at the boundary which reduces the gradient that we can use
        // Use an uneven bisection to try an ensure we have 2 points on each side
        if ( i==1 ) {
            y = 0.8*x[0] + 0.2*x[1];
        } else {
            y = 0.2*x[N-2] + 0.8*x[N-1];
        }
    } else {
        // Use cubic interpolation
        y = interp_pchip(N,r,x,0.0);
        // Check that y is within the range
        y = std::max(std::min(y,range[1]),range[0]);
        // Adjust y to move it toward the center
        y = 0.9*y + 0.1*(0.5*(range[0]+range[1]));
    }
    // Finished
    delete [] x;
    delete [] r;
    if ( range_out!=NULL ) {
        range_out[0] = range[0];
        range_out[1] = range[1];
    }
    return y;
}



