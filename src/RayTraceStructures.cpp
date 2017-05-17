#include "RayTraceStructures.h"
#include "utilities/MPI_functions.h"
#include "utilities/RayUtilities.h"

#include "ProfilerApp.h"

#include <math.h>
#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <string.h>


#ifdef USE_OPENACC
// Include this file for OpenACC API
#ifdef _OPENACC
#include <openacc.h>
#endif
#endif
#ifdef USE_CUDA
extern const RayTrace::ray_gain_struct *ray_gain_struct_copy_device_cuda(
    size_t N, const RayTrace::ray_gain_struct *arr );
extern const RayTrace::ray_seed_struct *ray_seed_struct_copy_device_cuda(
    const RayTrace::ray_seed_struct &seed );
extern void ray_gain_struct_free_device_cuda( size_t N, const RayTrace::ray_gain_struct *arr );
extern void ray_seed_struct_free_device_cuda( const RayTrace::ray_seed_struct *dev_ptr );
#endif

#ifdef USE_ATOMIC_MODEL
#include "ProfilerApp.h"
#include "AtomicModel/file_utilities.h"
#include "AtomicModel/interp.h"
#else
#include "AtomicModel/interp.h"
namespace file_utilities {
static void write_scalar_int( FILE *, const char *, int ) {}
static void write_variable_float( FILE *, const char *, size_t, const float *, bool = false ) {}
static void write_variable_double( FILE *, const char *, size_t, const double *, bool = false ) {}
};
#endif


/**
 *  Helper function for deep copy.
 *  Copy-in an array and also set an associated pointer in device memory.
 *  Treatment for case where N>0, but array_host is NULL:  Pointer on device set to NULL, no copy.
 * @param[in] array_host  Pointer to array of ELEMENT_T
 * @param[in] device_arr_ptr_address  Device address for an ELEMENT_T * which will be set to
 * copied-in array location.
 * @param[in] N  Number of ELEMENT_T in the array to copy-in.
 */
#ifdef USE_OPENACC
template <typename ELEMENT_T>
static void copy_in_array_and_set_devptr(
    const ELEMENT_T *array_host, ELEMENT_T **device_arr_ptr_address, size_t N )
{
    if ( array_host && N > 0 ) {
        ELEMENT_T *arr = const_cast<ELEMENT_T *>( array_host ); // bad API const-ness spec.

        // copy the array
        ELEMENT_T *devptr = static_cast<ELEMENT_T *>( acc_copyin( arr, N * sizeof( ELEMENT_T ) ) );

        // set the array pointer on the device
        acc_memcpy_to_device( device_arr_ptr_address, &devptr, sizeof( devptr ) );
    } else { // detect case where source pointer is NULL.
        ELEMENT_T *null_ptr( NULL );
        acc_memcpy_to_device( device_arr_ptr_address, &null_ptr, sizeof( null_ptr ) );
    }
}
#endif


// Functions to check if two values are equal
inline bool approx_equal( double x, double y, double tol = 1e-6 )
{
    return ( 2.0 * fabs( ( x - y ) / ( x + y ) ) < tol ) || ( x + y == 0.0 );
}


// Functions to check if two vectors are equal
inline bool approx_equal( size_t N, const double *x, const double *y, double tol = 1e-6 )
{
    bool equal = true;
    for ( size_t i = 0; i < N; i++ )
        equal = equal && ( ( 2.0 * fabs( ( x[i] - y[i] ) / ( x[i] + y[i] ) ) < tol ) ||
                             ( x[i] + y[i] == 0.0 ) );
    return equal;
}


// Helper function to pack a value to a char array and increment N_bytes
template <class TYPE>
inline void pack_buffer( TYPE value, size_t &N_bytes, char *data )
{
    memcpy( &data[N_bytes], &value, sizeof( TYPE ) );
    N_bytes += sizeof( TYPE );
}

// Helper function to unpack a value from a char array and increment N_bytes
template <class TYPE>
inline TYPE unpack_buffer( size_t &N_bytes, const char *data )
{
    TYPE value;
    memcpy( &value, &data[N_bytes], sizeof( TYPE ) );
    N_bytes += sizeof( TYPE );
    return value;
}


/**********************************************************************
* This function reads the byte array header (if used).                *
* Note:  This function must work if no byte header is used, or if the *
* data structure changes.                                             *
* Variables:                                                          *
*    data  - Input:  Pointer to the byte array                        *
*    data2 - Output: Pointer to byte array starting after the header  *
**********************************************************************/
RayTrace::byte_array_header::byte_array_header()
{
    id = 237; // DO NOT MODIFY!!!
    RAY_ASSERT( sizeof( byte_array_header ) == 16 );
    size_int    = sizeof( int );
    size_double = sizeof( double );
    version     = 0;
    type        = 0;
    for ( int i   = 0; i < 2; i++ )
        unused[i] = 0;
    for ( int i    = 0; i < 5; i++ )
        N_bytes[i] = 0;
    for ( int i  = 0; i < 4; i++ )
        flags[i] = 0;
}
RayTrace::byte_array_header RayTrace::load_byte_header( const char *data, char **data2 )
{
    RAY_ASSERT( sizeof( byte_array_header ) == 16 );
    byte_array_header head;
    const byte_array_header *data1 = reinterpret_cast<const byte_array_header *>( data );
    if ( data1[0].id == 237 ) {
        // The header is used
        *data2 = const_cast<char *>( &data[sizeof( byte_array_header )] );
        head   = data1[0];
        RAY_ASSERT( sizeof( int ) == head.size_int );
        RAY_ASSERT( sizeof( double ) == head.size_double );
    } else {
        // The header is not used (we are looking at old data)
        *data2       = const_cast<char *>( data );
        head.version = 0;
    }
    return head;
}
// Function to set N_bytes in the byte_array_header
void RayTrace::set_N_bytes( byte_array_header *head, size_t N_bytes )
{
    RAY_ASSERT( N_bytes < 1099511627776 );
    if ( sizeof( unsigned int ) == 4 ) {
        head->N_bytes[0] = (unsigned char) ( N_bytes / 4294967296 );
        unsigned int r   = (unsigned int) ( N_bytes % 4294967296 );
        memcpy( &head->N_bytes[1], &r, 4 );
    } else {
        // Need a 4 byte unsigned integer
        assert( false );
    }
}
// Function to read N_bytes in the byte_array_header
size_t RayTrace::read_N_bytes( const byte_array_header &head )
{
    size_t N_bytes = 0;
    N_bytes        = head.N_bytes[0];
    N_bytes *= ( (size_t) 4294967296 );
    if ( sizeof( unsigned int ) == 4 ) {
        unsigned int r;
        memcpy( &r, &head.N_bytes[1], 4 );
        N_bytes += r;
    } else {
        // Need a 4 byte unsigned integer
        RAY_ASSERT( false );
    }
    return N_bytes;
}
// Function to check that N_bytes in the byte_array_header matches the provided number
// Note: we only started being careful about this in version 2 or greater
void RayTrace::check_N_bytes( const byte_array_header &head, size_t N_bytes )
{
    size_t N_bytes_head = read_N_bytes( head );
    if ( N_bytes != N_bytes_head && N_bytes_head != 0 && head.version >= 2 ) {
        std::string message =
            "Error: Number of bytes read does not equal the number of bytes in the header\n";
        message += Utilities::stringf(
            "   N_bytes_read = %i, N_bytes_header = %i\n", (int) N_bytes, (int) N_bytes_head );
        RAY_ERROR( message );
    }
}


/******************************************************************
* Constructors/Destructors for EUV_beam_struct                    *
******************************************************************/
RayTrace::EUV_beam_struct::EUV_beam_struct()
    : run_ASE( true ),
      run_sat( true ),
      run_refract( true ),
      R_scale( -1 ),
      G_scale( -1 ),
      lambda( 0 ),
      A( 0 ),
      Nc( 0 ),
      x( NULL ),
      y( NULL ),
      a( NULL ),
      b( NULL ),
      z( NULL ),
      v( NULL ),
      dx( 0 ),
      dy( 0 ),
      da( 0 ),
      db( 0 ),
      dz( 0 ),
      dv( NULL ),
      v0( 0 ),
      nx( 0 ),
      ny( 0 ),
      nz( 0 ),
      na( 0 ),
      nb( 0 ),
      nv( 0 )
{
}
RayTrace::EUV_beam_struct::~EUV_beam_struct() { delete_data(); }
#ifdef ENABLE_MOVE_CONSTRUCTOR
RayTrace::EUV_beam_struct::EUV_beam_struct( EUV_beam_struct &&rhs ) : EUV_beam_struct()
{
    swap( rhs );
}
RayTrace::EUV_beam_struct &RayTrace::EUV_beam_struct::operator=( EUV_beam_struct &&rhs )
{
    swap( rhs );
    return *this;
}
#endif
void RayTrace::EUV_beam_struct::initialize( int Nx, int Ny, int Nz, int Na, int Nb, int Nv )
{
    // Delete existing data
    delete_data();
    // Initialize the data
    nx = Nx;
    ny = Ny;
    nz = Nz;
    na = Na;
    nb = Nb;
    nv = Nv;
    x  = new double[nx];
    y  = new double[ny];
    z  = new double[nz];
    a  = new double[na];
    b  = new double[nb];
    v  = new double[nv];
    dv = new double[nv];
    memset( x, 0, nx * sizeof( double ) );
    memset( y, 0, ny * sizeof( double ) );
    memset( z, 0, nz * sizeof( double ) );
    memset( a, 0, na * sizeof( double ) );
    memset( b, 0, nb * sizeof( double ) );
    memset( v, 0, nv * sizeof( double ) );
    memset( dv, 0, nv * sizeof( double ) );
}
void RayTrace::EUV_beam_struct::swap( EUV_beam_struct &rhs )
{
    std::swap( run_ASE, rhs.run_ASE );
    std::swap( run_sat, rhs.run_sat );
    std::swap( run_refract, rhs.run_refract );
    std::swap( R_scale, rhs.R_scale );
    std::swap( G_scale, rhs.G_scale );
    std::swap( lambda, rhs.lambda );
    std::swap( A, rhs.A );
    std::swap( Nc, rhs.Nc );
    std::swap( x, rhs.x );
    std::swap( y, rhs.y );
    std::swap( a, rhs.a );
    std::swap( b, rhs.b );
    std::swap( z, rhs.z );
    std::swap( v, rhs.v );
    std::swap( dx, rhs.dx );
    std::swap( dy, rhs.dy );
    std::swap( da, rhs.da );
    std::swap( db, rhs.db );
    std::swap( dz, rhs.dz );
    std::swap( dv, rhs.dv );
    std::swap( v0, rhs.v0 );
    std::swap( nx, rhs.nx );
    std::swap( ny, rhs.ny );
    std::swap( nz, rhs.nz );
    std::swap( na, rhs.na );
    std::swap( nb, rhs.nb );
    std::swap( nv, rhs.nv );
}
void RayTrace::EUV_beam_struct::copy( const RayTrace::EUV_beam_struct &rhs )
{
    // Delete existing data
    delete_data();
    // Copy the data
    run_ASE     = rhs.run_ASE;
    run_sat     = rhs.run_sat;
    run_refract = rhs.run_refract;
    R_scale     = rhs.R_scale;
    G_scale     = rhs.G_scale;
    lambda      = rhs.lambda;
    A           = rhs.A;
    Nc          = rhs.Nc;
    dx          = rhs.dx;
    dy          = rhs.dy;
    da          = rhs.da;
    db          = rhs.db;
    dz          = rhs.dz;
    v0          = rhs.v0;
    nx          = rhs.nx;
    ny          = rhs.ny;
    nz          = rhs.nz;
    na          = rhs.na;
    nb          = rhs.nb;
    nv          = rhs.nv;
    x           = new double[nx];
    y           = new double[ny];
    z           = new double[nz];
    a           = new double[na];
    b           = new double[nb];
    v           = new double[nv];
    dv          = new double[nv];
    memcpy( x, rhs.x, nx * sizeof( double ) );
    memcpy( y, rhs.y, ny * sizeof( double ) );
    memcpy( z, rhs.z, nz * sizeof( double ) );
    memcpy( a, rhs.a, na * sizeof( double ) );
    memcpy( b, rhs.b, nb * sizeof( double ) );
    memcpy( v, rhs.v, nv * sizeof( double ) );
    memcpy( dv, rhs.dv, nv * sizeof( double ) );
}
void RayTrace::EUV_beam_struct::delete_data()
{
    delete[] x;
    x = NULL;
    delete[] y;
    y = NULL;
    delete[] a;
    a = NULL;
    delete[] b;
    b = NULL;
    delete[] z;
    z = NULL;
    delete[] v;
    v = NULL;
    delete[] dv;
    dv          = NULL;
    run_ASE     = true;
    run_sat     = true;
    run_refract = true;
    R_scale     = -1.0;
    G_scale     = -1.0;
    lambda      = 0.0;
    A           = 0.0;
    Nc          = 0.0;
    dx          = 0.0;
    dy          = 0.0;
    da          = 0.0;
    db          = 0.0;
    dz          = 0.0;
    v0          = 0.0;
    nx          = 0;
    ny          = 0;
    na          = 0;
    nb          = 0;
    nv          = 0;
}
bool RayTrace::EUV_beam_struct::valid() const
{
    bool is_valid = true;
    for ( int i = 0; i < nx; i++ ) {
        if ( x[i] != x[i] ) {
            is_valid = false;
        }
    }
    for ( int i = 0; i < ny; i++ ) {
        if ( y[i] != y[i] ) {
            is_valid = false;
        }
    }
    for ( int i = 0; i < nz; i++ ) {
        if ( z[i] != z[i] ) {
            is_valid = false;
        }
    }
    for ( int i = 0; i < na; i++ ) {
        if ( a[i] != a[i] ) {
            is_valid = false;
        }
    }
    for ( int i = 0; i < nb; i++ ) {
        if ( b[i] != b[i] ) {
            is_valid = false;
        }
    }
    for ( int i = 0; i < nv; i++ ) {
        if ( v[i] != v[i] ) {
            is_valid = false;
        }
    }
    for ( int i = 0; i < nv; i++ ) {
        if ( dv[i] != dv[i] ) {
            is_valid = false;
        }
    }
    return is_valid;
}
bool RayTrace::EUV_beam_struct::operator==( const RayTrace::EUV_beam_struct &rhs ) const
{
    if ( nx != rhs.nx || ny != rhs.ny || nz != rhs.nz || na != rhs.na || nb != rhs.nb ||
         nv != rhs.nv )
        return false;
    if ( run_ASE != rhs.run_ASE || run_sat != rhs.run_sat || run_refract != rhs.run_refract )
        return false;
    if ( !approx_equal( R_scale, rhs.R_scale ) || !approx_equal( G_scale, rhs.G_scale ) )
        return false;
    if ( !approx_equal( lambda, rhs.lambda ) || !approx_equal( A, rhs.A ) ||
         !approx_equal( Nc, rhs.Nc ) )
        return false;
    if ( !approx_equal( dx, rhs.dx ) || !approx_equal( dy, rhs.dy ) ||
         !approx_equal( dx, rhs.dx ) || !approx_equal( da, rhs.da ) || !approx_equal( v0, rhs.v0 ) )
        return false;
    bool equal = true;
    equal      = equal && approx_equal( nx, x, rhs.x );
    equal      = equal && approx_equal( ny, y, rhs.y );
    equal      = equal && approx_equal( na, a, rhs.a );
    equal      = equal && approx_equal( nb, b, rhs.b );
    equal      = equal && approx_equal( nv, v, rhs.v );
    equal      = equal && approx_equal( nv, dv, rhs.dv );
    return equal;
}


/**********************************************************************
* Function to compress an euv_beam data to a byte array               *
**********************************************************************/
std::pair<char *, size_t> RayTrace::EUV_beam_struct::pack( int ) const
{
    // First determine the number of bytes needed to store the data and allocate the data
    size_t N_bytes = sizeof( byte_array_header ); // Storage space for the byte arry header
    N_bytes += 3 * sizeof( bool );                // Storage space for the bool scalar values
    N_bytes += 7 * sizeof( int );                 // Storage space for the int scalar values
    N_bytes += 10 * sizeof( double );             // Storage space for the double scalar values
    N_bytes += nx * sizeof( double );             // Storage space for x
    N_bytes += ny * sizeof( double );             // Storage space for y
    N_bytes += nz * sizeof( double );             // Storage space for z
    N_bytes += na * sizeof( double );             // Storage space for a
    N_bytes += nb * sizeof( double );             // Storage space for b
    N_bytes += 2 * nv * sizeof( double );         // Storage space for v and dv
    char *data = new char[N_bytes];
    if ( data == NULL )
        return std::pair<char *, size_t>( (char *) NULL, 0 );
    // Create and save the byte array header
    RAY_ASSERT( sizeof( byte_array_header ) == 16 );
    byte_array_header *data1 = reinterpret_cast<byte_array_header *>( data );
    data1[0]                 = byte_array_header();
    byte_array_header &head  = *data1;
    head.version             = 2;
    head.type                = 2;
    // Save run_ASE, run_sat, run_refract
    size_t Ns = sizeof( byte_array_header );
    pack_buffer<bool>( run_ASE, Ns, data );
    pack_buffer<bool>( run_sat, Ns, data );
    pack_buffer<bool>( run_refract, Ns, data );
    // Save nx, ny, na, nb, nv, nz_sum
    pack_buffer<int>( nx, Ns, data );
    pack_buffer<int>( ny, Ns, data );
    pack_buffer<int>( nz, Ns, data );
    pack_buffer<int>( na, Ns, data );
    pack_buffer<int>( nb, Ns, data );
    pack_buffer<int>( nv, Ns, data );
    pack_buffer<int>( 0, Ns, data ); // Old field kept for backward compatibility (nz_sub)
    // Save R_scale, G_scale, lambda, Nc, dx, dy, dz, da, db, v0
    pack_buffer<double>( R_scale, Ns, data );
    pack_buffer<double>( G_scale, Ns, data );
    pack_buffer<double>( lambda, Ns, data );
    pack_buffer<double>( Nc, Ns, data );
    pack_buffer<double>( dx, Ns, data );
    pack_buffer<double>( dy, Ns, data );
    pack_buffer<double>( dz, Ns, data );
    pack_buffer<double>( da, Ns, data );
    pack_buffer<double>( db, Ns, data );
    pack_buffer<double>( v0, Ns, data );
    // Save x, y, a, b, z, v, dv
    for ( int i = 0; i < nx; i++ )
        pack_buffer<double>( x[i], Ns, data );
    for ( int i = 0; i < ny; i++ )
        pack_buffer<double>( y[i], Ns, data );
    for ( int i = 0; i < nz; i++ )
        pack_buffer<double>( z[i], Ns, data );
    for ( int i = 0; i < na; i++ )
        pack_buffer<double>( a[i], Ns, data );
    for ( int i = 0; i < nb; i++ )
        pack_buffer<double>( b[i], Ns, data );
    for ( int i = 0; i < nv; i++ )
        pack_buffer<double>( v[i], Ns, data );
    for ( int i = 0; i < nv; i++ )
        pack_buffer<double>( dv[i], Ns, data );
    RAY_ASSERT( N_bytes == Ns );
    set_N_bytes( &head, N_bytes );
    return std::pair<char *, size_t>( data, N_bytes );
}


/**********************************************************************
* Function to convert a byte array to a euv_beam data structure       *
**********************************************************************/
void RayTrace::EUV_beam_struct::unpack( std::pair<const char *, size_t> data )
{
    // Delete existing data
    delete_data();
    // Read the header information
    char *data2            = NULL;
    byte_array_header head = load_byte_header( data.first, &data2 );
    if ( head.version > 0 && head.type != 2 )
        RAY_ERROR( "Error: The byte array does not appear to contain euv_beam data" );
    // Save run_ASE, run_sat, run_refract
    size_t N_bytes = 0;
    run_ASE        = unpack_buffer<bool>( N_bytes, data2 );
    run_sat        = unpack_buffer<bool>( N_bytes, data2 );
    run_refract    = unpack_buffer<bool>( N_bytes, data2 );
    // Save nx, ny, na, nb, nv, nz_sum
    nx = unpack_buffer<int>( N_bytes, data2 );
    ny = unpack_buffer<int>( N_bytes, data2 );
    nz = unpack_buffer<int>( N_bytes, data2 );
    na = unpack_buffer<int>( N_bytes, data2 );
    nb = unpack_buffer<int>( N_bytes, data2 );
    nv = unpack_buffer<int>( N_bytes, data2 );
    unpack_buffer<int>( N_bytes, data2 ); // Old field kept for backward compatibility (nz_sub)
    // nz_sub used to be stored in data_int[5], now unused
    if ( nx < 1 || ny < 1 || nz < 1 || na < 1 || nb < 1 || nv < 1 )
        RAY_ERROR( "Internal error" );
    // Save R_scale, G_scale, lambda, Nc, dx, dy, dz, da, db, v0
    R_scale = unpack_buffer<double>( N_bytes, data2 );
    G_scale = unpack_buffer<double>( N_bytes, data2 );
    lambda  = unpack_buffer<double>( N_bytes, data2 );
    Nc      = unpack_buffer<double>( N_bytes, data2 );
    dx      = unpack_buffer<double>( N_bytes, data2 );
    dy      = unpack_buffer<double>( N_bytes, data2 );
    dz      = unpack_buffer<double>( N_bytes, data2 );
    da      = unpack_buffer<double>( N_bytes, data2 );
    db      = unpack_buffer<double>( N_bytes, data2 );
    v0      = unpack_buffer<double>( N_bytes, data2 );
    // Allocate space for x, y, a, b, z, v, dv
    x  = new double[nx];
    y  = new double[ny];
    z  = new double[nz];
    a  = new double[na];
    b  = new double[nb];
    v  = new double[nv];
    dv = new double[nv];
    // Save x, y, a, b, z, v, dv
    for ( int i = 0; i < nx; i++ )
        x[i]    = unpack_buffer<double>( N_bytes, data2 );
    for ( int i = 0; i < ny; i++ )
        y[i]    = unpack_buffer<double>( N_bytes, data2 );
    for ( int i = 0; i < nz; i++ )
        z[i]    = unpack_buffer<double>( N_bytes, data2 );
    for ( int i = 0; i < na; i++ )
        a[i]    = unpack_buffer<double>( N_bytes, data2 );
    for ( int i = 0; i < nb; i++ )
        b[i]    = unpack_buffer<double>( N_bytes, data2 );
    for ( int i = 0; i < nv; i++ )
        v[i]    = unpack_buffer<double>( N_bytes, data2 );
    for ( int i = 0; i < nv; i++ )
        dv[i]   = unpack_buffer<double>( N_bytes, data2 );
    // Check the number of bytes read
    check_N_bytes( head, N_bytes + sizeof( byte_array_header ) );
}


/******************************************************************
* Constructors/Destructors for seed_beam_shape_struct             *
******************************************************************/
RayTrace::seed_beam_shape_struct::seed_beam_shape_struct()
{
    n   = 0;
    nv  = 0;
    T   = NULL;
    It  = NULL;
    Ivt = NULL;
}
RayTrace::seed_beam_shape_struct::seed_beam_shape_struct(
    const RayTrace::seed_beam_shape_struct &rhs )
{
    T   = NULL;
    It  = NULL;
    Ivt = NULL;
    initialize( rhs.n, rhs.nv );
    memcpy( T, rhs.T, n * sizeof( double ) );
    memcpy( It, rhs.It, 3 * n * sizeof( double ) );
    memcpy( Ivt, rhs.Ivt, 3 * n * nv * sizeof( double ) );
}
RayTrace::seed_beam_shape_struct &RayTrace::seed_beam_shape_struct::operator=(
    const RayTrace::seed_beam_shape_struct &rhs )
{
    if ( this != &rhs ) {
        this->initialize( rhs.n, rhs.nv );
        memcpy( this->T, rhs.T, n * sizeof( double ) );
        memcpy( this->It, rhs.It, 3 * n * sizeof( double ) );
        memcpy( this->Ivt, rhs.Ivt, 3 * n * nv * sizeof( double ) );
    }
    return *this;
}
RayTrace::seed_beam_shape_struct::~seed_beam_shape_struct() { delete_data(); }
void RayTrace::seed_beam_shape_struct::initialize( int nT, int nV )
{
    // Delete existing data
    delete_data();
    // Initialize the data
    n   = nT;
    nv  = nV;
    T   = new double[n];
    It  = new double[3 * n];
    Ivt = new double[3 * n * nv];
    memset( T, 0, n * sizeof( double ) );
    memset( It, 0, 3 * n * sizeof( double ) );
    memset( Ivt, 0, 3 * n * nv * sizeof( double ) );
}
void RayTrace::seed_beam_shape_struct::delete_data()
{
    n  = 0;
    nv = 0;
    delete[] T;
    T = NULL;
    delete[] It;
    It = NULL;
    delete[] Ivt;
    Ivt = NULL;
}
bool RayTrace::seed_beam_shape_struct::valid() const
{
    bool is_valid = true;
    for ( int i = 0; i < n; i++ ) {
        if ( T[i] != T[i] ) {
            is_valid = false;
        }
    }
    for ( int i = 0; i < 3 * n; i++ ) {
        if ( It[i] != It[i] ) {
            is_valid = false;
        }
    }
    for ( int i = 0; i < 3 * n * nv; i++ ) {
        if ( Ivt[i] != Ivt[i] ) {
            is_valid = false;
        }
    }
    return is_valid;
}
bool RayTrace::seed_beam_shape_struct::operator==(
    const RayTrace::seed_beam_shape_struct &rhs ) const
{
    if ( n != rhs.n || nv != rhs.nv )
        return false;
    bool equal = true;
    equal      = equal && approx_equal( n, T, rhs.T );
    equal      = equal && approx_equal( 3 * n, It, rhs.It );
    equal      = equal && approx_equal( 3 * n * nv, Ivt, rhs.Ivt );
    return equal;
}


/**********************************************************************
* Function to convert seed_beam data to a byte array                  *
**********************************************************************/
std::pair<char *, size_t> RayTrace::seed_beam_shape_struct::pack( int compression ) const
{
    if ( n <= 0 || nv <= 0 )
        RAY_ERROR( "seed_beam_shape_struct is invalid (n==0 or nv==0)" );
    // Estimate the space required and allocate the data
    size_t N_bytes_estimate = sizeof( byte_array_header ); // Storage space for the byte arry header
    N_bytes_estimate += 2 * sizeof( int );        // Storage space for the int scalar values
    N_bytes_estimate += 3 * sizeof( int );        // Storage space for the bytes used for each array
    N_bytes_estimate += n * sizeof( double );     // Storage space for T
    N_bytes_estimate += 3 * n * sizeof( double ); // Storage space for It
    N_bytes_estimate += 3 * n * nv * sizeof( double ); // Storage space for Ivt
    char *data = new char[N_bytes_estimate];
    if ( data == NULL )
        return std::pair<char *, size_t>( (char *) NULL, 0 );
    // Create and save the byte array header
    assert( sizeof( byte_array_header ) == 16 );
    byte_array_header *data1 = reinterpret_cast<byte_array_header *>( data );
    data1[0]                 = byte_array_header();
    byte_array_header &head  = *data1;
    head.version             = 2;
    head.type                = 6;
    head.flags[0]            = (unsigned char) compression;
    // Save n, nv, and compression
    size_t pos = sizeof( byte_array_header );
    pack_buffer<int>( n, pos, data );
    pack_buffer<int>( nv, pos, data );
    // Save T, It, Ivt
    if ( compression == 0 ) {
        // Store the raw data
        pack_buffer<int>( n, pos, data );
        pack_buffer<int>( 3 * n, pos, data );
        pack_buffer<int>( 3 * n * nv, pos, data );
        for ( int j = 0; j < n; j++ )
            pack_buffer<double>( T[j], pos, data );
        for ( int j = 0; j < n * 3; j++ )
            pack_buffer<double>( It[j], pos, data );
        for ( int j = 0; j < nv * n * 3; j++ )
            pack_buffer<double>( Ivt[j], pos, data );
    } else if ( compression == 1 ) {
        // Remove zeros
        unsigned char *data_T = NULL, *data_It = NULL, *data_Ivt = NULL;
        size_t size_T   = Utilities::compress_array( n, T, compression, &data_T );
        size_t size_It  = Utilities::compress_array( n * 3, It, compression, &data_It );
        size_t size_Ivt = Utilities::compress_array( nv * n * 3, Ivt, compression, &data_Ivt );
        pack_buffer<int>( (int) size_T, pos, data );
        pack_buffer<int>( (int) size_It, pos, data );
        pack_buffer<int>( (int) size_Ivt, pos, data );
        memcpy( &data[pos], data_T, size_T );
        pos += size_T;
        memcpy( &data[pos], data_It, size_It );
        pos += size_It;
        memcpy( &data[pos], data_Ivt, size_Ivt );
        pos += size_Ivt;
        delete[] data_T;
        delete[] data_It;
        delete[] data_Ivt;
    } else if ( compression == 2 ) {
        // Remove zeros and store the arrays in single precision
        unsigned char *data_T = NULL, *data_It = NULL, *data_Ivt = NULL;
        float *tmp_T   = new float[n];
        float *tmp_It  = new float[n * 3];
        float *tmp_Ivt = new float[nv * n * 3];
        for ( int j  = 0; j < n; j++ )
            tmp_T[j] = (float) T[j];
        for ( int j   = 0; j < n * 3; j++ )
            tmp_It[j] = (float) It[j];
        for ( int j     = 0; j < nv * n * 3; j++ )
            tmp_Ivt[j]  = (float) Ivt[j];
        size_t size_T   = Utilities::compress_array( n, tmp_T, compression, &data_T );
        size_t size_It  = Utilities::compress_array( n * 3, tmp_It, compression, &data_It );
        size_t size_Ivt = Utilities::compress_array( nv * n * 3, tmp_Ivt, compression, &data_Ivt );
        delete[] tmp_T;
        delete[] tmp_It;
        delete[] tmp_Ivt;
        pack_buffer<int>( (int) size_T, pos, data );
        pack_buffer<int>( (int) size_It, pos, data );
        pack_buffer<int>( (int) size_Ivt, pos, data );
        memcpy( &data[pos], data_T, size_T );
        pos += size_T;
        memcpy( &data[pos], data_It, size_It );
        pos += size_It;
        memcpy( &data[pos], data_Ivt, size_Ivt );
        pos += size_Ivt;
        delete[] data_T;
        delete[] data_It;
        delete[] data_Ivt;
    }
    size_t N_bytes = pos;
    RAY_ASSERT( N_bytes <= N_bytes_estimate );
    set_N_bytes( &head, N_bytes );
    return std::pair<char *, size_t>( data, N_bytes );
}


/**********************************************************************
* Function to convert a byte array to a seed_beam data structure      *
**********************************************************************/
void RayTrace::seed_beam_shape_struct::unpack( std::pair<const char *, size_t> data_in )
{
    // Delete the existing data
    delete_data();
    // Read the header information
    const char *data       = data_in.first;
    char *data2            = NULL;
    byte_array_header head = load_byte_header( data, &data2 );
    if ( head.version > 0 && head.type != 6 )
        RAY_ERROR( "Error: The byte array does not appear to contain seed_beam_shape data" );
    unsigned char compression = head.flags[0];
    size_t N_bytes            = data2 - data;
    size_t N_bytes_head       = read_N_bytes( head );
    if ( N_bytes_head == 0 && compression != 0 )
        RAY_ERROR( "Error: the byte array header appears invalid" );
    // Read n, nv
    int *data_int = (int *) &data[N_bytes];
    n             = data_int[0];
    nv            = data_int[1];
    N_bytes += 2 * sizeof( int );
    // Read the bytes for each array
    data_int        = (int *) &data[N_bytes];
    size_t size_T   = data_int[0];
    size_t size_It  = data_int[1];
    size_t size_Ivt = data_int[2];
    N_bytes += 3 * sizeof( int );
    if ( compression == 0 ) {
        // No compression, read the raw data
        T                   = new double[n];
        It                  = new double[3 * n];
        Ivt                 = new double[3 * n * nv];
        memcpy(T,&data[N_bytes],n*sizeof(double));
        N_bytes += n*sizeof(double);
        memcpy(It,&data[N_bytes],3*n*sizeof(double));
        N_bytes += 3*n*sizeof(double);
        memcpy(Ivt,&data[N_bytes],3*n*nv*sizeof(double));
        N_bytes += 3*n*nv*sizeof(double);
    } else if ( compression == 1 ) {
        // The data was saved without zeros
        Utilities::decompress_array( n, size_T, (unsigned char *) &data[N_bytes], compression, &T );
        N_bytes += size_T;
        Utilities::decompress_array(
            3 * n, size_It, (unsigned char *) &data[N_bytes], compression, &It );
        N_bytes += size_It;
        Utilities::decompress_array(
            3 * n * nv, size_Ivt, (unsigned char *) &data[N_bytes], compression, &Ivt );
        N_bytes += size_Ivt;
    } else if ( compression == 2 ) {
        // The data was saved without zeros and in single precision
        float *tmp_data_T, *tmp_data_It, *tmp_data_Ivt;
        Utilities::decompress_array(
            n, size_T, (unsigned char *) &data[N_bytes], compression, &tmp_data_T );
        N_bytes += size_T;
        Utilities::decompress_array(
            3 * n, size_It, (unsigned char *) &data[N_bytes], compression, &tmp_data_It );
        N_bytes += size_It;
        Utilities::decompress_array(
            3 * n * nv, size_Ivt, (unsigned char *) &data[N_bytes], compression, &tmp_data_Ivt );
        N_bytes += size_Ivt;
        T   = new double[n];
        It  = new double[3 * n];
        Ivt = new double[3 * n * nv];
        for ( int j = 0; j < n; j++ )
            T[j]    = tmp_data_T[j];
        for ( int j = 0; j < 3 * n; j++ )
            It[j]   = tmp_data_It[j];
        for ( int j = 0; j < 3 * n * nv; j++ )
            Ivt[j]  = tmp_data_Ivt[j];
        delete[] tmp_data_T;
        delete[] tmp_data_It;
        delete[] tmp_data_Ivt;
    } else {
        RAY_ERROR( "Error: Unsupported compression type" );
    }
    // Check the number of bytes read
    check_N_bytes( head, N_bytes );
}


/******************************************************************
* Constructors/Destructors for seed_beam_shape_struct             *
******************************************************************/
RayTrace::seed_beam_struct::seed_beam_struct()
{
    x = NULL;
    y = NULL;
    a = NULL;
    b = NULL;
    delete_data();
}
RayTrace::seed_beam_struct::~seed_beam_struct() { delete_data(); }
void RayTrace::seed_beam_struct::initialize( int Nx, int Ny, int Na, int Nb )
{
    // Delete existing data
    delete_data();
    // Initialize the data
    nx = Nx;
    ny = Ny;
    na = Na;
    nb = Nb;
    x  = new double[nx];
    y  = new double[ny];
    a  = new double[na];
    b  = new double[nb];
    memset( x, 0, nx * sizeof( double ) );
    memset( y, 0, ny * sizeof( double ) );
    memset( a, 0, na * sizeof( double ) );
    memset( b, 0, nb * sizeof( double ) );
    seed_shape.clear();
    tau.clear();
    use_transform.clear();
}
RayTrace::seed_beam_struct::seed_beam_struct( seed_beam_struct &&rhs ):
    seed_beam_struct()
{
    swap( rhs );
}
RayTrace::seed_beam_struct& RayTrace::seed_beam_struct::operator=( seed_beam_struct &&rhs )
{
    if ( this != &rhs )
        swap( rhs );
    return *this;
}
void RayTrace::seed_beam_struct::delete_data()
{
    dx     = 0;
    dy     = 0;
    da     = 0;
    db     = 0;
    nx     = 0;
    ny     = 0;
    na     = 0;
    nb     = 0;
    Wx     = 0;
    Wy     = 0;
    Wa     = 0;
    Wb     = 0;
    x0     = 0;
    y0     = 0;
    a0     = 0;
    b0     = 0;
    Wv     = 0;
    Wt     = 0;
    t0     = 0;
    E      = 0;
    target = 0;
    chirp  = 0;
    delete[] x;
    x = NULL;
    delete[] y;
    y = NULL;
    delete[] a;
    a = NULL;
    delete[] b;
    b = NULL;
    seed_shape.clear();
    tau.clear();
    use_transform.clear();
}
void RayTrace::seed_beam_struct::swap( seed_beam_struct& rhs)
{
    std::swap( x,  rhs.x );
    std::swap( y,  rhs.y );
    std::swap( a,  rhs.a );
    std::swap( b,  rhs.b );
    std::swap( dx, rhs.dx );
    std::swap( dy, rhs.dy );
    std::swap( da, rhs.da );
    std::swap( db, rhs.db );
    std::swap( nx, rhs.nx );
    std::swap( ny, rhs.ny );
    std::swap( na, rhs.na );
    std::swap( nb, rhs.nb );
    std::swap( Wx, rhs.Wx );
    std::swap( Wy, rhs.Wy );
    std::swap( Wa, rhs.Wa );
    std::swap( Wb, rhs.Wb );
    std::swap( Wv, rhs.Wv );
    std::swap( Wt, rhs.Wt );
    std::swap( x0, rhs.x0 );
    std::swap( y0, rhs.y0 );
    std::swap( a0, rhs.a0 );
    std::swap( b0, rhs.b0 );
    std::swap( t0, rhs.t0 );
    std::swap( E,  rhs.E );
    std::swap( target, rhs.target );
    std::swap( chirp, rhs.chirp );
    std::swap( seed_shape, rhs.seed_shape );
    std::swap( tau, rhs.tau );
    std::swap( use_transform, rhs.use_transform );
}
bool RayTrace::seed_beam_struct::valid() const
{
    bool is_valid = true;
    for ( int i = 0; i < nx; i++ ) {
        if ( x[i] != x[i] ) {
            is_valid = false;
        }
    }
    for ( int i = 0; i < ny; i++ ) {
        if ( y[i] != y[i] ) {
            is_valid = false;
        }
    }
    for ( int i = 0; i < na; i++ ) {
        if ( a[i] != a[i] ) {
            is_valid = false;
        }
    }
    for ( int i = 0; i < nb; i++ ) {
        if ( b[i] != b[i] ) {
            is_valid = false;
        }
    }
    for ( size_t i = 0; i < seed_shape.size(); i++ ) {
        if ( !seed_shape[i].valid() )
            is_valid = false;
        if ( tau[i] != tau[i] )
            is_valid = false;
    }
    return is_valid;
}
bool RayTrace::seed_beam_struct::operator==( const RayTrace::seed_beam_struct &rhs ) const
{
    if ( nx != rhs.nx || ny != rhs.ny || na != rhs.na || nb != rhs.nb )
        return false;
    if ( !approx_equal( dx, rhs.dx ) || !approx_equal( dy, rhs.dy ) ||
         !approx_equal( da, rhs.da ) || !approx_equal( db, rhs.db ) )
        return false;
    if ( !approx_equal( Wx, rhs.Wx ) || !approx_equal( Wy, rhs.Wy ) ||
         !approx_equal( Wa, rhs.Wa ) || !approx_equal( Wb, rhs.Wb ) )
        return false;
    if ( !approx_equal( Wv, rhs.Wv ) || !approx_equal( Wt, rhs.Wt ) )
        return false;
    if ( !approx_equal( x0, rhs.x0 ) || !approx_equal( y0, rhs.y0 ) ||
         !approx_equal( a0, rhs.a0 ) || !approx_equal( b0, rhs.b0 ) )
        return false;
    if ( !approx_equal( t0, rhs.t0 ) || !approx_equal( E, rhs.E ) ||
         !approx_equal( target, rhs.target ) || !approx_equal( chirp, rhs.chirp ) )
        return false;
    bool equal = true;
    equal      = equal && approx_equal( nx, x, rhs.x );
    equal      = equal && approx_equal( ny, y, rhs.y );
    equal      = equal && approx_equal( na, a, rhs.a );
    equal      = equal && approx_equal( nb, b, rhs.b );
    for ( size_t i = 0; i < seed_shape.size(); i++ ) {
        // if ( seed_shape[i] != rhs.seed_shape[i] )
        //    equal = false;
        if ( !approx_equal( tau[i], rhs.tau[i] ) )
            equal = false;
        if ( use_transform[i] != use_transform[i] )
            equal = false;
    }
    return equal;
}


/**********************************************************************
* Function to convert seed_beam data to a byte array                  *
**********************************************************************/
std::pair<char *, size_t> RayTrace::seed_beam_struct::pack( int compression ) const
{
    // Check the inputs
    if ( compression < 0 || compression > 2 ) {
        // Note: the compression option only needs to apply to the seed_shape structures
        perr << "Unknown or unsupported compression type\n";
        return std::pair<char *, size_t>( (char *) NULL, 0 );
    }
    // Pack the seed_shape structures so we know how much space they will need
    int N = (int) seed_shape.size();
    std::pair<char *, size_t> *seed_shape_data = NULL;
    if ( N > 0 ) {
        seed_shape_data = new std::pair<char *, size_t>[N];
        for ( int i            = 0; i < N; i++ )
            seed_shape_data[i] = seed_shape[i].pack( compression );
    }
    // Estimate the number of bytes needed to store the data and allocate the data
    size_t N_bytes_est = 0;
    N_bytes_est += sizeof( byte_array_header ); // Storage space for the byte arry header
    N_bytes_est += 4 * sizeof( int );           // Storage space for the int values
    N_bytes_est += 18 * sizeof( double );       // Storage space for the double values
    N_bytes_est += nx * sizeof( double );       // Storage space for x
    N_bytes_est += ny * sizeof( double );       // Storage space for y
    N_bytes_est += na * sizeof( double );       // Storage space for a
    N_bytes_est += nb * sizeof( double );       // Storage space for b
    N_bytes_est += sizeof( int );               // Storage space for N
    N_bytes_est += N * sizeof( double );        // Storage space for tau
    N_bytes_est += N * sizeof( bool );          // Storage space for use_transform
    for ( int i = 0; i < N; i++ )
        N_bytes_est += seed_shape_data[i].second + sizeof( int ); // Storage space for seed_shap
    char *data = new char[N_bytes_est];
    if ( data == NULL ) {
        for ( int i = 0; i < N; i++ )
            delete[] seed_shape_data[i].first;
        delete[] seed_shape_data;
        return std::pair<char *, size_t>( (char *) NULL, 0 );
    }
    // Create and save the byte array header
    RAY_ASSERT( sizeof( byte_array_header ) == 16 );
    byte_array_header *data1 = reinterpret_cast<byte_array_header *>( data );
    data1[0]                 = byte_array_header();
    byte_array_header &head  = *data1;
    head.version             = 2;
    head.type                = 3;
    head.flags[0]            = (char) compression;
    // Save nx, ny, na, nb
    size_t pos = sizeof( byte_array_header );
    pack_buffer<int>( nx, pos, data );
    pack_buffer<int>( ny, pos, data );
    pack_buffer<int>( na, pos, data );
    pack_buffer<int>( nb, pos, data );
    // Save dx, dy, da, db, Wx, Wy, Wa, Wb, Wv, Wt, x0, y0, a0, b0, t0, E, target, chirp
    pack_buffer<double>( dx, pos, data );
    pack_buffer<double>( dy, pos, data );
    pack_buffer<double>( da, pos, data );
    pack_buffer<double>( db, pos, data );
    pack_buffer<double>( Wx, pos, data );
    pack_buffer<double>( Wy, pos, data );
    pack_buffer<double>( Wa, pos, data );
    pack_buffer<double>( Wb, pos, data );
    pack_buffer<double>( Wv, pos, data );
    pack_buffer<double>( Wt, pos, data );
    pack_buffer<double>( x0, pos, data );
    pack_buffer<double>( y0, pos, data );
    pack_buffer<double>( a0, pos, data );
    pack_buffer<double>( b0, pos, data );
    pack_buffer<double>( t0, pos, data );
    pack_buffer<double>( E, pos, data );
    pack_buffer<double>( target, pos, data );
    pack_buffer<double>( chirp, pos, data );
    // Save x, y, a, b
    for ( int i = 0; i < nx; i++ )
        pack_buffer<double>( x[i], pos, data );
    for ( int i = 0; i < ny; i++ )
        pack_buffer<double>( y[i], pos, data );
    for ( int i = 0; i < na; i++ )
        pack_buffer<double>( a[i], pos, data );
    for ( int i = 0; i < nb; i++ )
        pack_buffer<double>( b[i], pos, data );
    // Save N, tau, and use_transform
    pack_buffer<int>( N, pos, data );
    for ( int i = 0; i < N; i++ )
        pack_buffer<double>( tau[i], pos, data );
    for ( int i = 0; i < N; i++ )
        pack_buffer<bool>( use_transform[i], pos, data );
    // Save seed_shape
    for ( int i = 0; i < N; i++ ) {
        int N_bytes = static_cast<int>( seed_shape_data[i].second );
        pack_buffer<int>( N_bytes, pos, data );
        memcpy( &data[pos], seed_shape_data[i].first, N_bytes );
        pos += N_bytes;
    }
    size_t N_bytes = pos;
    // Check the results and finish
    if ( N_bytes != N_bytes_est ) {
        perr << "Error converting data\n";
        pos = 0;
        delete[] data;
        return std::pair<char *, size_t>( (char *) NULL, 0 );
    }
    set_N_bytes( &head, N_bytes );
    // Delete the temporary data
    for ( int i = 0; i < N; i++ )
        delete[] seed_shape_data[i].first;
    delete[] seed_shape_data;
    return std::pair<char *, size_t>( data, N_bytes );
}


/**********************************************************************
* Function to convert a byte array to a seed_beam data structure      *
**********************************************************************/
size_t read_old_seed_shape_data(
    const char *data, int nv, int compression, RayTrace::seed_beam_shape_struct &seed_shape );
void RayTrace::seed_beam_struct::unpack( std::pair<const char *, size_t> data_in )
{
    // Delete exisiting data
    delete_data();
    // Read the header information
    const char *data       = data_in.first;
    char *data_head        = NULL;
    byte_array_header head = load_byte_header( data, &data_head );
    if ( head.version > 0 && head.type != 3 )
        RAY_ERROR( "Error: The byte array does not appear to contain seed_beam data" );
    unsigned char compression = head.flags[0];
    size_t N_bytes_head       = read_N_bytes( head );
    if ( N_bytes_head == 0 && compression != 0 )
        RAY_ERROR( "Error: the byte array header appears invalid" );
    // Load nx, ny, na, nb
    size_t pos = data_head - data;
    nx         = unpack_buffer<int>( pos, data );
    ny         = unpack_buffer<int>( pos, data );
    na         = unpack_buffer<int>( pos, data );
    nb         = unpack_buffer<int>( pos, data );
    // Load dx, dy, da, db, Wx, Wy, Wa, Wb, Wv, Wt, x0, y0, a0, b0, t0, E, target
    dx     = unpack_buffer<double>( pos, data );
    dy     = unpack_buffer<double>( pos, data );
    da     = unpack_buffer<double>( pos, data );
    db     = unpack_buffer<double>( pos, data );
    Wx     = unpack_buffer<double>( pos, data );
    Wy     = unpack_buffer<double>( pos, data );
    Wa     = unpack_buffer<double>( pos, data );
    Wb     = unpack_buffer<double>( pos, data );
    Wv     = unpack_buffer<double>( pos, data );
    Wt     = unpack_buffer<double>( pos, data );
    x0     = unpack_buffer<double>( pos, data );
    y0     = unpack_buffer<double>( pos, data );
    a0     = unpack_buffer<double>( pos, data );
    b0     = unpack_buffer<double>( pos, data );
    t0     = unpack_buffer<double>( pos, data );
    E      = unpack_buffer<double>( pos, data );
    target = unpack_buffer<double>( pos, data );
    chirp  = unpack_buffer<double>( pos, data );
    // Load x, y, a, b
    x = new double[nx];
    y = new double[ny];
    a = new double[na];
    b = new double[nb];
    for ( int i = 0; i < nx; i++ )
        x[i]    = unpack_buffer<double>( pos, data );
    for ( int i = 0; i < ny; i++ )
        y[i]    = unpack_buffer<double>( pos, data );
    for ( int i = 0; i < na; i++ )
        a[i]    = unpack_buffer<double>( pos, data );
    for ( int i = 0; i < nb; i++ )
        b[i]    = unpack_buffer<double>( pos, data );
    // Load N, tau, and use_transform
    if ( head.version >= 2 ) {
        // Use new data format
        int N = unpack_buffer<int>( pos, data );
        if ( N > 0 ) {
            // Load tau, and use_transform
            tau.resize( N );
            for ( int i = 0; i < N; i++ )
                tau[i]  = unpack_buffer<double>( pos, data );
            use_transform.resize( N );
            for ( int i          = 0; i < N; i++ )
                use_transform[i] = unpack_buffer<bool>( pos, data );
            // Load seed_shape
            seed_shape.resize( N );
            for ( int i = 0; i < N; i++ ) {
                size_t N_bytes_tmp = static_cast<size_t>( unpack_buffer<int>( pos, data ) );
                std::pair<const char *, size_t> data_seed_shape( &data[pos], N_bytes_tmp );
                seed_shape[i].unpack( data_seed_shape );
                pos += N_bytes_tmp;
            }
        }
    } else if ( head.version == 1 ) {
        // Use old data format
        int N  = unpack_buffer<int>( pos, data );
        int nv = unpack_buffer<int>( pos, data );
        if ( N != 0 ) {
            // Load seed_shape
            if ( N < 0 )
                RAY_ERROR( "Error loading seed_beam.seed_shape, N<0 or nv does not match" );
            use_transform.resize( N );
            tau.resize( N );
            seed_shape.resize( N );
            for ( int i = 0; i < N; i++ ) {
                // Save seed.use_transform
                use_transform[i] = unpack_buffer<bool>( pos, data );
                // Save seed.tau
                tau[i] = unpack_buffer<double>( pos, data );
                // Save the seed.shape
                pos += read_old_seed_shape_data( &data[pos], nv, compression, seed_shape[i] );
            }
        }
    } else {
        throw std::logic_error( "Unknown data format for seed beam" );
    }
    // Check the number of bytes read
    check_N_bytes( head, pos );
}
// Function to read the data for the old format used to store seed_shape data
// Keep for backward compatibility
size_t read_old_seed_shape_data(
    const char *data, int nv, int compression, RayTrace::seed_beam_shape_struct &seed_shape )
{
    size_t N_bytes      = 0;
    const int *data_int = (const int *) data;
    int n               = data_int[0];
    N_bytes += sizeof( int );
    int n_T   = n;
    int n_It  = n * 3;
    int n_Ivt = nv * n * 3;
    double *T = NULL, *It = NULL, *Ivt = NULL;
    if ( compression == 0 ) {
        // No compression, read the raw data
        T                         = new double[n_T];
        It                        = new double[n_It];
        Ivt                       = new double[n_Ivt];
        memcpy( T, &data[N_bytes], n_T * sizeof( double ) );
        N_bytes += n_T * sizeof( double );
        memcpy( It, &data[N_bytes], n_It * sizeof( double ) );
        N_bytes += n_It * sizeof( double );
        memcpy( Ivt, &data[N_bytes], n_Ivt * sizeof( double ) );
        N_bytes += n_Ivt * sizeof( double );
    } else if ( compression == 1 ) {
        // The data was saved without zeros
        data_int        = (int *) &data[N_bytes];
        size_t size_T   = data_int[0];
        size_t size_It  = data_int[1];
        size_t size_Ivt = data_int[2];
        N_bytes += 3 * sizeof( int );
        Utilities::decompress_array(
            n_T, N_bytes, (unsigned char *) &data[N_bytes], compression, &T );
        N_bytes += size_T;
        Utilities::decompress_array(
            n_It, N_bytes, (unsigned char *) &data[N_bytes], compression, &It );
        N_bytes += size_It;
        Utilities::decompress_array(
            n_Ivt, N_bytes, (unsigned char *) &data[N_bytes], compression, &Ivt );
        N_bytes += size_Ivt;
    } else if ( compression == 2 ) {
        // The data was saved without zeros and in single precision
        data_int        = (int *) &data[N_bytes];
        size_t size_T   = data_int[0];
        size_t size_It  = data_int[1];
        size_t size_Ivt = data_int[2];
        N_bytes += 3 * sizeof( int );
        float *tmp_data_T, *tmp_data_It, *tmp_data_Ivt;
        Utilities::decompress_array(
            n_T, N_bytes, (unsigned char *) &data[N_bytes], compression, &tmp_data_T );
        N_bytes += size_T;
        Utilities::decompress_array(
            n_It, N_bytes, (unsigned char *) &data[N_bytes], compression, &tmp_data_It );
        N_bytes += size_It;
        Utilities::decompress_array(
            n_Ivt, N_bytes, (unsigned char *) &data[N_bytes], compression, &tmp_data_Ivt );
        N_bytes += size_Ivt;
        T   = new double[n_T];
        It  = new double[n_It];
        Ivt = new double[n_Ivt];
        for ( int j = 0; j < n_T; j++ )
            T[j]    = tmp_data_T[j];
        for ( int j = 0; j < n_It; j++ )
            It[j]   = tmp_data_It[j];
        for ( int j = 0; j < n_Ivt; j++ )
            Ivt[j]  = tmp_data_Ivt[j];
    } else {
        RAY_ERROR( "Error: Unsupported compression type" );
    }
    seed_shape.initialize( n, nv );
    memcpy( seed_shape.T, T, n_T * sizeof( double ) );
    memcpy( seed_shape.It, It, n_It * sizeof( double ) );
    memcpy( seed_shape.Ivt, Ivt, n_Ivt * sizeof( double ) );
    delete[] T;
    delete[] It;
    delete[] Ivt;
    return N_bytes;
}


/******************************************************************
* Constructors/Destructors for seed_beam_shape_struct             *
******************************************************************/
RayTrace::ray_seed_struct::ray_seed_struct()
{
    f0 = 0.0;
    for ( int i = 0; i < 5; i++ ) {
        dim[i] = 0;
        x[i]   = NULL;
        f[i]   = NULL;
    }
}
RayTrace::ray_seed_struct::~ray_seed_struct() { delete_data(); }
void RayTrace::ray_seed_struct::initialize( int DIM[5] )
{
    // Delete existing data
    delete_data();
    // Initialize the data
    for ( int i = 0; i < 5; i++ ) {
        dim[i] = DIM[i];
        x[i]   = new double[dim[i]];
        f[i]   = new double[dim[i]];
        memset( x[i], 0, dim[i] * sizeof( double ) );
        memset( f[i], 0, dim[i] * sizeof( double ) );
    }
}
void RayTrace::ray_seed_struct::delete_data()
{
    for ( int i = 0; i < 5; i++ ) {
        dim[i] = 0;
        delete[] x[i];
        delete[] f[i];
        x[i] = NULL;
        f[i] = NULL;
    }
}
bool RayTrace::ray_seed_struct::is_zero( const EUV_beam_struct &euv_beam ) const
{
    bool is_zero = f0 < 1e-100;
    double I_max = 0.0;
    for ( int i = 0; i < euv_beam.nx; i++ ) {
        if ( euv_beam.x[i] < x[0][0] || euv_beam.x[i] > x[0][dim[0] - 1] ) {
            continue;
        }
        I_max = std::max( I_max, interp::interp_linear( dim[0], x[0], f[0], euv_beam.x[i] ) );
    }
    is_zero = is_zero || I_max < 1e-100;
    I_max   = 0.0;
    for ( int j = 0; j < euv_beam.ny; j++ ) {
        if ( euv_beam.y[j] < x[1][0] || euv_beam.y[j] > x[1][dim[1] - 1] ) {
            continue;
        }
        I_max = std::max( I_max, interp::interp_linear( dim[1], x[1], f[1], euv_beam.y[j] ) );
    }
    is_zero = is_zero || I_max < 1e-100;
    I_max   = 0.0;
    for ( int k = 0; k < euv_beam.na; k++ ) {
        if ( euv_beam.a[k] < x[2][0] || euv_beam.a[k] > x[2][dim[2] - 1] ) {
            continue;
        }
        I_max = std::max( I_max, interp::interp_linear( dim[2], x[2], f[2], euv_beam.a[k] ) );
    }
    is_zero = is_zero || I_max < 1e-100;
    I_max   = 0.0;
    for ( int m = 0; m < euv_beam.nb; m++ ) {
        if ( euv_beam.b[m] < x[3][0] || euv_beam.b[m] > x[3][dim[3] - 1] ) {
            continue;
        }
        I_max = std::max( I_max, interp::interp_linear( dim[3], x[3], f[3], euv_beam.b[m] ) );
    }
    return is_zero;
}
std::pair<char *, size_t> RayTrace::ray_seed_struct::pack( int ) const
{
    size_t N_bytes = 5 * sizeof( int ) + sizeof( double );
    for ( int i = 0; i < 5; i++ )
        N_bytes += 2 * dim[i] * sizeof( double );
    char *data = new char[N_bytes];
    size_t pos = 0;
    memcpy( &data[pos], dim, 5 * sizeof( int ) );
    pos += 5 * sizeof( int );
    for ( int i = 0; i < 5; i++ ) {
        memcpy( &data[pos], x[i], dim[i] * sizeof( double ) );
        pos += dim[i] * sizeof( double );
        memcpy( &data[pos], f[i], dim[i] * sizeof( double ) );
        pos += dim[i] * sizeof( double );
    }
    memcpy( &data[pos], &f0, sizeof( double ) );
    pos += sizeof( double );
    RAY_ASSERT( pos == N_bytes );
    return std::pair<char *, size_t>( data, N_bytes );
}
void RayTrace::ray_seed_struct::unpack( std::pair<const char *, size_t> data_in )
{
    const char *data     = data_in.first;
    const size_t N_bytes = data_in.second;
    size_t pos           = 0;
    memcpy( dim, &data[pos], 5 * sizeof( int ) );
    pos += 5 * sizeof( int );
    for ( int i = 0; i < 5; i++ ) {
        x[i] = new double[dim[i]];
        f[i] = new double[dim[i]];
        memcpy( x[i], &data[pos], dim[i] * sizeof( double ) );
        pos += dim[i] * sizeof( double );
        memcpy( f[i], &data[pos], dim[i] * sizeof( double ) );
        pos += dim[i] * sizeof( double );
    }
    memcpy( &f0, &data[pos], sizeof( double ) );
    pos += sizeof( double );
    RAY_ASSERT( pos == N_bytes );
}
const RayTrace::ray_seed_struct *RayTrace::ray_seed_struct::copy_device() const
{
#if defined( USE_OPENACC ) && defined( _OPENACC )
    // Had several options to deep copy the member arrays.
    // Taking the approach assuming this structure is *rarely* instantiated, modified,
    // or deleted from the device. Thus allocation on device will be most compatible with
    // the host version.  Several array transfers are required for each deep copy.
    ray_seed_struct *instance = const_cast<ray_seed_struct *>( this ); // bad API const-ness spec.
    // allocation for struct and copy of f0 scalar and array dim[5].
    ray_seed_struct *dev_instance =
        static_cast<ray_seed_struct *>( acc_copyin( instance, sizeof( ray_seed_struct ) ) );

    for ( int ai = 0; ai < 5; ++ai ) {
        copy_in_array_and_set_devptr( this->x[ai], &( dev_instance->x[ai] ), this->dim[ai] );
        copy_in_array_and_set_devptr( this->f[ai], &( dev_instance->f[ai] ), this->dim[ai] );
    }
    return dev_instance;
#elif defined( USE_CUDA )
    return ray_seed_struct_copy_device_cuda( *this );
#elif defined( USE_KOKKOS )
    RAY_ERROR( "Not Finished" );
    return NULL;
#else
    return this; // no action
#endif
}
void RayTrace::ray_seed_struct::free_device(
    const ray_seed_struct *host_seed, const ray_seed_struct *device_seed )
{
    NULL_USE( host_seed );
    if ( device_seed == NULL )
        return;
#if defined( USE_OPENACC ) && defined( _OPENACC )

    // Assuming that the deep dynamic allocations in the host array of structurs have not
    // changed since copy_device() call was made. In other words, the array pointers
    // x and f in the host ray_seed_struct pointers have not been reseated.

    ray_seed_struct *instance =
        const_cast<ray_seed_struct *>( host_seed ); // bad API const-ness spec.

    // delete the member arrays
    for ( int ai = 0; ai < 5; ++ai ) {
        acc_delete( instance->x[ai], sizeof( instance->x[0][0] ) * instance->dim[ai] );
        acc_delete( instance->f[ai], sizeof( instance->f[0][0] ) * instance->dim[ai] );
    }
    // now we can delete the structure
    acc_delete( instance, sizeof( *host_seed ) );


#elif defined( USE_CUDA )
    ray_seed_struct_free_device_cuda( device_seed );
#elif defined( USE_KOKKOS )
    RAY_ERROR( "Not Finished" );
#else
                 // no action
#endif
}


/******************************************************************
* Constructors/Destructors for intensity_step_struct              *
******************************************************************/
RayTrace::intensity_step_struct::intensity_step_struct()
{
    memset( this, 0, sizeof( intensity_step_struct ) );
}
RayTrace::intensity_step_struct::~intensity_step_struct() { delete_data(); }
void RayTrace::intensity_step_struct::initialize(
    int nx_in, int ny_in, int na_in, int nb_in, int nv_in, int N_seed_in )
{
    // Delete exisiting data
    delete_data();
    // Copy the array sizes
    RAY_ASSERT( N_seed_in <= N_SEED_MAX );
    nx     = nx_in;
    ny     = ny_in;
    na     = na_in;
    nb     = nb_in;
    nv     = nv_in;
    N_seed = N_seed_in;
    // Initialize the total image data
    E_v   = new double[nv];
    image = new double[nx * ny];
    E_ang = new double[na * nb];
    W     = new double[nx * ny];
    memset( E_v, 0, nv * sizeof( double ) );
    memset( image, 0, nx * ny * sizeof( double ) );
    memset( E_ang, 0, na * nb * sizeof( double ) );
    memset( W, 0, nx * ny * sizeof( double ) );
    // Allocate space for the seed images
    for ( int s = 0; s < N_seed; s++ ) {
        E_v_seed[s]   = new double[nv];
        image_seed[s] = new double[nx * ny];
        E_ang_seed[s] = new double[na * nb];
        memset( E_v_seed[s], 0, nv * sizeof( double ) );
        memset( image_seed[s], 0, nx * ny * sizeof( double ) );
        memset( E_ang_seed[s], 0, na * nb * sizeof( double ) );
    }
}
void RayTrace::intensity_step_struct::delete_data()
{
    delete[] E_v;
    E_v = NULL;
    delete[] image;
    image = NULL;
    delete[] E_ang;
    E_ang = NULL;
    delete[] W;
    W = NULL;
    for ( int i = 0; i < N_SEED_MAX; i++ ) {
        delete[] E_v_seed[i];
        E_v_seed[i] = NULL;
        delete[] image_seed[i];
        image_seed[i] = NULL;
        delete[] E_ang_seed[i];
        E_ang_seed[i] = NULL;
    }
    memset( this, 0, sizeof( intensity_step_struct ) );
}
void RayTrace::intensity_step_struct::zero()
{
    memset( E_v, 0, nv * sizeof( double ) );
    memset( image, 0, nx * ny * sizeof( double ) );
    memset( E_ang, 0, na * nb * sizeof( double ) );
    memset( W, 0, nx * ny * sizeof( double ) );
    for ( int s = 0; s < N_seed; s++ ) {
        memset( E_v_seed[s], 0, nv * sizeof( double ) );
        memset( image_seed[s], 0, nx * ny * sizeof( double ) );
        memset( E_ang_seed[s], 0, na * nb * sizeof( double ) );
    }
}
void RayTrace::intensity_step_struct::copy( const intensity_step_struct &rhs )
{
    if ( nx != rhs.nx || ny != rhs.ny || na != rhs.na || nb != rhs.nb || nv != rhs.nv ||
         N_seed != rhs.N_seed )
        RAY_ERROR( "Step data is not compatible" );
    memcpy( E_v, rhs.E_v, nv * sizeof( double ) );
    memcpy( image, rhs.image, nx * ny * sizeof( double ) );
    memcpy( E_ang, rhs.E_ang, na * nb * sizeof( double ) );
    memcpy( W, rhs.W, nx * ny * sizeof( double ) );
    for ( int s = 0; s < N_seed; s++ ) {
        memcpy( E_v_seed[s], rhs.E_v_seed[s], nv * sizeof( double ) );
        memcpy( image_seed[s], rhs.image_seed[s], nx * ny * sizeof( double ) );
        memcpy( E_ang_seed[s], rhs.E_ang_seed[s], na * nb * sizeof( double ) );
    }
}
void RayTrace::intensity_step_struct::add( const intensity_step_struct &rhs, bool add_W )
{
    if ( nx != rhs.nx || ny != rhs.ny || na != rhs.na || nb != rhs.nb || nv != rhs.nv ||
         N_seed != rhs.N_seed )
        RAY_ERROR( "Step data is not compatible" );
    for ( int i = 0; i < nv; i++ )
        E_v[i] += rhs.E_v[i];
    for ( int i = 0; i < nx * ny; i++ )
        image[i] += rhs.image[i];
    for ( int i = 0; i < na * nb; i++ )
        E_ang[i] += rhs.E_ang[i];
    for ( int s = 0; s < N_seed; s++ ) {
        for ( int i = 0; i < nv; i++ )
            E_v_seed[s][i] += rhs.E_v_seed[s][i];
        for ( int i = 0; i < nx * ny; i++ )
            image_seed[s][i] += rhs.image_seed[s][i];
        for ( int i = 0; i < na * nb; i++ )
            E_ang_seed[s][i] += rhs.E_ang_seed[s][i];
    }
    if ( add_W ) {
        for ( int i = 0; i < nx * ny; i++ )
            W[i] += rhs.W[i];
    }
}
void RayTrace::intensity_step_struct::sum_reduce( MPI_Comm comm )
{
#ifdef USE_MPI
    if ( MPI_size( comm ) == 1 )
        return;
    if ( nx == 0 )
        RAY_ERROR( "Data was not initialize with 'initialize', so 'sum_reduce' is not available" );
    PROFILE_START( "Sum reduce images" );
    // Allocate memory to use as a temporary buffer
    int Nm       = ( nv + 2 * nx * ny + na * nb ) + ( nv + nx * ny + na * nb ) * N_seed;
    double *mem1 = new double[Nm];
    double *mem2 = new double[Nm];
    // Copy the variables to the buffer
    memcpy( &mem1[0], E_v, nv * sizeof( double ) );
    memcpy( &mem1[nv], image, nx * ny * sizeof( double ) );
    memcpy( &mem1[nv + nx * ny], W, nx * ny * sizeof( double ) );
    memcpy( &mem1[nv + 2 * nx * ny], E_ang, na * nb * sizeof( double ) );
    for ( int s = 0; s < N_seed; s++ ) {
        int Ns = ( nv + 2 * nx * ny + na * nb ) + ( nv + nx * ny + na * nb ) * s;
        memcpy( &mem1[Ns], E_v_seed[s], nv * sizeof( double ) );
        memcpy( &mem1[Ns + nv], image_seed[s], nx * ny * sizeof( double ) );
        memcpy( &mem1[Ns + nv + nx * ny], E_ang_seed[s], na * nb * sizeof( double ) );
    }
    // Perform the communication
    MPI_Barrier( comm );
    MPI_Allreduce( mem1, mem2, Nm, MPI_DOUBLE, MPI_SUM, comm );
    // Copy the variables from the buffer
    memcpy( E_v, &mem2[0], nv * sizeof( double ) );
    memcpy( image, &mem2[nv], nx * ny * sizeof( double ) );
    memcpy( W, &mem2[nv + nx * ny], nx * ny * sizeof( double ) );
    memcpy( E_ang, &mem2[nv + 2 * nx * ny], na * nb * sizeof( double ) );
    for ( int s = 0; s < N_seed; s++ ) {
        int Ns = ( nv + 2 * nx * ny + na * nb ) + ( nv + nx * ny + na * nb ) * s;
        memcpy( E_v_seed[s], &mem2[Ns], nv * sizeof( double ) );
        memcpy( image_seed[s], &mem2[Ns + nv], nx * ny * sizeof( double ) );
        memcpy( E_ang_seed[s], &mem2[Ns + nv + nx * ny], na * nb * sizeof( double ) );
    }
    delete[] mem1;
    delete[] mem2;
    PROFILE_STOP( "Sum reduce images" );
#else
    NULL_USE( comm );
#endif
}
bool RayTrace::intensity_step_struct::valid()
{
    bool neg = false;
    bool nan = false;
    for ( int i = 0; i < nv; i++ ) {
        neg = neg || E_v[i] < 0;
        nan = nan || E_v[i] != E_v[i];
    }
    for ( int i = 0; i < nx * ny; i++ ) {
        neg = neg || image[i] < 0;
        nan = nan || image[i] != image[i];
    }
    for ( int i = 0; i < na * nb; i++ ) {
        neg = neg || E_ang[i] < 0;
        nan = nan || E_ang[i] != E_ang[i];
    }
    for ( int s = 0; s < N_seed; s++ ) {
        for ( int i = 0; i < nv; i++ ) {
            neg = neg || E_v_seed[s][i] < 0;
            nan = nan || E_v_seed[s][i] != E_v_seed[s][i];
        }
        for ( int i = 0; i < nx * ny; i++ ) {
            neg = neg || image_seed[s][i] < 0;
            nan = nan || image_seed[s][i] != image_seed[s][i];
        }
        for ( int i = 0; i < na * nb; i++ ) {
            neg = neg || E_ang_seed[s][i] < 0;
            nan = nan || E_ang_seed[s][i] != E_ang_seed[s][i];
        }
    }
    for ( int i = 0; i < nx * ny; i++ ) {
        neg = neg || W[i] < 0;
        nan = nan || W[i] != W[i];
    }
    return !neg && !nan;
}
void RayTrace::intensity_step_struct::swap( RayTrace::intensity_step_struct &rhs )
{
    std::swap( nx, rhs.nx );
    std::swap( ny, rhs.ny );
    std::swap( na, rhs.na );
    std::swap( nb, rhs.nb );
    std::swap( nv, rhs.nv );
    std::swap( E_v, rhs.E_v );
    std::swap( image, rhs.image );
    std::swap( E_ang, rhs.E_ang );
    std::swap( W, rhs.W );
    std::swap( N_seed, rhs.N_seed );
    for ( int i = 0; i < N_SEED_MAX; i++ ) {
        std::swap( E_v_seed[i], rhs.E_v_seed[i] );
        std::swap( image_seed[i], rhs.image_seed[i] );
        std::swap( E_ang_seed[i], rhs.E_ang_seed[i] );
    }
}


/******************************************************************
* Constructors/Destructors for intensity_struct                   *
******************************************************************/
RayTrace::intensity_struct::intensity_struct()
    : E_v( NULL ),
      image( NULL ),
      E_ang( NULL ),
      E_sum( NULL ),
      I_it( NULL ),
      E_tot( 0 ),
      W( NULL ),
      N_seed( 0 ),
      N( 0 ),
      nx( 0 ),
      ny( 0 ),
      na( 0 ),
      nb( 0 ),
      nv( 0 )
{
    memset( E_v_seed, 0, sizeof( E_v_seed ) );
    memset( image_seed, 0, sizeof( E_v_seed ) );
    memset( E_ang_seed, 0, sizeof( E_v_seed ) );
    memset( E_sum_seed, 0, sizeof( E_v_seed ) );
    memset( I_it_seed, 0, sizeof( E_v_seed ) );
    memset( E_tot_seed, 0, sizeof( E_v_seed ) );
}
RayTrace::intensity_struct::~intensity_struct() { delete_data(); }
#ifdef ENABLE_MOVE_CONSTRUCTOR
RayTrace::intensity_struct::intensity_struct( intensity_struct &&rhs ) : intensity_struct()
{
    swap( rhs );
}
RayTrace::intensity_struct &RayTrace::intensity_struct::operator=( intensity_struct &&rhs )
{
    if ( this == &rhs )
        return *this;
    delete_data();
    swap( rhs );
    return *this;
}
#endif
void RayTrace::intensity_struct::initialize(
    int N_in, int nx_in, int ny_in, int na_in, int nb_in, int nv_in, int N_seed_in )
{
    // Delete exisiting data
    delete_data();
    // Copy the array sizes
    RAY_ASSERT( N_seed_in <= N_SEED_MAX );
    N      = N_in;
    nx     = nx_in;
    ny     = ny_in;
    na     = na_in;
    nb     = nb_in;
    nv     = nv_in;
    N_seed = N_seed_in;
    // Initialize the total image
    E_v   = new double[nv * N];
    image = new double[nx * ny * N];
    E_ang = new double[na * nb * N];
    E_sum = new double[N];
    I_it  = new double[N];
    W     = new double[nx * ny * N];
    memset( E_v, 0, nv * N * sizeof( double ) );
    memset( image, 0, nx * ny * N * sizeof( double ) );
    memset( E_ang, 0, na * nb * N * sizeof( double ) );
    memset( E_sum, 0, N * sizeof( double ) );
    memset( I_it, 0, N * sizeof( double ) );
    memset( W, 0, nx * ny * N * sizeof( double ) );
    E_tot = 0.0;
    // Allocate space for the seed images
    for ( int s = 0; s < N_seed; s++ ) {
        E_v_seed[s]   = new double[nv * N];
        image_seed[s] = new double[nx * ny * N];
        E_ang_seed[s] = new double[na * nb * N];
        E_sum_seed[s] = new double[N];
        I_it_seed[s]  = new double[N];
        memset( E_v_seed[s], 0, nv * N * sizeof( double ) );
        memset( image_seed[s], 0, nx * ny * N * sizeof( double ) );
        memset( E_ang_seed[s], 0, na * nb * N * sizeof( double ) );
        memset( E_sum_seed[s], 0, N * sizeof( double ) );
        memset( I_it_seed[s], 0, N * sizeof( double ) );
        E_tot_seed[s] = 0.0;
    }
}
void RayTrace::intensity_struct::delete_data()
{
    delete[] E_v;
    E_v = NULL;
    delete[] image;
    image = NULL;
    delete[] E_ang;
    E_ang = NULL;
    delete[] E_sum;
    E_sum = NULL;
    delete[] I_it;
    I_it = NULL;
    delete[] W;
    W     = NULL;
    E_tot = 0;
    for ( int i = 0; i < N_seed; i++ ) {
        delete[] E_v_seed[i];
        E_v_seed[i] = NULL;
        delete[] image_seed[i];
        image_seed[i] = NULL;
        delete[] E_ang_seed[i];
        E_ang_seed[i] = NULL;
        delete[] E_sum_seed[i];
        E_sum_seed[i] = NULL;
        delete[] I_it_seed[i];
        I_it_seed[i]  = NULL;
        E_tot_seed[i] = 0;
    }
    N_seed = nx = ny = na = nb = nv = 0;
}
void RayTrace::intensity_struct::zero()
{
    E_tot = 0.0;
    memset( E_v, 0, nv * N * sizeof( double ) );
    memset( image, 0, nx * ny * N * sizeof( double ) );
    memset( E_ang, 0, na * nb * N * sizeof( double ) );
    memset( E_sum, 0, N * sizeof( double ) );
    memset( I_it, 0, N * sizeof( double ) );
    memset( W, 0, nx * ny * N * sizeof( double ) );
    for ( int s = 0; s < N_seed; s++ ) {
        E_tot_seed[s] = 0.0;
        memset( E_v_seed[s], 0, nv * N * sizeof( double ) );
        memset( image_seed[s], 0, nx * ny * N * sizeof( double ) );
        memset( E_ang_seed[s], 0, na * nb * N * sizeof( double ) );
        memset( E_sum_seed[s], 0, N * sizeof( double ) );
        memset( I_it_seed[s], 0, N * sizeof( double ) );
    }
}
void RayTrace::intensity_struct::copy_step(
    int i, const EUV_beam_struct &euv_beam, const intensity_step_struct &I_step )
{
    // Check the inputs
    RAY_ASSERT( nx == I_step.nx && ny == I_step.ny && na == I_step.na && nb == I_step.nb &&
                nv == I_step.nv );
    RAY_ASSERT( nx == euv_beam.nx && na == euv_beam.na && nb == euv_beam.nb && nv == euv_beam.nv );
    if ( euv_beam.y[0] >= 0 )
        RAY_ASSERT( ny == 2 * euv_beam.ny );
    else
        RAY_ASSERT( ny == euv_beam.ny );
    // Copy E_v, image, E_ang, W
    memcpy( &E_v[i * nv], I_step.E_v, nv * sizeof( double ) );
    memcpy( &image[i * nx * ny], I_step.image, nx * ny * sizeof( double ) );
    memcpy( &W[i * nx * ny], I_step.W, nx * ny * sizeof( double ) );
    memcpy( &E_ang[i * na * nb], I_step.E_ang, na * nb * sizeof( double ) );
    for ( int s = 0; s < N_seed; s++ ) {
        memcpy( &E_v_seed[s][i * nv], I_step.E_v_seed[s], nv * sizeof( double ) );
        memcpy( &image_seed[s][i * nx * ny], I_step.image_seed[s], nx * ny * sizeof( double ) );
        memcpy( &E_ang_seed[s][i * na * nb], I_step.E_ang_seed[s], na * nb * sizeof( double ) );
    }
    // Calculate and fill E_sum
    E_sum[i] = 0.0;
    for ( int j = 0; j < nx * ny; j++ )
        E_sum[i] += I_step.image[j];
    I_it[i] = 0.0;
    for ( int s = 0; s < N_seed; s++ ) {
        E_sum_seed[s][i] = 0.0;
        for ( int j = 0; j < nx * ny; j++ )
            E_sum_seed[s][i] += I_step.image_seed[s][j];
        I_it_seed[s][i] = 0.0;
    }
}
bool RayTrace::intensity_struct::operator==( const RayTrace::intensity_struct &rhs ) const
{
    if ( N != rhs.N || nx != rhs.nx || ny != rhs.ny || na != rhs.na || nb != rhs.nb ||
         nv != rhs.nv || N_seed != rhs.N_seed )
        return false;
    bool equal = true;
    equal      = equal && approx_equal( E_tot, rhs.E_tot );
    equal      = equal && approx_equal( N * nv, E_v, rhs.E_v );
    equal      = equal && approx_equal( N * nx * ny, image, rhs.image );
    equal      = equal && approx_equal( N * na * nb, E_ang, rhs.E_ang );
    equal      = equal && approx_equal( N, E_sum, rhs.E_sum );
    equal      = equal && approx_equal( N, I_it, rhs.I_it );
    equal      = equal && approx_equal( N * nx * ny, W, rhs.W );
    for ( int s = 0; s < N_seed; s++ ) {
        equal = equal && approx_equal( E_tot_seed[s], rhs.E_tot_seed[s] );
        equal = equal && approx_equal( N * nv, E_v_seed[s], rhs.E_v_seed[s] );
        equal = equal && approx_equal( N * nx * ny, image_seed[s], rhs.image_seed[s] );
        equal = equal && approx_equal( N * na * nb, E_ang_seed[s], rhs.E_ang_seed[s] );
        equal = equal && approx_equal( N, E_sum_seed[s], rhs.E_sum_seed[s] );
        equal = equal && approx_equal( N, I_it_seed[s], rhs.I_it_seed[s] );
    }
    return equal;
}
void RayTrace::intensity_struct::swap( RayTrace::intensity_struct &rhs )
{
    std::swap( nx, rhs.nx );
    std::swap( ny, rhs.ny );
    std::swap( na, rhs.na );
    std::swap( nb, rhs.nb );
    std::swap( nv, rhs.nv );
    std::swap( E_v, rhs.E_v );
    std::swap( image, rhs.image );
    std::swap( E_ang, rhs.E_ang );
    std::swap( E_sum, rhs.E_sum );
    std::swap( I_it, rhs.I_it );
    std::swap( E_tot, rhs.E_tot );
    std::swap( W, rhs.W );
    std::swap( N_seed, rhs.N_seed );
    for ( int i = 0; i < N_SEED_MAX; i++ ) {
        std::swap( E_v_seed[i], rhs.E_v_seed[i] );
        std::swap( image_seed[i], rhs.image_seed[i] );
        std::swap( E_ang_seed[i], rhs.E_ang_seed[i] );
        std::swap( E_sum_seed[i], rhs.E_sum_seed[i] );
        std::swap( I_it_seed[i], rhs.I_it_seed[i] );
        std::swap( E_tot_seed[i], rhs.E_tot_seed[i] );
    }
}


/**********************************************************************
* Constructors/Destructors for ray_gain_struct                        *
**********************************************************************/
RayTrace::ray_gain_struct::ray_gain_struct() { memset( this, 0, sizeof( ray_gain_struct ) ); }
void RayTrace::ray_gain_struct::initialize( int Nx_in, int Ny_in, int Nv_in, bool use_emiss )
{
    Nx  = Nx_in;
    Ny  = Ny_in;
    Nv  = Nv_in;
    x   = new double[Nx];
    y   = new double[Ny];
    n   = new double[Nx * Ny];
    g0  = new float[Nx * Ny];
    gv  = new float[Nx * Ny * Nv];
    gv0 = new float[Nx * Ny];
    memset( x, 0, Nx * sizeof( double ) );
    memset( y, 0, Ny * sizeof( double ) );
    memset( n, 0, Nx * Ny * sizeof( double ) );
    memset( g0, 0, Nx * Ny * sizeof( float ) );
    memset( gv, 0, Nx * Ny * Nv * sizeof( float ) );
    memset( gv0, 0, Nx * Ny * sizeof( float ) );
    if ( use_emiss ) {
        E0 = new float[Nx * Ny];
        memset( E0, 0, Nx * Ny * sizeof( float ) );
    }
}
RayTrace::ray_gain_struct::~ray_gain_struct()
{
    delete[] x;
    x = NULL;
    delete[] y;
    y = NULL;
    delete[] n;
    n = NULL;
    delete[] g0;
    g0 = NULL;
    delete[] E0;
    E0 = NULL;
    delete[] gv;
    gv = NULL;
    delete[] gv0;
    gv0 = NULL;
}
void RayTrace::ray_gain_struct::writeData( FILE *fid, const char *prefix ) const
{
    char tmp[20];
    sprintf( tmp, "%sNx", prefix );
    file_utilities::write_scalar_int( fid, tmp, Nx );
    sprintf( tmp, "%sNy", prefix );
    file_utilities::write_scalar_int( fid, tmp, Ny );
    sprintf( tmp, "%sNv", prefix );
    file_utilities::write_scalar_int( fid, tmp, Nv );
    sprintf( tmp, "%sx", prefix );
    file_utilities::write_variable_double( fid, tmp, Nx, x, false );
    sprintf( tmp, "%sy", prefix );
    file_utilities::write_variable_double( fid, tmp, Ny, y, false );
    sprintf( tmp, "%sn", prefix );
    file_utilities::write_variable_double( fid, tmp, Nx * Ny, n, false );
    sprintf( tmp, "%sg0", prefix );
    file_utilities::write_variable_float( fid, tmp, Nx * Ny, g0, false );
    if ( E0 != NULL ) {
        sprintf( tmp, "%sE0", prefix );
        file_utilities::write_variable_float( fid, tmp, Nx * Ny, E0, false );
    }
    sprintf( tmp, "%sgv", prefix );
    file_utilities::write_variable_float( fid, tmp, Nx * Ny * Nv, gv, false );
    sprintf( tmp, "%sgv0", prefix );
    file_utilities::write_variable_float( fid, tmp, Nx * Ny, gv0, false );
}
std::pair<char *, size_t> RayTrace::ray_gain_struct::pack( int ) const
{
    // Estimate the size required
    size_t N_bytes = 3 * sizeof( int );
    N_bytes += ( Nx + Ny ) * sizeof( double ); // Space for x and y
    N_bytes += Nx * Ny * sizeof( double );     // Space for n
    N_bytes += 3 * Nx * Ny * sizeof( float );  // Space for g0, E0, gv0
    N_bytes += Nx * Ny * Nv * sizeof( float ); // Space for gv
    // Create and pack the buffer
    char *data = new char[N_bytes];
    size_t pos = 0;
    pack_buffer<int>( Nx, pos, data );
    pack_buffer<int>( Ny, pos, data );
    pack_buffer<int>( Nv, pos, data );
    memcpy( &data[pos], x, Nx * sizeof( double ) );
    pos += Nx * sizeof( double );
    memcpy( &data[pos], y, Ny * sizeof( double ) );
    pos += Ny * sizeof( double );
    memcpy( &data[pos], n, Nx * Ny * sizeof( double ) );
    pos += Nx * Ny * sizeof( double );
    memcpy( &data[pos], g0, Nx * Ny * sizeof( float ) );
    pos += Nx * Ny * sizeof( float );
    memcpy( &data[pos], E0, Nx * Ny * sizeof( float ) );
    pos += Nx * Ny * sizeof( float );
    memcpy( &data[pos], gv, Nx * Ny * Nv * sizeof( float ) );
    pos += Nx * Ny * Nv * sizeof( float );
    memcpy( &data[pos], gv0, Nx * Ny * sizeof( float ) );
    pos += Nx * Ny * sizeof( float );
    RAY_ASSERT( pos == N_bytes );
    return std::pair<char *, size_t>( data, N_bytes );
}
void RayTrace::ray_gain_struct::unpack( std::pair<const char *, size_t> data_in )
{
    const char *data     = data_in.first;
    const size_t N_bytes = data_in.second;
    size_t pos           = 0;
    Nx                   = unpack_buffer<int>( pos, data );
    Ny                   = unpack_buffer<int>( pos, data );
    Nv                   = unpack_buffer<int>( pos, data );
    x                    = new double[Nx];
    memcpy( x, &data[pos], Nx * sizeof( double ) );
    pos += Nx * sizeof( double );
    y = new double[Ny];
    memcpy( y, &data[pos], Ny * sizeof( double ) );
    pos += Ny * sizeof( double );
    n = new double[Nx * Ny];
    memcpy( n, &data[pos], Nx * Ny * sizeof( double ) );
    pos += Nx * Ny * sizeof( double );
    g0 = new float[Nx * Ny];
    memcpy( g0, &data[pos], Nx * Ny * sizeof( float ) );
    pos += Nx * Ny * sizeof( float );
    E0 = new float[Nx * Ny];
    memcpy( E0, &data[pos], Nx * Ny * sizeof( float ) );
    pos += Nx * Ny * sizeof( float );
    gv = new float[Nx * Ny * Nv];
    memcpy( gv, &data[pos], Nx * Ny * Nv * sizeof( float ) );
    pos += Nx * Ny * Nv * sizeof( float );
    gv0 = new float[Nx * Ny];
    memcpy( gv0, &data[pos], Nx * Ny * sizeof( float ) );
    pos += Nx * Ny * sizeof( float );
    RAY_ASSERT( pos == N_bytes );
}
const RayTrace::ray_gain_struct *RayTrace::ray_gain_struct::copy_device(
    size_t N, const ray_gain_struct *arr )
{
#if defined( USE_CUDA )
    return ray_gain_struct_copy_device_cuda( N, arr );
#elif defined( USE_OPENACC ) && defined( _OPENACC )
    ray_gain_struct *instances = const_cast<ray_gain_struct *>( arr ); // bad API const-ness spec.
    ray_gain_struct *dev_instances =
        static_cast<ray_gain_struct *>( acc_copyin( instances, N * sizeof( ray_gain_struct ) ) );
    // Had several options to deep copy the member arrays.
    // Taking the approach assuming this structure is *rarely* instantiated, modified,
    // or deleted from the device. Thus allocation on device will be most compatible with
    // the host version.  Several small transfers required for deep copy
    for ( int i = 0; i < N; i++ ) {
        const int &Nx = arr[i].Nx;
        const int &Ny = arr[i].Ny;
        const int &Nv = arr[i].Nv;
        // 'deep copy' member arrays, updating the structure already on the device.
        copy_in_array_and_set_devptr( instances[i].x, &( dev_instances[i].x ), Nx );
        copy_in_array_and_set_devptr( instances[i].y, &( dev_instances[i].y ), Ny );
        copy_in_array_and_set_devptr( instances[i].n, &( dev_instances[i].n ), Nx * Ny );
        copy_in_array_and_set_devptr( instances[i].g0, &( dev_instances[i].g0 ), Nx * Ny );
        copy_in_array_and_set_devptr( instances[i].E0, &( dev_instances[i].E0 ), Nx * Ny );
        copy_in_array_and_set_devptr( instances[i].gv0, &( dev_instances[i].gv0 ), Nx * Ny );
        copy_in_array_and_set_devptr( instances[i].gv, &( dev_instances[i].gv ), Nx * Ny * Nv );
    }
    return dev_instances;
#elif defined( USE_KOKKOS )
    RAY_ERROR( "Not Finished" );
    return NULL;
#else
    NULL_USE( N );
    NULL_USE( arr );
    RAY_ERROR( "No device detected" );
    return arr;
#endif
}
void RayTrace::ray_gain_struct::free_device(
    size_t N, const ray_gain_struct *host_arr, const ray_gain_struct *device_arr )
{
    if ( device_arr == NULL )
        return;
#if defined( USE_CUDA )
    NULL_USE( host_arr );
    ray_gain_struct_free_device_cuda( N, device_arr );
#elif defined( USE_OPENACC ) && defined( _OPENACC )

    ray_gain_struct *instances =
        const_cast<ray_gain_struct *>( host_arr ); // bad API const-ness spec.

    // Options:
    // acc_free(device pointer) we would have to copy the struct member array pointers
    // back to the host before each call, since they were not stored on the host.
    //
    // acc_delete(host pointer, size ) we have all the host pointers, but will have to assume the
    // host structures were not modified between copy_device() and now. The fact that the API
    // requires a size argument to this function seems silly. It would not make sense
    // to partially free or overly free. The spec isn't even explicit that the argument
    // is in unites of Bytes.


    // Assuming that the deep dynamic allocations in the host array of structurs have not
    // changed since copy_device() call was made. In other words, the array pointers
    // x,y,n,etc. in the host ray_gain_struct pointers have not been reseated.
    for ( int i = 0; i < N; i++ ) {
        const int &Nx = host_arr[i].Nx;
        const int &Ny = host_arr[i].Ny;
        const int &Nv = host_arr[i].Nv;

        // delete the member arrays
        acc_delete( instances[i].x, sizeof( instances->x[0] ) * Nx );
        acc_delete( instances[i].y, sizeof( instances->y[0] ) * Ny );
        acc_delete( instances[i].n, sizeof( instances->n[0] ) * Nx * Ny );
        acc_delete( instances[i].g0, sizeof( instances->g0[0] ) * Nx * Ny );
        acc_delete( instances[i].E0, sizeof( instances->E0[0] ) * Nx * Ny );
        acc_delete( instances[i].gv0, sizeof( instances->gv0[0] ) * Nx * Ny );
        acc_delete( instances[i].gv, sizeof( instances->gv[0] ) * Nx * Ny * Nv );
    }
    // now we can delete the array of structures
    acc_delete( instances, N * sizeof( ray_gain_struct ) );
#elif defined( USE_KOKKOS )
    RAY_ERROR( "Not Finished" );
#else
    NULL_USE( N );
    NULL_USE( host_arr );
    RAY_ERROR( "No device detected" );
#endif
}


/**********************************************************************
* create_image_struct                                                 *
**********************************************************************/
RayTrace::create_image_struct::create_image_struct()
{
    N          = 0;
    N_start    = 0;
    N_parallel = 1;
    dz         = 0.0;
    euv_beam   = NULL;
    seed_beam  = NULL;
    gain       = NULL;
    seed       = NULL;
    image      = NULL;
    I_ang      = NULL;
}
RayTrace::create_image_struct::~create_image_struct()
{
    free( image );
    free( I_ang );
}
std::pair<char *, size_t> RayTrace::create_image_struct::pack( int compression ) const
{
    // Pack the sub structures
    std::pair<char *, size_t> euv_beam_data = euv_beam->pack( compression );
    std::pair<char *, size_t> seed_beam_data( NULL, 0 );
    std::pair<char *, size_t> *gain_data = new std::pair<char *, size_t>[N];
    for ( int i      = 0; i < N; i++ )
        gain_data[i] = gain[i].pack();
    std::pair<char *, size_t> seed_data( NULL, 0 );
    if ( seed_beam != NULL )
        seed_beam_data = seed_beam->pack( compression );
    if ( seed != NULL )
        seed_data = seed->pack( compression );
    // Estimate the total size required
    size_t N_bytes = 3 * sizeof( int ) + sizeof( double );
    N_bytes += sizeof( unsigned int ) + euv_beam_data.second;
    N_bytes += sizeof( unsigned int ) + seed_beam_data.second;
    N_bytes += sizeof( unsigned int ) + seed_data.second;
    for ( int i = 0; i < N; i++ )
        N_bytes += sizeof( unsigned int ) + gain_data[i].second;
    N_bytes += 2 * sizeof( bool );
    if ( image != NULL )
        N_bytes += euv_beam->nx * euv_beam->ny * euv_beam->nv * sizeof( double );
    if ( I_ang != NULL )
        N_bytes += euv_beam->na * euv_beam->nb * sizeof( double );
    // Copy basic info to the buffer
    char *data = new char[N_bytes];
    size_t pos = 0;
    pack_buffer<int>( N, pos, data );
    pack_buffer<int>( N_start, pos, data );
    pack_buffer<int>( N_parallel, pos, data );
    pack_buffer<double>( dz, pos, data );
    // Copy euv_beam to the buffer
    pack_buffer<unsigned int>( (unsigned int) euv_beam_data.second, pos, data );
    memcpy( &data[pos], euv_beam_data.first, euv_beam_data.second );
    pos += euv_beam_data.second;
    // Copy seed_beam to the buffer
    pack_buffer<unsigned int>( (unsigned int) seed_beam_data.second, pos, data );
    memcpy( &data[pos], seed_beam_data.first, seed_beam_data.second );
    pos += seed_beam_data.second;
    // Copy gain to the buffer
    for ( int i = 0; i < N; i++ ) {
        pack_buffer<unsigned int>( (unsigned int) gain_data[i].second, pos, data );
        memcpy( &data[pos], gain_data[i].first, gain_data[i].second );
        pos += gain_data[i].second;
    }
    // Copy seed to the buffer
    pack_buffer<unsigned int>( (unsigned int) seed_data.second, pos, data );
    memcpy( &data[pos], seed_data.first, seed_data.second );
    pos += seed_data.second;
    // Copy image/I_ang to the buffer
    pack_buffer<bool>( image != NULL, pos, data );
    if ( image != NULL ) {
        memcpy( &data[pos], image, euv_beam->nx * euv_beam->ny * euv_beam->nv * sizeof( double ) );
        pos += euv_beam->nx * euv_beam->ny * euv_beam->nv * sizeof( double );
    }
    pack_buffer<bool>( I_ang != NULL, pos, data );
    if ( I_ang != NULL ) {
        memcpy( &data[pos], I_ang, euv_beam->na * euv_beam->nb * sizeof( double ) );
        pos += euv_beam->na * euv_beam->nb * sizeof( double );
    }
    // Finished
    RAY_ASSERT( pos == N_bytes );
    return std::pair<char *, size_t>( data, N_bytes );
}
void RayTrace::create_image_struct::unpack( std::pair<const char *, size_t> data_in )
{
    const char *data     = data_in.first;
    const size_t N_bytes = data_in.second;
    size_t pos           = 0;
    free( image );
    image = NULL;
    free( I_ang );
    I_ang = NULL;
    // Copy basic info from the buffer
    N          = unpack_buffer<int>( pos, data );
    N_start    = unpack_buffer<int>( pos, data );
    N_parallel = unpack_buffer<int>( pos, data );
    dz         = unpack_buffer<double>( pos, data );
    // Copy euv_beam from the buffer
    unsigned int N_bytes_tmp = unpack_buffer<unsigned int>( pos, data );
    if ( N_bytes_tmp > 0 ) {
        EUV_beam_struct *tmp = new EUV_beam_struct;
        tmp->unpack( std::pair<const char *, size_t>( &data[pos], N_bytes_tmp ) );
        euv_beam = tmp;
        pos += N_bytes_tmp;
    } else {
        euv_beam = NULL;
    }
    // Copy seed_beam from the buffer
    N_bytes_tmp = unpack_buffer<unsigned int>( pos, data );
    if ( N_bytes_tmp > 0 ) {
        seed_beam_struct *tmp = new seed_beam_struct;
        tmp->unpack( std::pair<const char *, size_t>( &data[pos], N_bytes_tmp ) );
        seed_beam = tmp;
        pos += N_bytes_tmp;
    } else {
        seed_beam = NULL;
    }
    // Copy gain from the buffer
    ray_gain_struct *gain2 = new ray_gain_struct[N];
    gain                   = gain2;
    for ( int i = 0; i < N; i++ ) {
        N_bytes_tmp = unpack_buffer<unsigned int>( pos, data );
        gain2[i].unpack( std::pair<const char *, size_t>( &data[pos], N_bytes_tmp ) );
        pos += N_bytes_tmp;
    }
    // Copy seed from the buffer
    N_bytes_tmp = unpack_buffer<unsigned int>( pos, data );
    if ( N_bytes_tmp > 0 ) {
        ray_seed_struct *tmp = new ray_seed_struct;
        tmp->unpack( std::pair<const char *, size_t>( &data[pos], N_bytes_tmp ) );
        seed = tmp;
        pos += N_bytes_tmp;
    } else {
        seed = NULL;
    }
    // Copy image/I_ang from the buffer
    bool copy_image = unpack_buffer<bool>( pos, data );
    if ( copy_image ) {
        image = (double *) malloc( euv_beam->nx * euv_beam->ny * euv_beam->nv * sizeof( double ) );
        memcpy( image, &data[pos], euv_beam->nx * euv_beam->ny * euv_beam->nv * sizeof( double ) );
        pos += euv_beam->nx * euv_beam->ny * euv_beam->nv * sizeof( double );
    }
    bool copy_I_ang = unpack_buffer<bool>( pos, data );
    if ( copy_I_ang ) {
        I_ang = (double *) malloc( euv_beam->na * euv_beam->nb * sizeof( double ) );
        memcpy( I_ang, &data[pos], euv_beam->na * euv_beam->nb * sizeof( double ) );
        pos += euv_beam->na * euv_beam->nb * sizeof( double );
    }
    // Finished
    RAY_ASSERT( pos == N_bytes );
}


