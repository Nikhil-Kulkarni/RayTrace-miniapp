#ifndef included_RayUtilities
#define included_RayUtilities

#include <cstring>
#include <iostream>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>


namespace Utilities {


//! Subroutine to compress an array
/*!
 * This function compresses an array by removing the zero-values entries.
 * This function returns the number of bytes in the compressed array.
 * @param N         Number of values in the array
 * @param data      Input array
 * @param method    Compression method to use:
 *                      0: Store raw data
 *                      1: Remove zeros
 *                      2: Convert to single precision and remove zeros
 * @param cdata     Compressed data array (allocated by function)
 */
template <class TYPE>
size_t compress_array( size_t N, const TYPE data[], int compression, unsigned char *cdata[] );


//! Subroutine to de-compress an array
/*!
 * This function de-compresses an array created by removing the zero-values entries.
 * @param N         Number of values in the array
 * @param N_bytes   Number of bytes in the incoming stream
 * @param cdata     Input compressed array
 * @param method    Compression method used:
 *                      0: Store raw data
 *                      1: Remove zeros
 *                      2: Convert to single precision and remove zeros
 * @param data      Output data array (allocated by function)
 */
template <class TYPE>
void decompress_array(
    size_t N, size_t N_bytes, const unsigned char cdata[], int compression, TYPE *data[] );

// Version of sprintf that returns a std::string
inline std::string stringf( const char *format, ... );

// Spcializations for bool
template <>
size_t compress_array<bool>( size_t N, const bool data[], int, unsigned char *cdata[] );
template <>
void decompress_array<bool>(
    size_t N, size_t N_bytes, const unsigned char cdata[], int, bool *data[] );
};


/****************************************************************************
* pout, plog, perr, printp                                                  *
****************************************************************************/
extern std::ostream &pout;
extern std::ostream &perr;
extern std::ostream &plog;
inline int printp( const char *format, ... )
{
    va_list ap;
    va_start( ap, format );
    char tmp[1024];
    int n = vsprintf( tmp, format, ap );
    va_end( ap );
    pout << tmp;
    pout.flush();
    return n;
}
inline std::string Utilities::stringf( const char *format, ... )
{
    va_list ap;
    va_start( ap, format );
    char tmp[1024];
    vsprintf( tmp, format, ap );
    va_end( ap );
    return std::string( tmp );
}


#include "utilities/RayUtilityMacros.h"


/****************************************************************************
* Functions to compress/decompres an array without the zero-valued entries  *
****************************************************************************/
template <class TYPE>
size_t Utilities::compress_array( size_t N, const TYPE data[], int method, unsigned char *cdata[] )
{
    size_t N_bytes = 0;
    if ( method == 0 ) {
        N_bytes = N * sizeof( TYPE );
        *cdata  = new unsigned char[N_bytes];
        memcpy( *cdata, data, N_bytes );
    } else if ( method == 1 ) {
        size_t N_zeros = 0;
        for ( size_t i = 0; i < N; i++ ) {
            if ( data[i] == 0 )
                N_zeros++;
        }
        // Determine the optimum storage type:
        if ( N_zeros == N ) {
            // Special case where everything is zero
            *cdata    = new unsigned char[1];
            *cdata[0] = 7;
            N_bytes   = 1;
        } else if ( ( N - N_zeros ) * sizeof( TYPE ) + ( N + 7 ) / 8 >= N * sizeof( TYPE ) ) {
            // We are better off storing the data as a dense array
            N_bytes     = N * sizeof( TYPE );
            *cdata      = new unsigned char[N_bytes];
            TYPE *data2 = (TYPE *) *cdata;
            for ( size_t i = 0; i < N; i++ )
                data2[i]   = data[i];
        } else {
            // Store the data as a bit array indicating if each value is non-zero, and then a dense
            // array of non-zeroed values
            N_bytes = ( N + 7 ) / 8 + ( N - N_zeros ) * sizeof( TYPE );
            *cdata  = new unsigned char[N_bytes];
            for ( size_t i    = 0; i < N_bytes; i++ )
                ( *cdata )[i] = 0;
            // Create a byte array to store which values are zero
            for ( size_t i = 0; i < N; i++ ) {
                if ( data[i] == 0 )
                    continue;
                unsigned char mask = 1 << ( i % 8 );
                ( *cdata )[i / 8]  = ( *cdata )[i / 8] | mask;
            }
            // Store the non-zero values
            TYPE *data2 = (TYPE *) &( *cdata )[( N + 7 ) / 8];
            size_t j    = 0;
            for ( size_t i = 0; i < N; i++ ) {
                if ( data[i] != 0 ) {
                    data2[j] = data[i];
                    j++;
                }
            }
        }
    } else if ( method == 2 ) {
        // Convert to single precision then compress array to remove zeros
        float *tmp = new float[N];
        for ( size_t i = 0; i < N; i++ )
            tmp[i]     = static_cast<float>( data[i] );
        N_bytes        = Utilities::compress_array<float>( N, tmp, 1, cdata );
        delete[] tmp;
    } else {
        RAY_ERROR( "Unknown compression method" );
    }
    return N_bytes;
}
template <class TYPE>
void Utilities::decompress_array(
    size_t N, size_t N_bytes_in, const unsigned char cdata[], int method, TYPE *data[] )
{
    *data = new TYPE[N];
    memset( *data, 0, N * sizeof( TYPE ) );
    size_t N_bytes = 0;
    if ( method == 0 ) {
        N_bytes = N * sizeof( TYPE );
        RAY_ASSERT( N_bytes == N_bytes_in );
        memcpy( *data, cdata, N_bytes );
    } else if ( method == 1 ) {
        for ( size_t i   = 0; i < N; i++ )
            ( *data )[i] = 0;
        if ( N_bytes_in == 0 ) {
            // NULL array
            return;
        } else if ( N_bytes_in == 1 ) {
            // Empty array
            return;
        } else if ( N_bytes_in == N * sizeof( TYPE ) ) {
            // We are storing the dense array
            const TYPE *data2 = reinterpret_cast<const TYPE *>( cdata );
            for ( size_t i   = 0; i < N; i++ )
                ( *data )[i] = data2[i];
        } else {
            // We are storing a bit array indicating the non-zeroed entries, and the non-zero values
            // as a dense array
            const TYPE *data2 = reinterpret_cast<const TYPE *>( &cdata[( N + 7 ) / 8] );
            int j             = 0;
            for ( unsigned long int i = 0; i < N; i++ ) {
                unsigned char mask = 1 << ( i % 8 );
                unsigned char test = mask & cdata[i / 8];
                if ( test != 0 ) {
                    ( *data )[i] = data2[j];
                    j++;
                } else {
                    ( *data )[i] = 0.0;
                }
            }
        }
    } else if ( method == 2 ) {
        // Convert to single precision then compress array to remove zeros
        float *tmp = NULL;
        Utilities::decompress_array<float>( N, N_bytes_in, cdata, 1, &tmp );
        for ( size_t i   = 0; i < N; i++ )
            ( *data )[i] = tmp[i];
        delete[] tmp;
    } else {
        RAY_ERROR( "Unknown compression method" );
    }
}


#endif
