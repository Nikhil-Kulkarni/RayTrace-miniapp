#include "RayUtilities.h"

#include <fstream>
#include <iostream>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string.h>

// Atomic model utilities
#ifdef USE_ATOMIC_MODEL
#include "utilities/Utilities.h"
#else
namespace AtomicModel {
std::ostream &pout = std::cout;
std::ostream &plog = std::cout;
std::ostream &perr = std::cerr;
};
#endif


// Set the reference to pout, perr, plog
std::ostream &pout = AtomicModel::pout;
std::ostream &perr = AtomicModel::perr;
std::ostream &plog = AtomicModel::plog;


// Functions to compress a bool array
template <>
size_t Utilities::compress_array<bool>(
    size_t N, const bool data[], int method, unsigned char *cdata[] )
{
    size_t N_bytes = 0;
    if ( method == 0 ) {
        N_bytes = N;
        *cdata  = new unsigned char[N_bytes];
        memcpy( *cdata, data, N_bytes );
    } else {
        N_bytes = ( N + 7 ) / 8;
        *cdata  = new unsigned char[N_bytes];
        for ( size_t i    = 0; i < N_bytes; i++ )
            ( *cdata )[i] = 0;
        for ( size_t i = 0; i < N; i++ ) {
            if ( data[i] ) {
                unsigned char mask = ( (unsigned char) 1 ) << ( i % 8 );
                ( *cdata )[i / 8]  = ( *cdata )[i / 8] | mask;
            }
        }
    }
    return N_bytes;
}
template <>
void Utilities::decompress_array<bool>(
    size_t N, size_t N_bytes, const unsigned char cdata[], int method, bool *data[] )
{
    *data = new bool[N];
    if ( method == 0 ) {
        RAY_ASSERT( N_bytes == N );
        memcpy( *data, cdata, N );
    } else {
        RAY_ASSERT( N_bytes == ( N + 7 ) / 8 );
        for ( size_t i = 0; i < N; i++ ) {
            unsigned char mask = 1 << ( i % 8 );
            unsigned char test = mask & cdata[i / 8];
            ( *data )[i]       = ( test != 0 );
        }
    }
}
