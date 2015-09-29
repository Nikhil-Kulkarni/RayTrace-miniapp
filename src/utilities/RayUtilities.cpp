#include "RayUtilities.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <fstream>
#include <string.h>
#include <signal.h>

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

#ifdef USE_MPI
    #include "mpi.h"
#endif

// Detect the OS and include system dependent headers
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) || defined(_MSC_VER)
    // Note: windows has not been testeds
    #define USE_WINDOWS
    #include <windows.h>
    #include <process.h>
    #include <stdio.h>
    #include <tchar.h>
    #include <psapi.h>
    #include <DbgHelp.h>
    #define mkdir(path, mode) _mkdir(path)
    //#pragma comment(lib, psapi.lib) //added
    //#pragma comment(linker, /DEFAULTLIB:psapi.lib)
#elif defined(__APPLE__)
    #define USE_MAC
    #include <sys/time.h>
    #include <signal.h>
    #include <execinfo.h>
    #include <dlfcn.h>
    #include <mach/mach.h>
#elif defined(__linux) || defined(__unix) || defined(__posix)
    #define USE_LINUX
    #include <sys/time.h>
    #include <execinfo.h>
    #include <dlfcn.h>
    #include <malloc.h>
#else
    #error Unknown OS
#endif


// Set the reference to pout, perr, plog
std::ostream& pout = AtomicModel::pout;
std::ostream& perr = AtomicModel::perr;
std::ostream& plog = AtomicModel::plog;




// Functions to compress a bool array
template<>
size_t Utilities::compress_array<bool>( size_t N, const bool data[], int method, unsigned char *cdata[])
{
    size_t N_bytes = 0;
    if ( method==0 ) {
        N_bytes = N;
        *cdata = new unsigned char[N_bytes];
        memcpy(*cdata,data,N_bytes);
    } else {
        N_bytes = (N+7)/8;
        *cdata = new unsigned char[N_bytes];
        for (size_t i=0; i<N_bytes; i++)
            (*cdata)[i] = 0;
        for (size_t i=0; i<N; i++) {
            if ( data[i] ) {
                unsigned char mask = ((unsigned char)1)<<(i%8);
                (*cdata)[i/8] = (*cdata)[i/8] | mask;
            }
        }
    }
    return N_bytes;
}
template<>
void Utilities::decompress_array<bool>( size_t N, size_t N_bytes, const unsigned char cdata[], int method, bool *data[] )
{
    *data = new bool[N];
    if ( method==0 ) {
        RAY_ASSERT(N_bytes==N);
        memcpy(*data,cdata,N);
    } else {
        RAY_ASSERT(N_bytes==(N+7)/8);
        for (size_t i=0; i<N; i++) {
            unsigned char mask = 1<<(i%8);
            unsigned char test = mask & cdata[i/8];
            (*data)[i] = (test!=0);
        }
    }
}

