#ifndef INCLUDED_MPI_FUNCTIONS
#define INCLUDED_MPI_FUNCTIONS

#ifdef USE_MPI
#include "mpi.h"
#else
typedef int MPI_Comm;
#define MPI_COMM_WORLD 1
#endif


/**********************************************************************
* Some simple MPI helpers                                             *
**********************************************************************/
#ifdef USE_MPI
inline int MPI_rank( MPI_Comm comm )
{
    int rank = 0;
    int initialized;
    MPI_Initialized( &initialized );
    if ( initialized != 0 )
        MPI_Comm_rank( comm, &rank );
    return rank;
}
inline int MPI_size( MPI_Comm comm )
{
    int size = 1;
    int initialized;
    MPI_Initialized( &initialized );
    if ( initialized != 0 )
        MPI_Comm_size( comm, &size );
    return size;
}
#else
inline int MPI_rank( MPI_Comm ) { return 0; }
inline int MPI_size( MPI_Comm ) { return 1; }
#endif


/**********************************************************************
* If the error is 0 for all processors, then 0 will be returned.      *
* If there is 1 (or more) processors that contain a non-zero error    *
* code, then the first processor with a non-zero entry will be        *
* returned across all processors.                                     *
**********************************************************************/
#ifdef USE_MPI
inline int GATHER_ERROR( int error, MPI_Comm comm )
{
    int size = MPI_size( comm );
    if ( size == 1 )
        return error;
#if 1
    // Just call MPI abort (this removes the Allreduce call, but prevents complete error handling.
    if ( error != 0 )
        MPI_Abort( comm, error );
#else
    // Syncronize the error code for proper error handling
    int tmp1 = 0;
    int tmp2 = 0;
    if ( error != 0 )
        tmp1 = 1;
    MPI_Allreduce( &tmp1, &tmp2, 1, MPI_INT, MPI_MAX, comm );
    if ( tmp2 != 0 ) {
        int *error2 = new int[size];
        int error1  = MPI_Allgather( &error, 1, MPI_INT, error2, 1, MPI_INT, comm );
        error       = 0;
        for ( int i = 0; i < size; i++ ) {
            if ( error2[i] != 0 ) {
                error = error2[i];
                break;
            }
        }
        delete[] error2;
    }
#endif
    return error;
}
#else
inline int GATHER_ERROR( int error, MPI_Comm )
{
    return error;
}
#endif


#endif
