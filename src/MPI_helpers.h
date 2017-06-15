// Helper file to define some helper functions using MPI
#include <vector>

#ifdef USE_MPI

#include <mpi.h>

// MPI wrappers
inline void startup( int argc, char *argv[] ) {
    MPI_Init( &argc, &argv );
}
inline void shutdown( ) {
    MPI_Barrier( MPI_COMM_WORLD );
    MPI_Finalize();
}
inline void barrier() {
    MPI_Barrier( MPI_COMM_WORLD );
}
inline int rank() {
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    return rank;
}
inline int size() {
    int size;
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    return size;
}
inline int sumReduce( const int val ) {
    int result;
    MPI_Allreduce( &val, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    return result;
}
inline std::vector<double> gatherAll( const std::vector<double>& x ) {
    std::vector<double> y( x.size()*size(), 0 );
    MPI_Allgather( x.data(), x.size(), MPI_DOUBLE, y.data(), x.size(), MPI_DOUBLE, MPI_COMM_WORLD );
    return y;
}


#else

// No MPI
inline void startup( int, char*[] ) {}
inline void shutdown( ) {}
inline void barrier() {}
inline int  rank() { return 0; }
inline int  size() { return 1; }
inline int  sumReduce( const int val ) { return val; }
inline std::vector<double> gatherAll( const std::vector<double>& x ) { return x; }

#endif
