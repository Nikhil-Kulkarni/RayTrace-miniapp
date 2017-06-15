#ifndef CreateImageHelpers
#define CreateImageHelpers

#include <cstring>
#include <string>
#include <sstream>
#include <stdint.h>
#include <iostream>

#include "RayTrace.h"


// Function to call fread, checking that the proper length was read
void fread2( void *ptr, size_t size, size_t count, FILE *fid );


// Scale the problem size
void scale_problem( RayTrace::create_image_struct &info, double scale );


// Sleep for XX ms
void sleep_ms( int N );


// Get the time elapsed since startup
double getTime();


// Check the answer
bool check_ans( const double *image0, const double *I_ang0, const RayTrace::create_image_struct &data );


// Get the minimum value
double getMin( const std::vector<double>& x );


// Get the maximum value
double getMax( const std::vector<double>& x );


// Get the average value
double getAvg( const std::vector<double>& x );


// Get the standard deviation
double getDev( const std::vector<double>& x );


// Clas to hold options
class Options {
public:
    Options() {};
    int iterations = 1;
    double scale   = 1.0;
    std::vector<std::string> methods;
    std::vector<std::string> read_cmd( int argc, char *argv[] )
    {
        const char *err_msg = "CreateImage called with the wrong number of arguments:\n"
            "  CreateImage <args> file.dat\n"
            "Optional arguments:\n"
            "  -methods=METHODS  Comma seperated list of methods to test.  Default is all availible methods\n"
            "                    cpu, threads, OpenMP, Cuda, Cuda-MultiGPU, OpenAcc, Kokkos-Serial, "
            "Kokkos-Thread, Kokkos-OpenMP, Kokkos-Cuda\n"
            "  -iterations=N     Number of iterations to run.  Time returned will be "
            "the average time/iteration.\n"
            "  -scale=factor     Increate the size of the problem by ~ this factor. "
            "(2.0 - twice as expensive)\n"
            "                    Note: this will disable checking the answer.\n"
            "                    Note: the scale factor is only approximate.\n";
        std::vector<std::string> filenames;
        for ( int i = 1; i < argc; i++ ) {
            if ( argv[i][0] == '-' ) {
                // Processing an argument
                if ( strncmp( argv[i], "-methods=", 9 ) == 0 ) {
                    std::stringstream ss( &argv[i][9] );
                    std::string token;
                    while ( std::getline( ss, token, ',' ) )
                        methods.push_back( token );
                } else if ( strncmp( argv[i], "-iterations=", 12 ) == 0 ) {
                    iterations = atoi( &argv[i][12] );
                } else if ( strncmp( argv[i], "-scale=", 7 ) == 0 ) {
                    scale = atof( &argv[i][7] );
                } else {
                    std::cerr << "Unknown option: " << argv[i] << std::endl;
                    return std::vector<std::string>();
                }
            } else {
                // Processing a filename
                filenames.push_back( argv[i] );
            }
        }
        if ( filenames.empty() )
            std::cerr << err_msg;
        return filenames;
    }
};


#endif

