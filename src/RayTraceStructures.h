// This file contains structures for use with the RayTrace code
#ifndef included_RayTraceStructures
#define included_RayTraceStructures

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "CompilerFeatures.h"
#include "utilities/MPI_functions.h"

#ifdef USE_ATOMIC_MODEL
#include "AtomicModel.h"
#include "threadpool/thread_pool.h"
#else
#define ZMAX 92
namespace AtomicModel {
typedef void atomic_ion_struct;
typedef void ThreadPool;
};
#endif


// Set the parallel method used for the ray trace
// 1: Domain decomposition
// 2: Round-robin assignment
#define PARALLEL_METHOD 2


// Set the maximum number of seed beams
#define N_SEED_MAX 2


/*! \namespace RayTrace
    \brief A Ray Trace namespace
    This namespace provides routines for the Ray Trace model
*/
namespace RayTrace {


//! Structure to contain the information about the plasma zones for a given length
struct plasma_struct {
    int data_type;  //!<  Is the data 1.5D (1) or 2D (2)
    int N;          //!<  The number of temporal grid points
    int J;          //!<  The number of spatial grid points
    double *T;      //!<  The temporal grid (s) (1xN)
    double *x;      //!<  The x-coordinates (direction perpendicular to the target) (cm) (NxJ)
    double *y;      //!<  The y-coordinates (direction parallel to the target) (cm) (NxJ)
    double *Te;     //!<  The electron temperature (eV) (NxJ)
    double *Ti;     //!<  The ion temperature (eV) (NxJ)
    double *Ne;     //!<  The electron density (cm^-3) (NxJ)
    double *Ni;     //!<  The ion density (cm^-3) (NxJ)
    double *R;      //!<  The self-similar width (cm) (NxJ) (1.5D data only)
    int rad_type;   //!<  Radiation transport model
                    //!<      1 - Optically thin
                    //!<      2 - Optically thick
                    //!<      3 - Use the opacity of the zone
                    //!<      4 - Full radiation transport (only for 1.5D data)
    double *u;      //!<  The velocity of the zone (cm/s) (NxJ)
    double *du_dz;  //!<  The velocity gradient of the zone (cm/s) (NxJ)
    double *dz_rad; //!<  The radiative size to use for the zone (cm) (NxJ)
    //! Constructor used to initialize key values
    plasma_struct();
    //! Destructor
    ~plasma_struct();
#ifdef ENABLE_MOVE_CONSTRUCTOR
    //! Move constructor
    plasma_struct( plasma_struct &&rhs );
    //! Move assignment operator
    plasma_struct &operator=( plasma_struct &&rhs );
#endif
    //! Function to initialize the data
    void initialize( int N, int J, int data_type, int rad_type );
    //! Function to swap data with rhs
    void swap( plasma_struct &rhs );
    //! Function to delete the data pointed to
    void delete_data();
    //! Function to check if the data is valid (no NaNs)
    bool valid() const;
    /*!
     * This function converts the data structure (for a single length) to a byte array.
     * It returns the byte array and number of bytes used.
     * @param[in]  compression  The compression level to use:
     *                              0 - No compression (old data format)
     *                              1 - Don't store zeros, use full double precision
     *                              2 - Don't store zeros, use single precision
     */
    std::pair<char *, size_t> pack( int compression = 0 ) const;
    //! This function converts a byte array to fill the data structure.
    void unpack( std::pair<const char *, size_t> data );
    //! Comparison operator, two operators are equal if the data is the same to within tol
    bool operator==( const plasma_struct &rhs ) const;
    //! operator!=
    inline bool operator!=( const plasma_struct &rhs ) const
    {
        return !( this->operator==( rhs ) );
    }

protected:
    plasma_struct( const plasma_struct & );            // Private copy constructor
    plasma_struct &operator=( const plasma_struct & ); // Private assignment operator
};


//! Structure to contain the information with the gain information
struct gain_struct {
    bool *ii;        //!<  Which zones do we want to process the gain for (1xJ)
    double *g;       //!<  The gain of each zone (cm^-1) (NxJ)
    double *E;       //!<  The emissivity of each zone (NxJ)
    double *W;       //!<  The stimulated rate coefficient from nu->nl (NxJ)
    double *dvL;     //!<  The Lorentzian contribution to the lineshape (s^1) (NxJ)
    double *dvD;     //!<  The Gaussian contribution to the lineshape (s^1) (NxJ)
    int Z_max;       //!<  The maximum degree of ionization
    int M[ZMAX + 1]; //!<  The number of levels to use in each ion (1 x Z_max+1)
    int Z_l;         //!<  The laser ion (indexing starts at 0)
    int nu;          //!<  The laser upper level (indexing starts at 0)
    int nl;          //!<  The laser lower level (indexing starts at 0)
    //! Constructor used to initialize key values
    gain_struct();
    //! Destructor
    ~gain_struct();
#ifdef ENABLE_MOVE_CONSTRUCTOR
    //! Move constructor
    gain_struct( gain_struct &&rhs );
    //! Move assignment operator
    gain_struct &operator=( gain_struct &&rhs );
#endif
    //! Function to initialize the data
    void initialize( int N, int J );
    //! Function to initialize the data by copying an existing structure
    void initialize( const gain_struct &rhs );
    //! Function to swap data with rhs
    void swap( gain_struct &rhs );
    //! Function to delete the data pointed to
    void delete_data();
    //! Function to check if the data is valid (no NaNs)
    bool valid() const;
    /*!
     * This function converts the data structure (for a single length) to a byte array.
     * It returns the byte array and number of bytes used.
     * @param[in]  compression  The compression level to use:
     *                              0 - No compression (old data format)
     *                              1 - Don't store zeros, use full double precision
     *                              2 - Don't store zeros, use single precision
     */
    std::pair<char *, size_t> pack( int compression = 0 ) const;
    //! This function converts a byte array to fill the data structure.
    void unpack( std::pair<const char *, size_t> data );
    //! Comparison operator, two operators are equal if the data is the same to within tol
    bool operator==( const gain_struct &rhs ) const;
    //! operator!=
    inline bool operator!=( const gain_struct &rhs ) const { return !( this->operator==( rhs ) ); }
    //! Get N
    inline int get_N() const { return N; }
    //! Get J
    inline int get_J() const { return J; }
protected:
    gain_struct( const gain_struct & );            // Private copy constructor
    gain_struct &operator=( const gain_struct & ); // Private assignment operator
    int N, J;
};


//! Structure to contain the information with the gain information
struct EUV_beam_struct {
    bool run_ASE;     //!<  Include ASE emission in the calculation
    bool run_sat;     //!<  Include saturation efects in the calculation
    bool run_refract; //!<  Include refraction in the calculation
    double R_scale; //!<  Scale the self-similar width by a factor for the refraction effects (1.5D
    //! runs only)
    double G_scale; //!<  Scale the self-similar width by a factor for the gain effects (1.5D runs
    //! only)
    double lambda; //!<  Laser wavelength (cm)
    double A;      //!<  A coefficient (1/s)
    double Nc;     //!<  Critial density for the EUV laser (cm^-3)
    double *x;     //!<  The spatial grid perpendicular to the target surface (cm)
    double *y;     //!<  The spatial grid parallel to the target surface (cm)
    double *a;     //!<  The angular grid perpendicular to the target (mrad)
    double *b;     //!<  The angular grid parallel to the target (mrad)
    double *z;     //!<  The spatial grid along the line (cm)
    double *v;     //!<  The frequency grid
    double dx;     //!<  The spatial grid spacing perpendicular to the target surface (cm)
    double dy;     //!<  The spatial grid spacing parallel to the target surface (cm)
    double da;     //!<  The angular grid spacing perpendicular to the target (mrad)
    double db;     //!<  The angular grid spacing parallel to the target (mrad)
    double dz;     //!<  The spatial grid spacing along the line (cm)
    double *dv;    //!<  The frequency grid spacing
    double v0;     //!<  The center frequency of the laser
    int nx;        //!<  The number of points in x
    int ny;        //!<  The number of points in y
    int nz;        //!<  The number of points in z
    int na;        //!<  The number of points in a
    int nb;        //!<  The number of points in b
    int nv;        //!<  The number of points in v, dv
    //! Constructor used to initialize key values
    EUV_beam_struct();
    //! Destructor
    ~EUV_beam_struct();
#ifdef ENABLE_MOVE_CONSTRUCTOR
    //! Move constructor
    EUV_beam_struct( EUV_beam_struct &&rhs );
    //! Move assignment operator
    EUV_beam_struct &operator=( EUV_beam_struct &&rhs );
#endif
    //! Function to initialize the data
    void initialize( int nx, int ny, int nz, int na, int nb, int nv );
    //! Function to swap data with rhs
    void swap( EUV_beam_struct &rhs );
    //! Function to delete the data pointed to
    void delete_data();
    //! Function to copy the data from rhs
    void copy( const EUV_beam_struct &rhs );
    //! Function to check if the data is valid (no NaNs)
    bool valid() const;
    /*!
     * This function converts the data structure (for a single length) to a byte array.
     * It returns the byte array and number of bytes used.
     * @param[in]  compression  The compression level to use:
     *                              0 - No compression (old data format)
     *                              1 - Don't store zeros, use full double precision
     *                              2 - Don't store zeros, use single precision
     */
    std::pair<char *, size_t> pack( int compression = 0 ) const;
    //! This function converts a byte array to fill the data structure.
    void unpack( std::pair<const char *, size_t> data );
    //! Comparison operator, two operators are equal if the data is the same to within tol
    bool operator==( const EUV_beam_struct &rhs ) const;
    //! operator!=
    inline bool operator!=( const EUV_beam_struct &rhs ) const
    {
        return !( this->operator==( rhs ) );
    }

protected:
    EUV_beam_struct( const EUV_beam_struct & );            // Private copy constructor
    EUV_beam_struct &operator=( const EUV_beam_struct & ); // Private assignment operator
};


//! Structure to contain information used to get the temporal and frequency shape of the seed
struct seed_beam_shape_struct {
    int n;  //!<  Number of points in T_seed
    int nv; //!<  The number of points in v, dv
    double
        *T; //!<  Temporal grid used to describe the temporal shape of the seed profile ( 1 x nT )
    double *It;  //!<  Intensity profile of the seed ( 1 x nT x 3 )
    double *Ivt; //!<  Intensity-frequency profile of the seed ( nv x nT x 3 )
    //! Constructor used to initialize key values
    seed_beam_shape_struct();
    //! Destructor
    ~seed_beam_shape_struct();
    // Private copy constructor
    seed_beam_shape_struct( const seed_beam_shape_struct &rhs );
    // Private assignment operator
    seed_beam_shape_struct &operator=( const seed_beam_shape_struct &rhs );
    //! Function to initialize the data
    void initialize( int nT, int nv );
    //! Function to delete the data pointed to
    void delete_data();
    //! Function to check if the data is valid (no NaNs)
    bool valid() const;
    /*!
     * This function converts the data structure (for a single length) to a byte array.
     * It returns the byte array and number of bytes used.
     * @param[in]  compression  The compression level to use:
     *                              0 - No compression (old data format)
     *                              1 - Don't store zeros, use full double precision
     *                              2 - Don't store zeros, use single precision
     */
    std::pair<char *, size_t> pack( int compression = 0 ) const;
    //! This function converts a byte array to fill the data structure.
    void unpack( std::pair<const char *, size_t> data );
    //! Comparison operator, two operators are equal if the data is the same to within tol
    bool operator==( const seed_beam_shape_struct &rhs ) const;
    //! operator!=
    inline bool operator!=( const seed_beam_shape_struct &rhs ) const
    {
        return !( this->operator==( rhs ) );
    }
};


//! Structure to contain the information with the gain information
struct seed_beam_struct {
    double *x; //!<  The spatial grid perpendicular to the target surface (cm)
    double *y; //!<  The spatial grid parallel to the target surface (cm)
    double *a; //!<  The angular grid perpendicular to the target (mrad)
    double *b; //!<  The angular grid parallel to the target (mrad)
    double dx; //!<  The spatial grid spacing perpendicular to the target surface (cm)
    double dy; //!<  The spatial grid spacing parallel to the target surface (cm)
    double da; //!<  The angular grid spacing perpendicular to the target (mrad)
    double db; //!<  The angular grid spacing parallel to the target (mrad)
    int nx;    //!<  The number of points in x
    int ny;    //!<  The number of points in y
    int na;    //!<  The number of points in a
    int nb;    //!<  The number of points in b
    double Wx; //!<  FWHM of the seed perpendicular to the target surface (cm)
    double Wy; //!<  FWHM of the seed parallel to the target surface (cm)
    double Wa; //!<  FWHM of the seed in the angular direction perpendicular to the target (mrad)
    double Wb; //!<  FWHM of the seed in the angular direction parallel to the target (mrad)
    double Wv; //!<  FWHM of the seed bandwidth (dv/v0) (Assumes it is line-centered)
    double Wt; //!<  FWHM of the temporal profile of the seed
    double x0; //!<  Center of seed spacing perpendicular to the target surface (cm)
    double y0; //!<  Center of seed spacing parallel to the target surface (cm)
    double a0; //!<  Deflection of the seed perpendicular to the target (mrad)
    double b0; //!<  Deflection of the seed parallel to the target (mrad)
    double t0; //!<  Peak of the arival of the temporal profile (s)
    double E;  //!<  Energy of the seed (J)
    double target;                                  //!<  Target position (cm)
    double chirp;                                   //!<  The chirp of the seed beam
    std::vector<seed_beam_shape_struct> seed_shape; //!<  Seed beam shape information
    std::vector<double> tau;         //!<  The group velocity delay for each length segment
    std::vector<bool> use_transform; //!< Use the transform limited profile
    //! Constructor used to initialize key values
    seed_beam_struct();
    //! Destructor
    ~seed_beam_struct();
    //! Function to initialize the data
    void initialize( int nx, int ny, int na, int nb );
    //! Function to delete the data pointed to
    void delete_data();
    //! Function to check if the data is valid (no NaNs)
    bool valid() const;
    /*!
     * This function converts the data structure (for a single length) to a byte array.
     * It returns the byte array and number of bytes used.
     * @param[in]  compression  The compression level to use:
     *                              0 - No compression (old data format)
     *                              1 - Don't store zeros, use full double precision
     *                              2 - Don't store zeros, use single precision
     */
    std::pair<char *, size_t> pack( int compression = 0 ) const;
    //! This function converts a byte array to fill the data structure.
    void unpack( std::pair<const char *, size_t> data );
    //! Comparison operator, two operators are equal if the data is the same to within tol
    bool operator==( const seed_beam_struct &rhs ) const;
    //! operator!=
    inline bool operator!=( const seed_beam_struct &rhs ) const
    {
        return !( this->operator==( rhs ) );
    }

protected:
    seed_beam_struct( const seed_beam_struct & );            // Private copy constructor
    seed_beam_struct &operator=( const seed_beam_struct & ); // Private assignment operator
};


//! Structure to contain the plasma and information for a single point in time
struct plasma_gain_step_struct {
    int data_type; //!<  Is the data 1.5D (1) or 2D (2)
    int J;         //!<  The number of spatial grid points
    bool *ii;      //!<  Which zones do we want to process the gain for (J)
    double *x;     //!<  The x-coordinates (direction perpendicular to the target) (cm) (J)
    double *y;     //!<  The y-coordinates (direction parallel to the target) (cm) (J)
    double *Ne;    //!<  The electron density (cm^-3) (J)
    double *R;     //!<  The self-similar width (cm) (J) (1.5D data only)
    double *g;     //!<  The gain of each zone (cm^-1) (J)
    double *E;     //!<  The emissivity of each zone (J)
    double *dvL;   //!<  The Lorentzian contribution to the lineshape (s^1) (J)
    double *dvD;   //!<  The Gaussian contribution to the lineshape (s^1) (J)
    //! Empty constructor
    plasma_gain_step_struct(); //!<  Empty constructor
    //! Destructor
    ~plasma_gain_step_struct();
    //! Function to initialize the data
    void initialize( double T, const plasma_struct &plasma, const gain_struct &gain );
    //! Function to delete the data pointed to
    void delete_data();

protected:
    plasma_gain_step_struct( const plasma_gain_step_struct & ); // Private copy constructor
    plasma_gain_step_struct &operator=(
        const plasma_gain_step_struct & ); // Private assignment operator
};


//! Structure to contain the information about the gain for the ray trace for a single length
// Note: Since x, y, and n will be used to compute a gradient,
//    they should always be stored with double precision.
//    Other variables could be stored with single precision.
struct ray_gain_struct {
    int Nx;     //!<  Number of points in the x direction (distance from the target)
    int Ny;     //!<  Number of points in the y direction (distance parallel to the target)
    int Nv;     //!<  Number of frequencies
    double *x;  //!<  x grid (cm) ( Nx )
    double *y;  //!<  y grid (cm) ( Ny )
    double *n;  //!<  Index of refraction ( Nx x Ny )
    float *g0;  //!<  Gain at line center (cm^-1) ( Nx x Ny )
    float *E0;  //!<  Emissivity at line center ( Nx x Ny )
    float *gv;  //!<  Normalized lineshape function ( K x Nx x Ny )
    float *gv0; //!<  Lineshape function at line-center ( Nx x Ny )
    //! Empty constructor
    ray_gain_struct(); //!<  Empty constructor
    //! Destructor
    ~ray_gain_struct();
    //! Function to initialize the data
    void initialize( int Nx, int Ny, int Nv, bool use_emis );
    /*!
     * Function to write the data to a file using the file writer in AtomicModel
     * @param[in]  fid      The file to write to
     * @param[in]  prefix   Prefix to prepend to the name of each field
     */
    void writeData( FILE *fid, const char *prefix ) const;
    /*!
     * This function converts the data structure (for a single length) to a byte array.
     * It returns the byte array and number of bytes used.
     * @param[in]  compression  The compression level to use:
     *                              0 - No compression (old data format)
     *                              1 - Don't store zeros, use full double precision
     *                              2 - Don't store zeros, use single precision
     */
    std::pair<char *, size_t> pack( int compression = 0 ) const;
    //! This function converts a byte array to fill the data structure.
    void unpack( std::pair<const char *, size_t> data );
    /**
     * Deep Copy-in instances to accelerator device.  If the code is not compiled for any
     *   accelerators this will throw an error.
     * @param[in] N         Number of instances in the array to copy to device
     * @param[in] arr       Pointer to source array of host instances.
     * @return new device pointer to array of structures
     */
    static const ray_gain_struct *copy_device( size_t N, const ray_gain_struct *arr );
    /**
     * Free device array.
     * @param[in] N          Number of instances in the array
     * @param[in] host_arr   Pointer to host array to free from device
     * @param[in] device_arr Pointer to device array to free from device
     */
    static void free_device(
        size_t N, const ray_gain_struct *host_arr, const ray_gain_struct *device_arr );

protected:
    ray_gain_struct( const ray_gain_struct & );            // Private copy constructor
    ray_gain_struct &operator=( const ray_gain_struct & ); // Private assignment operator
};


//! Structure to contain the information about the seed for the ray trace
struct ray_seed_struct {
    int dim[5];   //!<  The size of each dimension
    double *x[5]; //!<  The grid (x,y,a,b,v)
    double *f[5]; //!<  The value of f at each grid point
    double f0;    //!<  A scaling factor
    //! Constructor used to initialize key values
    ray_seed_struct();
    //! Destructor
    ~ray_seed_struct();
    //! Function to initialize the data
    void initialize( int dim[5] );
    //! Function to delete the data pointed to
    void delete_data();
    //! Function to check if the seed beam is zero for the given euv_beam
    bool is_zero( const EUV_beam_struct &euv_beam ) const;
    /*!
     * This function converts the data structure (for a single length) to a byte array.
     * It returns the byte array and number of bytes used.
     * @param[in]  compression  The compression level to use:
     *                              0 - No compression (old data format)
     *                              1 - Don't store zeros, use full double precision
     *                              2 - Don't store zeros, use single precision
     */
    std::pair<char *, size_t> pack( int compression = 0 ) const;
    //! This function converts a byte array to fill the data structure.
    void unpack( std::pair<const char *, size_t> data );
    /**
     * Deep Copy-in instances to accelerator device.  If the code is not compiled for any
     *   accelerators this will return a pointer to this.
     * @return new device pointer to array of structures
     */
    const ray_seed_struct *copy_device() const;
    /**
     * Free device array.
     * @param[in] host_seed   Pointer to host structure to free from device
     * @param[in] device_seed Pointer to device structure to free from device
     */
    static void free_device( const ray_seed_struct *host_seed, const ray_seed_struct *device_seed );

protected:
    ray_seed_struct( const ray_seed_struct & );            // Private copy constructor
    ray_seed_struct &operator=( const ray_seed_struct & ); // Private assignment operator
};


//! Structure used to call create_image
//! Note: the user is responsible for deallocating the data
struct create_image_struct {
    int N;       //!<  Number of lengths
    int N_start; //!<  First ray to process (set to 0 for domain based decomposition, otherwise set
    //! to a unique number for each thread)
    int N_parallel; //!<  Number of rays processed in parallel (set to 0 for domain based
    //! decomposition, otherwise set to the number of threads)
    double dz;                       //!<  Grid size along the length
    const EUV_beam_struct *euv_beam; //!<  Input beam structure that contains information about the
    //! desired euv beam grid
    const seed_beam_struct *seed_beam; //!<  Input seed beam structure (may be NULL)
    const ray_gain_struct *gain;       //!<  Gain structure for each length (1xN)
    const ray_seed_struct *seed;       //!<  Seed structure (may be NULL)
    double *image; //!<  Output Spatial intensity image (Nv x Nx x Ny) (will be allocated by
    //! create_image and free'd by destructor)
    double *I_ang; //!<  Output Intensity in angular space  (Na x Nb)  (will be allocated by
    //! create_image and free'd by destructor)
    //! Constructor
    create_image_struct();
    //! Destructor
    ~create_image_struct();
    /*!
     * This function converts the data structure (for a single length) to a byte array.
     * It returns the byte array and number of bytes used.
     * @param[in]  compression  The compression level to use:
     *                              0 - No compression (old data format)
     *                              1 - Don't store zeros, use full double precision
     *                              2 - Don't store zeros, use single precision
     */
    std::pair<char *, size_t> pack( int compression = 0 ) const;
    //! This function converts a byte array to fill the data structure.
    void unpack( std::pair<const char *, size_t> data );

protected:
    create_image_struct( const create_image_struct & );            // Private copy constructor
    create_image_struct &operator=( const create_image_struct & ); // Private assignment operator
};


// Structure to contain the intensity variables for a single step
struct intensity_step_struct {
    double *E_v;                    //!<  Frequency profile (Nv)
    double *image;                  //!<  Near-field image ( Nx x Ny )
    double *E_ang;                  //!<  Far-field image ( Na x Nb )
    double *W;                      //!<  Stimulated emission rate (1/s) ( Nx x Ny )
    int N_seed;                     //!<  Number of seed beams
    double *E_v_seed[N_SEED_MAX];   //!<  Seeded frequency profile ( N_seed x Nv )
    double *image_seed[N_SEED_MAX]; //!<  Seeded near-field image ( N_seed x ( Nx x Ny ) )
    double *E_ang_seed[N_SEED_MAX]; //!<  Seeded far-field image ( N_seed x ( Na x Nb ) )
    // Constructor used to initialize key values
    intensity_step_struct();
    // De-constructor
    ~intensity_step_struct();
    //! Function to allocate and initialize the internal data
    void initialize( int nx, int ny, int na, int nb, int nv, int N_seed );
    //! Function to delete the internal data
    void delete_data();
    //! Function to zero out the data (does not affect the allocation)
    void zero();
    //! Function to add a step (Note: *this must the second step)
    void add( const intensity_step_struct &data, bool add_W );
    //! Function to copy the data
    void copy( const intensity_step_struct &rhs );
    //! Reduce the data across the given comm
    void sum_reduce( MPI_Comm comm );
    //! Check that the data is valid (no negitive or NaN intensities)
    bool valid();
    //! Swap the data with rhs
    void swap( intensity_step_struct &rhs );

private:
    intensity_step_struct( const intensity_step_struct & ); // Private copy constructor
    intensity_step_struct &operator=(
        const intensity_step_struct & ); // Private assignment operator
    int nx, ny, na, nb, nv;
    friend struct intensity_struct;
};


// Structure to contain the intensity variables
struct intensity_struct {
    double *E_v;                    //!<  Frequency profile ( Nv x N )
    double *image;                  //!<  Near-field image ( Nx x Ny x N )
    double *E_ang;                  //!<  Far-field image ( Na x Nb x N )
    double *E_sum;                  //!<  Temporal energy profile ( 1 x N )
    double *I_it;                   //!<  Temporal intensity profile ( 1 x N )
    double E_tot;                   //!<  Total energy
    double *W;                      //!<  Stimulated emission rate (1/s) ( Nx x Ny x N )
    int N_seed;                     //!<  Number of seed beams
    double *E_v_seed[N_SEED_MAX];   //!<  Seeded frequency profile ( N_seed x ( Nv x N ) )
    double *image_seed[N_SEED_MAX]; //!<  Seeded near-field image ( N_seed x ( Nx x Ny x N ) )
    double *E_ang_seed[N_SEED_MAX]; //!<  Seeded far-field image ( N_seed x ( Na x Nb x N ) )
    double *E_sum_seed[N_SEED_MAX]; //!<  Seeded temporal profile ( N_seed x ( 1 x N ) )
    double *I_it_seed[N_SEED_MAX];  //!<  Seeded temporal intensity profile ( N_seed x ( 1 x N ) )
    double E_tot_seed[N_SEED_MAX];  //!<  Seeded total energy ( N_seed x 1 )
    //! Constructor used to initialize key values
    intensity_struct();
    //! De-constructor
    ~intensity_struct();
#ifdef ENABLE_MOVE_CONSTRUCTOR
    //! Move constructor
    intensity_struct( intensity_struct &&rhs );
    //! Move assignment operator
    intensity_struct &operator=( intensity_struct &&rhs );
#endif
    //! Function to allocate and initialize the internal data
    void initialize( int N, int nx, int ny, int na, int nb, int nv, int N_seed );
    //! Function to delete the internal data
    void delete_data();
    //! Function to zero out the data (does not affect the allocation)
    void zero();
    //! Swap the data with rhs
    void swap( intensity_struct &rhs );
    //! Copy the data from a given step into
    /*!
     * @brief  Copy the step data
     * @details  This function will copy the data from a step into the structure
     * @param i             The index of the step we are copying
     * @param euv_beam      The euv_beam data
     * @param I_step        The step data
     */
    void copy_step( int i, const EUV_beam_struct &euv_beam, const intensity_step_struct &I_step );
    //! Comparison operator, two operators are equal if the data is the same to within tol
    bool operator==( const intensity_struct &rhs ) const;
    //! operator!=
    inline bool operator!=( const intensity_struct &rhs ) const
    {
        return !( this->operator==( rhs ) );
    }
    //! Access N
    inline int get_N() const { return N; }
    //! Access nx
    inline int get_nx() const { return nx; }
    //! Access ny
    inline int get_ny() const { return ny; }
    //! Access na
    inline int get_na() const { return na; }
    //! Access nb
    inline int get_nb() const { return nb; }
    //! Access nv
    inline int get_nv() const { return nv; }
private:
    intensity_struct( const intensity_struct & );            // Private copy constructor
    intensity_struct &operator=( const intensity_struct & ); // Private assignment operator
    int N, nx, ny, na, nb, nv;
};


//! Structure used to identify recursion tree for calc_step
struct tree_struct {
    int level;      //!<  Current level of the tree
    tree_struct *a; //!<  The first step of the recursion
    tree_struct *b; //!<  The second step of the recursion
    //! Constructor used to initialize key values
    tree_struct();
    //! De-constructor
    ~tree_struct();
    /*!
     * This function converts the data structure (for a single length) to a byte array.
     * It returns the byte array and number of bytes used.
     */
    std::pair<char *, size_t> pack() const;
    //! This function converts a byte array to fill the data structure.
    void unpack( std::pair<const char *, size_t> data );
    //! Comparison operator
    bool operator==( const tree_struct &rhs ) const;
    //! operator!=
    inline bool operator!=( const tree_struct &rhs ) const { return !( this->operator==( rhs ) ); }
    //! Get the maximum depth of the tree
    int depth() const;
    //! Get the total number of nodes of the tree
    int nodes() const;

protected:
    tree_struct( const tree_struct & );            // Private copy constructor
    tree_struct &operator=( const tree_struct & ); // Private assignment operator
};


//! Structure used to contain information about the load balancing
struct load_balance_struct {
    bool sync_N_pop; //!<  Do we want to syncronize N_ion and N_meta across all processors
    int size;        //!<  The number of processors
    int rank;        //!<  The rank of the current processor
    int J;           //!<  The number of zones (should match plasma.J)
    int *proc_zone;  //!<  Which processor is in charge of which zone (1xJ)
#if PARALLEL_METHOD == 1
    int i_ASE[4]; //!<  Starting index for [x,y,a,b] for the ASE computation (for the current
    //! processor)
    int n_ASE[4]; //!<  Number of point in [x,y,a,b] for the ASE computation (for the current
    //! processor)
    int i_seed[4]; //!<  Starting index for [x,y,a,b] for the seed computation (for the current
    //! processor)
    int n_seed[4]; //!<  Number of point in [x,y,a,b] for the seed computation (for the current
//! processor)
#elif PARALLEL_METHOD == 2
    int N_start; //!<  First ray to process (set to 0 for domain based decompoisition, otherwise set
    //! to a unique number for each thread)
    int N_parallel; //!<  Number of rays processed in parallel (set to 0 for domain based
                    //! decompoisition, otherwise set to the number of threads)
#else
#error
#endif
    MPI_Comm comm; //!< Communicator to use
    //! Empty constructor
    load_balance_struct();
    //! Constructor used to initialize key values
    explicit load_balance_struct( int J );
    //! Destructor
    ~load_balance_struct();
    //! Check that the load balance data is valid
    bool valid( int J, int rad_type, const bool *ii ) const;

protected:
    load_balance_struct( const load_balance_struct & );            // Private copy constructor
    load_balance_struct &operator=( const load_balance_struct & ); // Private assignment operator
};


//! Structure used to store the inputs for an apply call
struct apply_input_params {
    double T0;                       //!<  Initial time (s)
    double dT;                       //!<  Timestep (s)
    int NN;                          //!<  Number of lengths
    const EUV_beam_struct *euv_beam; //!<  EUV beam structure
    const plasma_struct *plasma;     //!<  Plasma structure (N)
    const gain_struct *gain;         //!<  Gain structure (N)
    int N_seed;                      //!<  Number of seed beams
    const seed_beam_struct
        *seed_beam;       //!<  Array containing th seed beams seed beam structure (N_seed)
    const double *N_meta; //!<  The current level populations (cm^-3) (sum(M) x J)
    const double *W0; //!<  The stimulated emission rate coefficients from nu to nl at time T0 (J)
    const AtomicModel::atomic_ion_struct *ion; //!< The atomic data for the laser species
    const load_balance_struct
        *load_balance;              //!< Structure containing the load balancing information
    AtomicModel::ThreadPool *tpool; //!<  The thread pool
    std::string interp_W_method;    //!<  Integration method for W
    // Constructor used to initialize key values
    apply_input_params();
    // De-constructor
    ~apply_input_params();
    // Private copy constructor
    apply_input_params( const apply_input_params & );
    // Private assignment operator
    apply_input_params &operator=( const apply_input_params & );

protected:
};


//! Structure used to store the results from an apply call
struct apply_output_results {
    int J;             //!<  The number of spatial points
    int Nt;            //!<  The number of spatial points
    double *W;         //!<  The new stimulated emission rates f(W,t) at T+dT (J)
    double *r;         //!<  The function residual G(W) = W-f(W,t) (J)
    double *Jii;       //!<  An approximate diagonal Jacobian for G(W)
    double *Hii;       //!<  An approximate diagonal Hessian for G(W)
    double *g;         //!<  The stimulated emission rate at T+dT (J)
    double *E;         //!<  The emissivity rate at T+dT (J)
    double *N_meta;    //!<  The level populations at T+dT (cm^-3) (sum(M) x J)
    tree_struct *tree; //!<  Optional tree structure returned by the recursive calc_step solver
    intensity_step_struct intensity; //!<  The output intensity structure at T+dT
    // Constructor used to initialize key values
    apply_output_results( int J, int Nt );
    // De-constructor
    ~apply_output_results();

protected:
    apply_output_results();                                          // Private empty constructor
    apply_output_results( const apply_output_results & );            // Private copy constructor
    apply_output_results &operator=( const apply_output_results & ); // Private assignment operator
};
};

#endif
