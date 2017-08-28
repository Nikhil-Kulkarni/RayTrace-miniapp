CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
IF ( ${CMAKE_MAJOR_VERSION} EQUAL 3 )
    IF ( NOT ( ${CMAKE_MINOR_VERSION} EQUAL 0 ) )
        CMAKE_POLICY(SET CMP0053 OLD)
    ENDIF()
ENDIF()

MESSAGE( "==============================" )
MESSAGE( "Configuring Ray Trace MiniApps" )
MESSAGE( "==============================" )


# Prevent users from building in place
IF ("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}" )
    MESSAGE( FATAL_ERROR "Building code in place is a bad idea" )
ENDIF ()


# Set the project name
PROJECT( RAYTRACE-MINIAPPS CXX )


# Set some common paths
SET( RAYTRACE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
SET( RAYTRACE_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR} )
IF( RAYTRACE_INSTALL_DIR )
    # Install path is set
ELSE()
    SET( RAYTRACE_INSTALL_DIR ${RAYTRACE_BUILD_DIR} )
ENDIF()


# Include macros
SET( PROJ RAYTRACE )
INCLUDE( "${RAYTRACE_SOURCE_DIR}/macros.cmake" )
SET_COMPILER_FLAGS()


# Check if we are using an MPI compiler
SET( CMAKE_REQUIRED_FLAGS ${CMAKE_CXX_FLAGS} )
CHECK_CXX_SOURCE_COMPILES(
    "#include <mpi.h>
     int main(int, char**) {
        MPI_Init(NULL, NULL);
        MPI_Finalize();
     }"
    USE_MPI )
IF ( USE_MPI )
    ADD_DEFINITIONS( -DUSE_MPI )
    MESSAGE( "Using MPI" )
ENDIF()


# Enable OpenMP
IF ( USE_OPENMP )
    ADD_DEFINITIONS( -DUSE_OPENMP )
    IF ( USING_GCC )
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    ELSEIF ( USING_CRAY )
    ELSEIF ( USING_PGCC )
        #SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -openmp")
        #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mp")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mp")
    ELSEIF ( USING_CLANG )
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -pthread")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread")
    ELSEIF ( USING_XL )
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qsmp=omp")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qsmp=omp")
    ELSE()
        MESSAGE(FATAL_ERROR "Compiling with OpenMP is not yet set for this compiler")
    ENDIF()
    MESSAGE( "Using OpenMP" )
ENDIF()


# Enable OpenACC
IF ( USE_OPENACC )
    ADD_DEFINITIONS( -DUSE_OPENACC )
    IF ( USING_PGCC )
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -acc")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc")
    ENDIF()
    MESSAGE( "Using OpenACC" )
ENDIF()


# Enable kokkos
IF ( USE_KOKKOS )
    ADD_DEFINITIONS( -DUSE_KOKKOS )
    VERIFY_PATH( ${KOKKOS_DIRECTORY} )
    VERIFY_PATH( ${KOKKOS_DIRECTORY}/include )
    VERIFY_PATH( ${KOKKOS_DIRECTORY}/lib )
    INCLUDE_DIRECTORIES( ${KOKKOS_DIRECTORY}/include )
    FIND_LIBRARY( KOKKOS_LIB  NAMES kokkos  PATHS "${KOKKOS_DIRECTORY}/lib"  NO_DEFAULT_PATH )
    IF ( NOT KOKKOS_LIB )
        MESSAGE(FATAL_ERROR "kokkos library not found in ${KOKKOS_DIRECTORY}/lib")
    ENDIF()
    IF ( EXISTS "${KOKKOS_DIRECTORY}/nvcc_wrapper" )
        SET( KOKKOS_WRAPPER "${KOKKOS_DIRECTORY}/nvcc_wrapper" )
    ENDIF()
    IF ( KOKKOS_WRAPPER )
        SET( CMAKE_CXX_COMPILER "${KOKKOS_WRAPPER}" )
    ELSE()
        MESSAGE("KOKKOS_WRAPPPER not set, set to the kokkos nvcc wrapper if compiling with CUDA")
    ENDIF()
    SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -fopenmp" )
    IF ( EXISTS KOKKOS_WRAPPER )
        SET( CMAKE_CXX_COMPILER "${KOKKOS_WRAPPER}" )
    ENDIF()
    MESSAGE( "Using kokkos" )
ELSE()
    NULL_USE(KOKKOS_DIRECTORY)
    NULL_USE(KOKKOS_WRAPPER)
ENDIF()


# Enable cuda
IF ( USE_CUDA )
    # Include FindCUDA
    INCLUDE( FindCUDA )
    IF ( NOT CUDA_FOUND )
        MESSAGE ( FATAL_ERROR "CUDA not found" )
    ENDIF()
    IF ( CUDA_FLAGS )
        SET( CUDA_NVCC_FLAGS "${CUDA_FLAGS} ${CXX_STD_FLAG}" )
    ENDIF()
    IF(NOT CUDA_NVCC_FLAGS)
        # Set minimum requirements
        SET( CUDA_NVCC_FLAGS "-arch=sm_30 ${CXX_STD_FLAG}" )
    ENDIF()
    IF ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
        SET( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo" )
    ENDIF()
    # SET( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CXX_STD_FLAG}" )
    IF( NOT CMAKE_BUILD_TYPE )
        MESSAGE(FATAL_ERROR "CMAKE_BUILD_TYPE is not set")
    ELSEIF( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
        SET( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -O0" )
    ELSEIF( ${CMAKE_BUILD_TYPE} STREQUAL "Release" )
        SET( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3" )
    ENDIF()
    SET( CUDA_PROPAGATE_HOST_FLAGS OFF )
    SET( CUDA_FIND_QUIETLY )
    STRING( REPLACE " " ";" CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} )
    INCLUDE_DIRECTORIES ( ${CUDA_INCLUDE_DIRS} )
    ADD_DEFINITIONS( "-DUSE_CUDA" ) 
    MESSAGE( "Using CUDA ${CUDA_VERSION}" ) 
    MESSAGE( "  CUDA_LIBRARIES = ${CUDA_LIBRARIES}" )
    MESSAGE( "  CUDA_INCLUDE   = ${CUDA_INCLUDE_DIRS}" )
    MESSAGE( "  CUDA_NVCC_FLAGS = ${CUDA_NVCC_FLAGS}" )
    MESSAGE( "  CUDA_TOOLKIT_ROOT = ${CUDA_TOOLKIT_ROOT_DIR}" )
    SET( EXTERNAL_LIBS ${EXTERNAL_LIBS} ${CUDA_LIBRARIES} )
ELSE()
    NULL_USE(CUDA_FLAGS)
ENDIF()


# Get the supported compiler features
INCLUDE( "${RAYTRACE_SOURCE_DIR}/WriteCompilerFeatures.cmake" )
WRITE_COMPILE_FEATURES( "${RAYTRACE_INSTALL_DIR}/include/CompilerFeatures.h" )
INCLUDE_DIRECTORIES( "${RAYTRACE_INSTALL_DIR}/include" )


# Create custom targets for distclean
ADD_DISTCLEAN( libRayTrace.* null_timer CreateImage* )


# Create the library
INCLUDE_DIRECTORIES( ${RAYTRACE_SOURCE_DIR} )
ADD_DEFINITIONS( -DDISABLE_WRITE_FAILED_RAYS )
SET( SOURCES RayTrace RayTraceImage.cpp RayTraceStructures.cpp utilities/RayUtilities.cpp AtomicModel/interp.cpp RayTraceImageCPU.cpp CreateImageHelpers.cpp )
IF ( USE_OPENACC )
    SET( SOURCES ${SOURCES} RayTraceImageOpenACC.cpp )
ENDIF()
IF ( USE_KOKKOS )
    #SET( SOURCES ${SOURCES} RayTraceImageKokkos.cpp )
ENDIF()
IF ( USE_CUDA )
    SET( CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE )
    SET( CMAKE_BUILD_WITH_INSTALL_RPATH TRUE )
    CUDA_COMPILE( CUBINS RayTraceImageCuda.cu )
ENDIF()
ADD_LIBRARY( ${SOURCES} ${CUBINS} )


# Add the applications
ADD_EXECUTABLE( CreateImage CreateImage.cpp )
TARGET_LINK_LIBRARIES( CreateImage RayTrace ${CUDA_LIBRARIES} ${KOKKOS_LIB} ${TIMER_LIBS} ${LDFLAGS} ${LDLIBS} )
INSTALL( TARGETS CreateImage DESTINATION ${RAYTRACE_INSTALL_DIR}/bin )


