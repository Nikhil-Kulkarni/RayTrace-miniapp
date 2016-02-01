// This file contains useful macros including ERROR, WARNING, INSIST, ASSERT, etc.
#ifndef included_RayUtilityMacros
#define included_RayUtilityMacros

#include <iostream>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Include AtomicModel::Utilities
#ifdef USE_ATOMIC_MODEL
#include "utilities/Utilities.h"
#else
namespace AtomicModel {
namespace Utilities {
inline void abort( const std::string &message, const std::string &filename, const int line )
{
    std::cerr << "Exiting due to abort:\n";
    std::cerr << message << std::endl;
    std::cerr << "called in file: " << filename << std::endl;
    std::cerr << "on line: " << line << std::endl;
    exit( -1 );
}
}
}
#endif


/*! \defgroup Macros Set of utility macro functions
 *  \details  These functions are a list of C++ macros that are used
 *            for common operations, including checking for errors.
 *  \addtogroup Macros
 *  @{
 */


/*! \def NULL_STATEMENT
 *  \brief    A null statement
 *  \details  A statement that does nothing, for insure++ make it something
 *            more complex than a simple C null statement to avoid a warning.
 */
#ifdef __INSURE__
#define NULL_STATEMENT            \
    do {                          \
        if ( 0 )                  \
            int nullstatement = 0 \
    }                             \
    }                             \
    while ( 0 )
#else
#define NULL_STATEMENT
#endif


/*! \def NULL_USE(variable)
 *  \brief    A null use of a variable
 *  \details  A null use of a variable, use to avoid GNU compiler warnings about unused variables.
 *  \param    variable  Variable to pretend to use
 */
#ifndef NULL_USE
#define NULL_USE( variable )                 \
    do {                                     \
        if ( 0 ) {                           \
            char *temp = (char *) &variable; \
            temp++;                          \
        }                                    \
    } while ( 0 )
#endif


/*! \def SUPPRESS_UNUSED_WARNING(function)
 *  \brief    Supress compiler warning for unused function
 *  \details  Supress compiler warning for unused function of global variable.
 *            Example usage (global scope): SUPRESS_UNUSED_WARNING(foo);
 *  \param    variable  Variable to pretend to use
 */
#define SUPPRESS_UNUSED_WARNING( function ) \
    void ( *_dummy_tmp_##function )( void ) = ( (void ( * )( void )) function )


/*! \def RAY_ERROR(MSG)
 *  \brief      Throw error
 *  \details    Throw an error exception from within any C++ source code.  The
 *     macro argument may be any standard ostream expression.  The file and
 *     line number of the abort are also printed.
 *  \param MSG  Error message to print
 */
#define RAY_ERROR( MSG )                                          \
    do {                                                          \
        AtomicModel::Utilities::abort( MSG, __FILE__, __LINE__ ); \
    } while ( 0 )


/*! \def RAY_WARNING(MSG)
 *  \brief   Print a warning
 *  \details Print a warning without exit.  Print file and line number of the warning.
 *  \param MSG  Warning message to print
 */
#define RAY_WARNING( MSG )                                                                 \
    do {                                                                                   \
        std::stringstream tboxos;                                                          \
        tboxos << MSG << std::ends;                                                        \
        printp( "WARNING: %s\n   Warning called in %s on line %i\n", tboxos.str().c_str(), \
            __FILE__, __LINE__ );                                                          \
    } while ( 0 )


/*! \def RAY_ASSERT(EXP)
 *  \brief Assert error
 *  \details Throw an error exception from within any C++ source code if the
 *     given expression is not true.  This is a parallel-friendly version
 *     of assert.
 *     The file and line number of the abort are printed along with the stack trace (if availible).
 *  \param EXP  Expression to evaluate
 */
#define RAY_ASSERT( EXP )                                                      \
    do {                                                                       \
        if ( !( EXP ) ) {                                                      \
            std::stringstream tboxos;                                          \
            tboxos << "Failed assertion: " << #EXP << std::ends;               \
            AtomicModel::Utilities::abort( tboxos.str(), __FILE__, __LINE__ ); \
        }                                                                      \
    } while ( 0 )


/*! \def RAY_INSIST(EXP,MSG)
 *  \brief Insist error
 *  \details Throw an error exception from within any C++ source code if the
 *     given expression is not true.  This will also print the given message.
 *     This is a parallel-friendly version of assert.
 *     The file and line number of the abort are printed along with the stack trace (if availible).
 *  \param EXP  Expression to evaluate
 *  \param MSG  Debug message to print
 */
#define RAY_INSIST( EXP, MSG )                                                 \
    do {                                                                       \
        if ( !( EXP ) ) {                                                      \
            std::stringstream tboxos;                                          \
            tboxos << "Failed insist: " << #EXP << std::endl;                  \
            tboxos << "Message: " << MSG << std::ends;                         \
            AtomicModel::Utilities::abort( tboxos.str(), __FILE__, __LINE__ ); \
        }                                                                      \
    } while ( 0 )


/*! @} */


#endif
