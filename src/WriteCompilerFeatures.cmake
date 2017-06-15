INCLUDE(CheckCXXCompilerFlag)
INCLUDE(CheckCXXSourceCompiles)


# CMake is adding extra options for extensions that we don't want
SET( CMAKE_CXX11_COMPILE_FEATURES -DDUMMY )
SET( CMAKE_CXX11_EXTENSION_COMPILE_OPTION -DDUMMY )
SET( CMAKE_CXX11_STANDARD_COMPILE_OPTION -DDUMMY )
SET( CMAKE_CXX14_COMPILE_FEATURES -DDUMMY )
SET( CMAKE_CXX14_EXTENSION_COMPILE_OPTION -DDUMMY )
SET( CMAKE_CXX14_STANDARD_COMPILE_OPTION -DDUMMY )
SET( CMAKE_CXX98_COMPILE_FEATURES -DDUMMY )
SET( CMAKE_CXX98_EXTENSION_COMPILE_OPTION -DDUMMY )
SET( CMAKE_CXX98_STANDARD_COMPILE_OPTION -DDUMMY )


# This function writes a list of the supported C++ compiler features
FUNCTION( WRITE_COMPILE_FEATURES FILENAME ${ARGN} )
    SET( PREFIX ${ARGN} )
    IF ( NOT PREFIX )
        SET( PREFIX " " )
    ELSE()
        SET( PREFIX "${PREFIX}_" )
    ENDIF()
    FILE(WRITE ${FILENAME} "// This is a automatically generated file to set define variables for supported C++ features\n" )
    TEST_FEATURE( SHARED_PTR ${FILENAME} ${PREFIX}
	    "#include <memory>
	     int main() {
	        std::shared_ptr<int> ptr;
	        return 0;
	     }"
    )
    TEST_FEATURE( STD_FUNCTION ${FILENAME} ${PREFIX}
	    "#include <functional>
         void myfun(int) { }
	     int main() {
            std::function<void(int)> f_display = myfun;
	        return 0;
	     }"
    )
    TEST_FEATURE( STD_TUPLE ${FILENAME} ${PREFIX}
	    "#include <tuple>
	     int main() {
            std::tuple<double,int> x(1,2);
	        return 0;
	     }"
    )
    TEST_FEATURE( VARIADIC_TEMPLATE ${FILENAME} ${PREFIX}
	    "#include <iostream>
         template<class... Ts> void test(Ts... ts) {}
	     int main() {
            test<int,double>(1,3);
	        return 0;
	     }"
    )
    TEST_FEATURE( THREAD_LOCAL ${FILENAME} ${PREFIX}
	    "#include <iostream>
         template<class... Ts> void test(Ts... ts) {}
	     int main() {
            thread_local static int id = 0;
	        return 0;
	     }"
    )
    TEST_FEATURE( MOVE_CONSTRUCTOR ${FILENAME} ${PREFIX}
	    "#include <iostream>
         struct st { int i; st(){} st(st&&){} private: st(st&); };
         st fun() { return st(); }
	     int main() {
            st tmp = fun();
	        return 0;
	     }"
    )
    TEST_FEATURE( STATIC_ASSERT ${FILENAME} ${PREFIX}
	    "#include <iostream>
	     int main() {
            static_assert(true,\"test\");
	        return 0;
	     }"
    )
ENDFUNCTION()


# This function trys to compile and then sets the appropriate macro
FUNCTION( TEST_FEATURE FEATURE_NAME FILENAME PREFIX CODE )
    IF ( ${CXX_STD} STREQUAL "98" )
        # Disable all features for 98
        SET( ${FEATURE_NAME} OFF )
    ELSE()
        # Check if compiler allows feature
        SET( CMAKE_REQUIRED_FLAGS ${CMAKE_CXX_FLAGS} )
        CHECK_CXX_SOURCE_COMPILES( "${CODE}" ${FEATURE_NAME} )
    ENDIF()
    IF ( ${FEATURE_NAME} )
        FILE(APPEND ${FILENAME} "#define ${PREFIX}ENABLE_${FEATURE_NAME}\n" )
    ELSE()
        FILE(APPEND ${FILENAME} "#define ${PREFIX}DISABLE_${FEATURE_NAME}\n" )
    ENDIF()
ENDFUNCTION()

