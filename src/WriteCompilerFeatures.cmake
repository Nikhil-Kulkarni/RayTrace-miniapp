INCLUDE(CheckCXXCompilerFlag)
INCLUDE(CheckCXXSourceCompiles)


# CMake is adding extra options for extensions that we don't want
SET( CMAKE_CXX11_COMPILE_FEATURES )
SET( CMAKE_CXX11_EXTENSION_COMPILE_OPTION )
SET( CMAKE_CXX11_STANDARD_COMPILE_OPTION )
SET( CMAKE_CXX14_COMPILE_FEATURES )
SET( CMAKE_CXX14_EXTENSION_COMPILE_OPTION )
SET( CMAKE_CXX14_STANDARD_COMPILE_OPTION )
SET( CMAKE_CXX98_COMPILE_FEATURES )
SET( CMAKE_CXX98_EXTENSION_COMPILE_OPTION -DUSE_CXX98 ) # This cannot by empty
SET( CMAKE_CXX98_STANDARD_COMPILE_OPTION )


# This function writes a list of the supported C++ compiler features
FUNCTION( WRITE_COMPILE_FEATURES FILENAME )
    FILE(WRITE ${FILENAME} "// This is a automatically generated file to set define variables for supported C++ features\n" )
    SET( CMAKE_REQUIRED_FLAGS ${CMAKE_CXX_FLAGS} )
    TEST_FEATURE( SHARED_PTR ${FILENAME}
	    "#include <memory>
	     int main() {
	        std::shared_ptr<int> ptr;
	        return 0;
	     }"
    )
    TEST_FEATURE( STD_FUNCTION ${FILENAME}
	    "#include <functional>
         void myfun(int) { }
	     int main() {
            std::function<void(int)> f_display = myfun;
	        return 0;
	     }"
    )
    TEST_FEATURE( STD_TUPLE ${FILENAME}
	    "#include <tuple>
	     int main() {
            std::tuple<double,int> x(1,2);
	        return 0;
	     }"
    )
    TEST_FEATURE( VARIADIC_TEMPLATE ${FILENAME}
	    "#include <iostream>
         template<class... Ts> void test(Ts... ts) {}
	     int main() {
            test<int,double>(1,3);
	        return 0;
	     }"
    )
    TEST_FEATURE( THREAD_LOCAL ${FILENAME}
	    "#include <iostream>
         template<class... Ts> void test(Ts... ts) {}
	     int main() {
            thread_local static int id = 0;
	        return 0;
	     }"
    )
    TEST_FEATURE( MOVE_CONSTRUCTOR ${FILENAME}
	    "#include <iostream>
         struct st { int i; st(){} st(st&&){} private: st(st&); };
         st fun() { return st(); }
	     int main() {
            st tmp = fun();
	        return 0;
	     }"
    )
ENDFUNCTION()


# This function trys to compile and then sets the appropriate macro
FUNCTION( TEST_FEATURE FEATURE_NAME FILENAME CODE )
    CHECK_CXX_SOURCE_COMPILES( "${CODE}" ${FEATURE_NAME} )
    IF ( ${FEATURE_NAME} )
        FILE(APPEND ${FILENAME} "#define ENABLE_${FEATURE_NAME}\n" )
    ELSE()
        FILE(APPEND ${FILENAME} "#define DISABLE_${FEATURE_NAME}\n" )
    ENDIF()
ENDFUNCTION()

