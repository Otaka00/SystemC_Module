cmake_minimum_required(VERSION 3.0)
project(Final_SystemC)

# Set the path to your SystemC installation
set(SYSTEMC_HOME "E:\\courses\\Graduation Project\\systemc-master")

# Find SystemC library
find_package(SystemCLanguage CONFIG REQUIRED)

# Set the path to your SystemC include directory
include_directories(${E:\\courses\\Graduation Project\\systemc-master\\systemc-master\\src})

# Add your source files
add_executable(Final_SystemC Final_SystemC.cpp)

# Link against the SystemC libraries
target_link_libraries(Final_SystemC.exe PRIVATE SystemC::systemc)
