cmake_minimum_required(VERSION 3.16)
project(detector_analysis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(MATIO REQUIRED IMPORTED_TARGET matio)

find_package(Matplot++ CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(detector_analysis main.cpp)

target_link_libraries(detector_analysis PRIVATE
    PkgConfig::MATIO
    Matplot++::matplot
    ${OpenCV_LIBS}
)

target_include_directories(detector_analysis PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)