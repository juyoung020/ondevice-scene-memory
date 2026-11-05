cmake_minimum_required(VERSION 3.16)
# C 도 켬: RTAB-Map → PCL → VTK 설정이 MPI::MPI_C 를 찾는다(Debian·Ubuntu 패키지)
project(rtab_bridge LANGUAGES C CXX)

# 입력(RealSense 카메라 또는 TUM 폴더) → RTAB-Map 위치 추적·지도 → 장면 스트림(위치 + 컬러 + 정렬된 깊이) → memmap --stream
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(BRIDGE_REALSENSE "librealsense2 가 있으면 RealSense 카메라 입력을 함께 빌드" ON)

find_package(RTABMap REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Threads REQUIRED)

add_executable(rtab_bridge
  src/main.cpp
  src/tum_source.cpp
  src/stream_writer.cpp
)
target_include_directories(rtab_bridge PRIVATE src)
target_link_libraries(rtab_bridge PRIVATE rtabmap::core ${OpenCV_LIBS} Threads::Threads)

if(BRIDGE_REALSENSE)
  find_package(realsense2 QUIET)
  if(realsense2_FOUND)
    message(STATUS "RealSense 입력 켬 (librealsense2 ${realsense2_VERSION})")
    target_sources(rtab_bridge PRIVATE src/realsense_source.cpp src/depth_align.cpp)
    target_compile_definitions(rtab_bridge PRIVATE BRIDGE_HAVE_REALSENSE=1)
    target_link_libraries(rtab_bridge PRIVATE realsense2::realsense2)
  else()
    message(STATUS "librealsense2 없음: TUM 재생 입력만 빌드")
  endif()
endif()
