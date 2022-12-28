cmake_minimum_required(VERSION 2.8)
project( ESPCN )
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package( OpenCV REQUIRED )
find_package(Threads REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( ESPCN ESPCN.cpp )
target_link_libraries( ESPCN ${OpenCV_LIBS} )
target_link_libraries(ESPCN PRIVATE Threads::Threads)


