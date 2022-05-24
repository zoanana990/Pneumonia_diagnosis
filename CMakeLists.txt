###########################################################################
# @author: Hsin Yu Chen
# @Date: 2021.12.29
###########################################################################

cmake_minimum_required(VERSION 2.8)

project(Final_Project)

# Generate Debug mode or Release mode
#SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release")

# link the c++ standard file system
link_libraries(stdc++fs)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# locate the pytorch library
#set(CMAKE_PREFIX_PATH /home/hsin/Library/libtorch)
set(Torch_DIR /home/hsin/Library/libtorch/share/cmake/Torch)
set(Qt5Widgets_DIR /usr/lib/x86_64-linux-gnu/cmake/Qt5Widgets)
set(Qt5Gui_DIR /usr/lib/x86_64-linux-gnu/cmake/Qt5Gui)
set(Qt5Core_DIR /usr/lib/x86_64-linux-gnu/cmake/Qt5Core)

# find all library we include
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TorchVision REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Core REQUIRED)
#qt5_wrap_cpp( MOC mainwindow.h)

#include_directories(${TORCH_INCLUDE_DIRS})

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Input the header file which is written by me
#add_executable(Final_Project main.cpp dataset.cpp dataset.h model.cpp model.h Classifier.h Classifier.cpp
#        )
add_executable(Final_Project main.cpp dataset.cpp dataset.h model.cpp model.h Classifier.h Classifier.cpp
        mainwindow.cpp mainwindow.h mainwindow.ui)
#add_executable(Final_Project main.cpp dataset.cpp dataset.h)

# link other library here
target_link_libraries(Final_Project "${TORCH_LIBRARIES}")
target_link_libraries(Final_Project ${OpenCV_LIBS})
target_link_libraries(Final_Project TorchVision::TorchVision)
target_link_libraries(Final_Project Qt5::Core Qt5::Gui Qt5::Widgets)

#set_property(TARGET Final_Project PROPERTY CXX_STANDARD 11 )