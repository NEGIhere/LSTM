cmake_minimum_required(VERSION 3.6)
project(LSTM)

if (UNIX)
    set(CMAKE_CXX_STANDARD 14)
    #include_directories("");
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funsafe-math-optimizations")
elseif (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -static-libgcc")
endif()

set(SOURCE_FILES
        src/LSTM.cpp
        old/Net.h
        old/Net.cpp
        old/Neuron.h
        old/Neuron.cpp
        src/utils/Utils.h
        src/utils/Utils.cpp
        src/utils/Config.h
        src/utils/Config.cpp
        src/FuncPredictTest.cpp
        src/FuncPredictTest.h src/RNNTest.cpp src/RNNTest.h src/utils/matrix.cpp src/utils/matrix.h)

add_executable(LSTM ${SOURCE_FILES})


if (UNIX)
    target_link_libraries(LSTM sfml-system sfml-graphics sfml-window)
elseif (WIN32)
    set(SFML_ROOT "E:/Development/CPP/SFML-master")
    set(CMAKE_MODULE_PATH "${SFML_ROOT}/cmake/Modules" ${CMAKE_MODULE_PATH})
    find_package(SFML 2 REQUIRED system graphics window)

    if (SFML_FOUND)
        include_directories(${SFML_INCLUDE_DIR})
        target_link_libraries(LSTM ${SFML_LIBRARIES})
    endif()

    # set(BOX2D_ROOT "E:/Development/CPP/Box2D-master/Box2D")
    # include_directories(${BOX2D_ROOT})
    # find_library(Box2D NAMES libBox2D HINTS "${BOX2D_ROOT}/lib")
    # target_link_libraries(LSTM ${Box2D})
endif()