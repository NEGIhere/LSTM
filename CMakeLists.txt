cmake_minimum_required(VERSION 3.6)
project(LSTM)

if (UNIX)
    set(CMAKE_CXX_STANDARD 14)
    #include_directories("");
endif()

set(SOURCE_FILES
        src/LSTM.cpp
        src/Net.h
        src/Net.cpp
        src/Neuron.h
        src/Neuron.cpp
        src/Utils.h
        src/Utils.cpp
        src/Config.h
        src/Config.cpp
        src/FuncPredictTest.cpp
        src/FuncPredictTest.h )

add_executable(LSTM ${SOURCE_FILES})


if (UNIX)
    target_link_libraries(LSTM sfml-graphics sfml-window sfml-system)
elseif (WIN32)

endif()