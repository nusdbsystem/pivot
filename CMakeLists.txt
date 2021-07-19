cmake_minimum_required(VERSION 3.10)
project(Pivot)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -lboost_program_options")

set(SOURCE_FILES main.cpp
        src/client/client.cpp
        src/client/client.h
        src/utils/util.h
        src/utils/util.cpp
        src/utils/encoder.h
        src/utils/encoder.cpp
        src/utils/djcs_t_aux.h
        src/utils/djcs_t_aux.cpp
        tests/test_encoder.cpp
        tests/test_djcs_t_aux.cpp
        tests/test_encoder.h
        tests/test_djcs_t_aux.h
        src/utils/pb_converter.h
        src/utils/pb_converter.cpp
        src/include/protobuf/common.pb.h
        src/include/protobuf/common.pb.cc
        src/include/protobuf/logistic.pb.h
        src/include/protobuf/logistic.pb.cc
        tests/test_pb_converter.h
        tests/test_pb_converter.cpp
        src/include/protobuf/keys.pb.h
        src/include/protobuf/keys.pb.cc
        src/include/protobuf/cart.pb.h
        src/include/protobuf/cart.pb.cc
        src/utils/spdz/spdz_util.h
        src/utils/spdz/spdz_util.cpp
        src/utils/score.h
        src/utils/score.cpp
        src/models/feature.h
        src/models/feature.cpp
        src/models/tree_node.h
        src/models/tree_node.cpp
        src/models/cart_tree.h
        src/models/cart_tree.cpp
        src/models/random_forest.h
        src/models/random_forest.cpp
        src/models/gbdt.h
        src/models/gbdt.cpp)

set(SPDZ_HOME "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Pivot-SPDZ/")
set(SCAPI_HOME "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libscapi/")
set(HCS_HOME "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libhcs/")


include(FindProtobuf)
find_package(Protobuf REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

include_directories(${SCAPI_HOME}/include
        ${Boost_INCLUDE_DIRS}
        ${PROTOBUF_INCLUDE_DIR}
        ${SPDZ_HOME}
        ${SPDZ_HOME}/local/include)

link_directories(/usr/ssl/lib ${SCAPI_HOME}/install/lib ${SPDZ_HOME}/local/lib)

add_executable(Pivot ${SOURCE_FILES})
target_link_libraries(Pivot
        /usr/lib/x86_64-linux-gnu/libgmp.so
        /usr/lib/x86_64-linux-gnu/libgmpxx.so
        ${HCS_HOME}/lib/libhcs.so
        ${SCAPI_HOME}/libscapi.a
        ${SPDZ_HOME}/libSPDZ.a
        ${Boost_LIBRARIES}
        libboost_thread.a libboost_system.a pthread crypto ssl mpirxx mpir cryptopp sodium
        ${PROTOBUF_LIBRARY})

