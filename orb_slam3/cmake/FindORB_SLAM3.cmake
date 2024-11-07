if(NOT ORB_SLAM3_ROOT_DIR)
    message(WARNING "The variable ORB_SLAM3_ROOT_DIR is not set! Please set this variable.")
endif()

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

list(APPEND ORB_SLAM3_INCLUDE_DIRS
    ${ORB_SLAM3_ROOT_DIR}/..
    ${ORB_SLAM3_ROOT_DIR}
    ${ORB_SLAM3_ROOT_DIR}/include
    ${ORB_SLAM3_ROOT_DIR}/include/CameraModels
    ${ORB_SLAM3_ROOT_DIR}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
)

list(APPEND ORB_SLAM3_LIBS
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${ORB_SLAM3_ROOT_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${ORB_SLAM3_ROOT_DIR}/Thirdparty/g2o/lib/libg2o.so
    -lboost_serialization
    -lcrypto
    ${ORB_SLAM3_ROOT_DIR}/lib/libORB_SLAM3.so
)

find_path(TMP_INCLUDE_DIR
    NAMES System.h
    HINTS ${ORB_SLAM3_INCLUDE_DIRS}
    NO_CACHE
)

find_library(TMP_LIB_DIR
    NAMES DBoW2 g2o ORB_SLAM3
    HINTS ${ORB_SLAM3_ROOT_DIR}/lib ${ORB_SLAM3_ROOT_DIR}/Thirdparty/DBoW2/lib ${ORB_SLAM3_ROOT_DIR}/Thirdparty/g2o/lib
    NO_CACHE
)

if(TMP_INCLUDE_DIR AND TMP_LIB_DIR)
    set(ORB_SLAM3_FOUND TRUE)
else()
    unset(ORB_SLAM3_INCLUDE_DIRS)
    unset(ORB_SLAM3_LIBS)
endif()

unset(TMP_INCLUDE_DIR)
unset(TMP_LIB_DIR)
