cmake_minimum_required(VERSION 3.16)

project(infertensorrt_cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Shared libraries need PIC; harmless for static builds on Windows.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
include(GenerateExportHeader)
include(CMakePackageConfigHelpers)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(INFERTENSORRT_BUILD_SHARED "Build infertensorrt_cpp as a shared library (DLL)" ON)
option(BUILD_EXAMPLES "Build example applications" ON)

if(INFERTENSORRT_BUILD_SHARED)
    set(INFERTENSORRT_LIBRARY_TYPE SHARED)
else()
    set(INFERTENSORRT_LIBRARY_TYPE STATIC)
endif()

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
# Library sources live in src/ (recursively), public headers in include/.
# src/main.cpp is reserved for the optional standalone executable below.
file(GLOB_RECURSE INFERTENSORRT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tensorrt/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/helpers/*.cpp"
)
list(REMOVE_ITEM INFERTENSORRT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
# src/samples/ holds standalone executables, not library sources.
list(FILTER INFERTENSORRT_SOURCES EXCLUDE REGEX "/src/samples/")

file(GLOB_RECURSE INFERTENSORRT_PUBLIC_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
)

# A DLL target needs at least one translation unit. Until the first real source
# is added, build a harmless placeholder so the template configures as-is.
if(NOT INFERTENSORRT_SOURCES)
    message(STATUS "infertensorrt_cpp: no sources in src/, using a placeholder TU")
    set(INFERTENSORRT_PLACEHOLDER "${CMAKE_CURRENT_BINARY_DIR}/infertensorrt_placeholder.cpp")
    if(NOT EXISTS "${INFERTENSORRT_PLACEHOLDER}")
        file(WRITE "${INFERTENSORRT_PLACEHOLDER}"
            "// Auto-generated placeholder so the DLL target has a source file\n"
            "// and produces an import library. Add real sources under src/ and\n"
            "// this file is no longer used.\n"
            "#include \"infertensorrt_cpp/export.h\"\n"
            "\n"
            "extern \"C\" INFERTENSORRT_EXPORT int infertensorrt_cpp_placeholder_anchor(void)\n"
            "{\n"
            "    return 0;\n"
            "}\n")
    endif()
    list(APPEND INFERTENSORRT_SOURCES "${INFERTENSORRT_PLACEHOLDER}")
endif()

# ---------------------------------------------------------------------------
# NVIDIA TensorRT sample helpers
# ---------------------------------------------------------------------------
set(TRT_SAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/tensorrt-samples")

set(TRT_SAMPLES_SOURCES
    "${TRT_SAMPLES_DIR}/common/logger.cpp"
    "${TRT_SAMPLES_DIR}/common/sampleUtils.cpp"
    "${TRT_SAMPLES_DIR}/utils/cacheUtils.cpp"
    "${TRT_SAMPLES_DIR}/utils/fileLock.cpp"
)

if(MSVC)
    # getopt implementation used by argsParser.h.
    list(APPEND TRT_SAMPLES_SOURCES "${TRT_SAMPLES_DIR}/common/getopt.c")
endif()

# ----------------------------------------------------------------------------
# Required dependencies
# ----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

# Prefer CUDAToolkit over the deprecated FindCUDA module.
if(DEFINED CUDA_TOOLKIT_ROOT_DIR AND NOT DEFINED CUDAToolkit_ROOT)
    set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
endif()
find_package(CUDAToolkit REQUIRED)

# TensorRT ships version-suffixed libraries (nvinfer_11.lib, libnvinfer.so.11),
# so resolve the major version from its headers.
if(NOT TENSORRT_DIR AND DEFINED ENV{TENSORRT_DIR})
    set(TENSORRT_DIR "$ENV{TENSORRT_DIR}")
endif()

if(TENSORRT_DIR)
    if(NOT EXISTS "${TENSORRT_DIR}/include/NvInferVersion.h")
        message(FATAL_ERROR
            "TENSORRT_DIR is set to '${TENSORRT_DIR}', but include/NvInferVersion.h was not found. "
            "Point TENSORRT_DIR at the TensorRT root directory.")
    endif()
    file(READ "${TENSORRT_DIR}/include/NvInferVersion.h" _trt_version_header)
    if(_trt_version_header MATCHES "TRT_MAJOR_ENTERPRISE[ \t]+([0-9]+)")
        set(_trt_major "${CMAKE_MATCH_1}")
    endif()
else()
    message(WARNING "TENSORRT_DIR is not set; TensorRT headers and libraries will not be available.")
endif()

# ---------------------------------------------------------------------------
# Library (DLL on Windows, .so/.dylib elsewhere)
# ---------------------------------------------------------------------------
add_library(infertensorrt_cpp ${INFERTENSORRT_LIBRARY_TYPE})
add_library(infertensorrt::infertensorrt_cpp ALIAS infertensorrt_cpp)

target_sources(infertensorrt_cpp PRIVATE
    ${INFERTENSORRT_SOURCES}
    ${TRT_SAMPLES_SOURCES}
)

target_include_directories(infertensorrt_cpp
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/public>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/tensorrt"
        # NVIDIA TensorRT sample helpers (common.h, logger.h, ...).
        "${TRT_SAMPLES_DIR}/common"
        # Lets common.h resolve "utils/cacheUtils.h".
        "${TRT_SAMPLES_DIR}"
)

target_link_libraries(infertensorrt_cpp
    PRIVATE
        CUDA::cudart
)

# Enable the ONNX parser parts of the TensorRT sample headers
# (parserOnnxConfig.h / sampleEntrypoints.h are guarded by this macro).
target_compile_definitions(infertensorrt_cpp PRIVATE TRT_BUILD_ONNX_PARSER=1)

if(TENSORRT_DIR)
    target_include_directories(infertensorrt_cpp PRIVATE "${TENSORRT_DIR}/include")

    # Kept in a variable so sample executables can link the same libraries.
    set(INFERTENSORRT_TRT_LIBRARIES)

    foreach(_trt_component nvinfer nvinfer_plugin nvonnxparser)
        if(_trt_major)
            set(_trt_names "${_trt_component}_${_trt_major}" "${_trt_component}")
        else()
            set(_trt_names "${_trt_component}")
        endif()
        find_library(_trt_${_trt_component}_library
            NAMES ${_trt_names}
            PATHS "${TENSORRT_DIR}/lib"
            NO_DEFAULT_PATH
        )
        if(_trt_${_trt_component}_library)
            list(APPEND INFERTENSORRT_TRT_LIBRARIES "${_trt_${_trt_component}_library}")
        else()
            message(WARNING "TensorRT component '${_trt_component}' not found under ${TENSORRT_DIR}/lib")
        endif()
    endforeach()

    target_link_libraries(infertensorrt_cpp PRIVATE ${INFERTENSORRT_TRT_LIBRARIES})
endif()

set_target_properties(infertensorrt_cpp PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    # Keep the DLL/import-library name stable for downstream consumers.
    OUTPUT_NAME "infertensorrt_cpp"
)

# Generates include/infertensorrt_cpp/export.h (build tree) with
# INFERTENSORRT_EXPORT / INFERTENSORRT_NO_EXPORT macros. Annotate every public
# function or class you want visible from the DLL with INFERTENSORRT_EXPORT.
generate_export_header(infertensorrt_cpp
    BASE_NAME INFERTENSORRT
    EXPORT_MACRO_NAME INFERTENSORRT_EXPORT
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/infertensorrt_cpp/export.h"
)

# ---------------------------------------------------------------------------
# Optional executable for local development
# ---------------------------------------------------------------------------
# Copy the DLL next to a consumer executable so it can be run from the build tree.
function(infertensorrt_copy_runtime target)
    if(WIN32 AND INFERTENSORRT_BUILD_SHARED)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:infertensorrt_cpp>"
                "$<TARGET_FILE_DIR:${target}>"
            VERBATIM)
    endif()
endfunction()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    add_executable(infertensorrt_cpp_app src/main.cpp)
    target_link_libraries(infertensorrt_cpp_app PRIVATE infertensorrt_cpp)
    infertensorrt_copy_runtime(infertensorrt_cpp_app)
endif()

# ---------------------------------------------------------------------------
# Optional example targets
# ---------------------------------------------------------------------------
if(BUILD_EXAMPLES)
    file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp")
    foreach(example_src IN LISTS EXAMPLE_SOURCES)
        get_filename_component(example_name "${example_src}" NAME_WE)
        add_executable(${example_name} "${example_src}")
        target_link_libraries(${example_name} PRIVATE infertensorrt_cpp)
        infertensorrt_copy_runtime(${example_name})
    endforeach()
endif()

# ---------------------------------------------------------------------------
# TensorRT face detection sample (webcam + OpenCV)
# ---------------------------------------------------------------------------
if(BUILD_EXAMPLES AND TENSORRT_DIR)
    add_executable(tensorrtfacedetecttest
        "${CMAKE_CURRENT_SOURCE_DIR}/src/samples/tensorrtfacedetecttest.cpp"
        # TensorRTRunner is a private implementation detail and is not exported
        # from the DLL, so compile it into the sample directly.
        "${CMAKE_CURRENT_SOURCE_DIR}/src/tensorrt/tensorrtrunner.cpp"
        ${TRT_SAMPLES_SOURCES}
    )

    target_include_directories(tensorrtfacedetecttest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/tensorrt"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${TRT_SAMPLES_DIR}/common"
        "${TRT_SAMPLES_DIR}"
        "${TENSORRT_DIR}/include"
    )

    target_compile_definitions(tensorrtfacedetecttest PRIVATE
        TRT_BUILD_ONNX_PARSER=1
        "INFERTENSORRT_DEFAULT_MODEL_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/../models/version-RFB-640.onnx\""
    )

    target_link_libraries(tensorrtfacedetecttest PRIVATE
        CUDA::cudart
        ${INFERTENSORRT_TRT_LIBRARIES}
        ${OpenCV_LIBS}
    )
endif()

# ---------------------------------------------------------------------------
# Install / export
# ---------------------------------------------------------------------------
install(TARGETS infertensorrt_cpp
    EXPORT infertensorrt_cppTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
        # include/tensorrt is private implementation detail.
        PATTERN "tensorrt" EXCLUDE
    )
endif()

# Export header is generated, so install it explicitly.
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/infertensorrt_cpp/export.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/infertensorrt_cpp"
)

set(INFERTENSORRT_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/infertensorrt_cpp")

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/infertensorrt_cppConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(EXPORT infertensorrt_cppTargets
    FILE infertensorrt_cppConfig.cmake
    NAMESPACE infertensorrt::
    DESTINATION "${INFERTENSORRT_CMAKE_DIR}"
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/infertensorrt_cppConfigVersion.cmake"
    DESTINATION "${INFERTENSORRT_CMAKE_DIR}"
)
