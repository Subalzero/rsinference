cmake_minimum_required(VERSION 3.16)

project(infertensorrt_cpp VERSION 0.1.0 LANGUAGES CXX)

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
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
list(REMOVE_ITEM INFERTENSORRT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

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
# Library (DLL on Windows, .so/.dylib elsewhere)
# ---------------------------------------------------------------------------
add_library(infertensorrt_cpp ${INFERTENSORRT_LIBRARY_TYPE})
add_library(infertensorrt::infertensorrt_cpp ALIAS infertensorrt_cpp)

target_sources(infertensorrt_cpp PRIVATE
    ${INFERTENSORRT_SOURCES}
    ${INFERTENSORRT_PUBLIC_HEADERS}
)

target_include_directories(infertensorrt_cpp
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

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
