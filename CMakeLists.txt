cmake_minimum_required(VERSION 3.20)
project(ll_llm VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# Build Options
# =============================================================================
option(LLLLM_BUILD_TESTS "Build test targets" OFF)
option(LLLLM_ENABLE_AVX2 "Enable AVX2 SIMD instructions" ON)

# =============================================================================
# Compiler Flags
# =============================================================================
if(MSVC)
    add_compile_options(/W4 /O2 /EHsc)
    if(LLLLM_ENABLE_AVX2)
        add_compile_options(/arch:AVX2)
    endif()
else()
    add_compile_options(-Wall -Wextra -O2)
    if(LLLLM_ENABLE_AVX2)
        add_compile_options(-mavx2 -mfma)
    endif()
endif()

# =============================================================================
# Fetch llama.cpp
# =============================================================================
include(FetchContent)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)

# llama.cpp build options — CPU only, no GPU backends
set(GGML_CUDA    OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN  OFF CACHE BOOL "" FORCE)
set(GGML_METAL   OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)

message(STATUS "Fetching llama.cpp (this may take a moment on first build)...")
FetchContent_MakeAvailable(llama_cpp)

# =============================================================================
# Main Executable
# =============================================================================
add_executable(ll_llm
    src/main.cpp
)

target_include_directories(ll_llm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
)

target_link_libraries(ll_llm PRIVATE
    llama
    ggml
)

# On Windows, link psapi for GetProcessMemoryInfo
if(WIN32)
    target_link_libraries(ll_llm PRIVATE psapi)
endif()

# =============================================================================
# Install
# =============================================================================
install(TARGETS ll_llm RUNTIME DESTINATION bin)

message(STATUS "")
message(STATUS "=== LL_LLM Configuration ===")
message(STATUS "  C++ Standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  AVX2 Enabled:  ${LLLLM_ENABLE_AVX2}")
message(STATUS "  Build Type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "============================")
message(STATUS "")
