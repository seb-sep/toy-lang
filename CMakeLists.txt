
# Top level boilerplate
cmake_minimum_required(VERSION 3.20.0)
project(toy-lang LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# source config, find and configure MLIR and LLVM
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MLIR REQUIRED CONFIG)
  find_package(LLVM REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)


  set(LLVM_LINK_COMPONENTS
    Core
    Support
  )

  llvm_map_components_to_libnames(llvm_libs ${LLVM_LINK_COMPONENTS})
  message(STATUS "LLVM Libraries: ${llvm_libs}")

  # Include directories and definitions
  include_directories(${LLVM_INCLUDE_DIRS})
  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  add_definitions(${LLVM_DEFINITIONS_LIST})

else()
  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()

# Python shenanigans
if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

# LLVM and MLIR include directories
set(STANDALONE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(STANDALONE_BINARY_DIR ${PROJECT_BINARY_DIR})
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${STANDALONE_SOURCE_DIR}/include)
include_directories(${STANDALONE_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# add subdirectories
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(compiler)
add_subdirectory(mlir)

# toyc target
add_llvm_executable(toyc 
  compiler/toyc.cpp 
  compiler/AST.cpp
  mlir/Dialect.cpp
  mlir/MLIRGen.cpp
)

llvm_update_compile_flags(toyc)
message(STATUS "LLVM_LIBS: ${llvm_libs}")
target_include_directories(toyc PRIVATE ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(toyc PRIVATE
#   ${llvm_libs}
  
#   MLIRSupport

# )

set(MLIR_LIBRARIES
  MLIRIR
  MLIRParser
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRControlFlowInterfaces
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRFuncDialect
  MLIRFuncTransforms
)

target_link_libraries(toyc PRIVATE
  ${MLIR_LIBRARIES}
  ${llvm_libs}
  LLVMSupport
)

mlir_check_link_libraries(toyc)