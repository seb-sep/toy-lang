# # Add source files for the parser
# set(PARSER_SOURCES
#   AST.cpp
#   toyc.cpp
# )

# find_package(LLVM REQUIRED CONFIG)

# message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
# message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")


# include_directories(${LLVM_INCLUDE_DIRS})
# separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
# add_definitions(${LLVM_DEFINITIONS_LIST})


# # Create a library for the parser
# add_library(Parser ${PARSER_SOURCES})
# target_include_directories(Parser PRIVATE ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(Parser PRIVATE
#   MLIRSupport
#   ${llvm_libs}
# )

# # Create the Toy compiler executable
# message(STATUS "llvm_libs: ${llvm_libs}")
# add_executable(toyc toyc.cpp)
# target_include_directories(toyc PRIVATE ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(toyc 
#   Parser
#   MLIRSupport
#   ${llvm_libs}
# )

# get_target_property(TOYC_LINK_LIBS toyc INTERFACE_LINK_LIBRARIES)
# message(STATUS "Libraries linked to toyc: ${TOYC_LINK_LIBS}")
# # Add the executable to the custom target
# # add_dependencies(ToyProject toyc)
