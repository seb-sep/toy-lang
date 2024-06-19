set(LLVM_TARGET_DEFINITIONS ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCombineIncGen)