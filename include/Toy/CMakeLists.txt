# Look into what add_mlir_dialect has to do with the ops.td tablegen
# add_mlir_dialect(Toy toy)
# add_mlir_doc(ToyDialect ToyDialect Toy/ -gen-dialect-doc)
# add_mlir_doc(ToyOps ToyOps Toy/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ToyIncGen)

# Look into what add_mlir_interfaces() has to do with this
set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
# add_public_tablegen_target(ShapeInferenceInterfaceIncGen)
add_mlir_interface(ShapeInferenceInterface)