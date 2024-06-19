
#ifndef SHAPE_INFERENCE_INTERFACE
#define SHAPE_INFERENCE_INTERFACE

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
    let description = [{
        Interface to access a registered method to infer the return types for an
        operation that can be used during type inference.
    }];

    // more on how to define interfaces here:
    // https://mlir.llvm.org/docs/Interfaces/#attributeoperationtype-interfaces
    let methods = [
        InterfaceMethod<
            "Infer and set the shape for this op", 
            "void", 
            "inferShapes">
    ];
}

#endif // SHAPE_INFERENCE_INTERFACE