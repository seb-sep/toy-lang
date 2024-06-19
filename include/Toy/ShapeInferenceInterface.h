#ifndef SHAPEINFERENCEINTERFACE_H
#define SHAPEINFERENCEINTERFACE_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace toy {
    #include "Toy/ShapeInferenceOpInterfaces.h.inc"
} // namespace toy
} // namespace mlir

#endif // SHAPEINFERENCEINTERFACE_H