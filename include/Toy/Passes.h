#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace toy {

std::unique_ptr<Pass> createShapeInferencePass();

std::unique_ptr<Pass> createLowerToAffinePass();

std::unique_ptr<Pass> createLowerToLLVMPass();
} // namespace toy
} // namespace mlir

#endif // TOY_PASSES_H
