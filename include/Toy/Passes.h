#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace toy {
std::unique_ptr<Pass> createShapeInferencePass();
} // namespace toy
} // namespace mlir

#endif // TOY_PASSES_H
