
#include "mlir/Pass/Pass.h"
#include "Toy/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "Toy/ShapeInferenceInterface.h"
#include "Toy/Passes.h"

using namespace mlir;
using namespace toy;

// must be specified for LLVM_DEBUG use
#define DEBUG_TYPE "shape-inference"

#include "Toy/ShapeInferenceInterface.cpp.inc"

namespace {
struct ShapeInferencePass 
    : public mlir::PassWrapper<ShapeInferencePass, OperationPass<FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

    // The actual shape inference pass
    void runOnOperation() override {
        FuncOp function = getOperation();

        // Worklist of ops that require shape inference 
        llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
        // Op walk: map the given callback over each contained
        // op, region, or block, depending on the signature of the lambda
        // the [] is capture list, & inside means that all used can be
        // captured by reference
        function.walk([&](mlir::Operation *op) {
            if (ReturnsDynamicShape(op))
                opWorklist.insert(op);
        });

        // iterate through the worklist and infer all the shapes
        while (!opWorklist.empty()) {
            // we can only start with the operands for which
            // all the input operands are inferred
            auto nextOp = llvm::find_if(opWorklist, AllOperandsInferred);
            // SmallPtrSet::end() is not an actual element, it's a special value indicating
            // that the set is empty
            if (nextOp == opWorklist.end())
                break;

            // remove the operation from the worklist
            Operation *op = *nextOp;
            opWorklist.erase(op);

            // Infer the output shape for the next op
            LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");
            // Remember that MLIR interfaces are CPP classes, not CPP interfaces
            if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
                shapeOp.inferShapes();
            } else {
                op->emitError("shape inference interface not implemented on this op");
                return signalPassFailure();
            }

        }

        // if the worklist is somehow not empty, we failed
        if (!opWorklist.empty()) {
            function.emitError("Shape inference failed, ") 
                << opWorklist.size() << " ops couldn't be inferred\n";
            return signalPassFailure();
        }
    }

    // Are all the operands of the op inferred?
    static bool AllOperandsInferred(Operation *op) {
        // apply the inner predicat of is a ranked tensor type
        // to all the operand types of the op
        // If they're a ranked tensor type, they're inferred
        return llvm::all_of(op->getOperandTypes(), [](Type operandType) {
            return llvm::isa<RankedTensorType>(operandType);
        });
    }

    // Does the op have a dynamically shaped result?
    static bool ReturnsDynamicShape(Operation *op) {
        // similar to the above, check if any of the result types
        // are not a tensor type
        return llvm::any_of(op->getResultTypes(), [](Type resultType) {
            return !llvm::isa<RankedTensorType>(resultType);
        });
    }

};

} // namespace

// helper for instantiating the pass
std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
    return std::make_unique<ShapeInferencePass>();
}