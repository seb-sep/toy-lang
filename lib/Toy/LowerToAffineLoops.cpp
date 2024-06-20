#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
using namespace mlir;

// convert RankedTensorType into MemRefType
static MemRefType convertTensorToMemRef(RankedTensorType t) {
    return MemRefType::get(t.getShape(), t.getElementType());
}

// the Location here tells you which block you're adding the alloc and dealloc to
static Value insertAllocAndDealloc(
    MemRefType t, Location loc, PatternRewriter &rewriter) {
    
    // create the alloc op
    auto alloc = rewriter.create<memref::AllocOp>(loc, t);

    // move the alloc to the beginning of its block
    auto *parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());

    // create the dealloc op
    auto dealloc = rewriter.create<memref::DeallocOp>(loc, t);

    // move the dealloc to the end of its block
    // why isn't this moveAfter() here?
    dealloc->moveBefore(&parentBlock->back());
    return alloc;

}

// Fn to process an iteration of a lowered loop.
// Loop iteration variable controls the iterations of a loop 
using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;
static void lowerOpToLoops(
    Operation *op, 
    ValueRange operands, 
    PatternRewriter &rewriter, 
    LoopIterationFn processIteration
) {
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    // rich location bs
    auto loc = op->getLoc();

    // allocation and deallocation for this op's result
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // create affine loop nest
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);
    // buildAffineLoopNest takes a callback which constructs the body of the innermost loop
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, tensorType.getShape(), steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
            // we generate the value to store by calling the loop iteration fn,
            // and store one value at each index
            Value valueToStore = processIteration(nestedBuilder, operands, ivs);
            nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc, ivs);
        }
    );

    rewriter.replaceOp(op, alloc);
}

// lowering toy.transpose to affine loop nest
struct TransposeOpLowering : public ConversionPattern {
    TransposeOpLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(toy::TransposeOp::getOperationName(), 1, ctx) {}

    mlir::LogicalResult matchAndRewrite(
        mlir::Operation *op,
        ArrayRef<mlir::Value> operands,
        mlir::ConversionPatternRewriter &rewriter
    ) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter,
            [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs) {
                // Adaptor auto-created from the ODS??? Lets us adapt the transpose op
                // to the memref operands
                toy::TransposeOpAdaptor transposeAdaptor(memRefOperands);
                Value input = transposeAdaptor.getInput();

                // Load the element by loading the transpose input in the reverse order
                SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                return builder.create<affine::AffineLoadOp>(loc, input, reverseIvs);
            }
        );
        return success();
    }
};


namespace {
// Binary op rewriting from toy to affine
// the template here means that the struct is fully generic
// on whatever BinaryOp and LoweredBinaryOp are
template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering: public ConversionPattern {
    BinaryOpLowering(MLIRContext *ctx) 
        // this syntax with the : is how you call the constructor of the parent class,
        // it just so happens in this case that it's all you need to do,
        // so the brackets are empty
        : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}
    
    LogicalResult matchAndRewrite(
        mlir::Operation *op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter
    ) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter,
            [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs) {
                typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

                // https://mlir.llvm.org/docs/Dialects/Affine/#affineload-affineaffineloadop
                auto loadedLhs = builder.create<affine::AffineLoadOp>(
                    loc, binaryAdaptor.getLhs(), loopIvs);
                auto loadedRhs = builder.create<affine::AffineLoadOp>(
                    loc, binaryAdaptor.getRhs(), loopIvs);
                
                // essentially, you're looping over each value in the TensorType,
                // and in the inner loop nest, loading the proper left and right hand values
                // with the indices from loopIvs, and then building the next proper 
                // op with those values (like an add)
                return builder.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
            });
        return success();
    }
};
} // namespace

// question: why can I instantiate the lowered op with any op I want?
// how do I know that toy can lower to arith AddOps?
using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;


// constant op lowering
// ConversionPatterns also accept additional operands to be replaced, RewritePatterns don't
// in this case, ConstantOp has no operands (only static attrs), so no need for ConversionPatterns
struct ConstantOpLowering : public OpRewritePattern<toy::ConstantOp> {
    using OpRewritePattern<toy::ConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(toy::ConstantOp op, PatternRewriter &rewriter) const final {
        DenseElementsAttr constantValue = op.getValue();
        Location loc = op->getLoc();

        // Assign the constant values to a memref allocation
        auto tensorType = llvm::cast<RankedTensorType>(op.getType());
        auto memRefType = convertTensorToMemRef(tensorType);
        auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

        // Generate index ops
        auto valueShape = memRefType.getShape();
        // constantIndices holds a range of index ops from 0 to the largest required idx
        SmallVector<Value, 8> constantIndices;

        if (!valueShape.empty()) {
            for (auto i : llvm::seq<int64_t>(0, *llvm::max_element(valueShape))) 
                constantIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, i));
        } else {
            constantIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
        }

        SmallVector<Value, 2> indices;
        auto valueIt = constantValue.value_begin<FloatAttr>();
        std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
            // only store elements when you've hit the base case of the recursion
            // Ex: you have a (2, 2, 2) tensor, the store happens after you've already
            // indexed into the first two values and you set to the last index
            if (dimension == valueShape.size()) {
                rewriter.create<affine::AffineStoreOp>(
                    loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
                    llvm::ArrayRef(indices));
                return;
            }

            // otherwise, recurse one dimension deeper
            for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
                // push and pop the current dimension you're on,
                // then move onto the next dimension (row) to store up
                indices.push_back(constantIndices[i]);
                storeElements(dimension + 1);
                indices.pop_back();
            }
        };

        // start storing at the zeroth dimension
        storeElements(0);

        // we put everything in the alloc op block, so we can replace this constant op with that block
        rewriter.replaceOp(op, alloc);
        return success();
    }
};

// toy func op to affine
struct FuncOpLowering : public OpConversionPattern<toy::FuncOp> {
    using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
        // we expect to only need to lower the main function
        if (op.getName() != "main")
            return failure();

        // Main should have no inputs and no results
        if (op.getNumArguments() != 0 || op.getFunctionType().getNumResults() != 0) {
            return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
                diag << "Expected 'main' to have 0 inputs and 0 results";
            });
        }

        // Create a new function using the func dialect, but same region
        auto func = rewriter.create<func::FuncOp>(
            op.getLoc(), op.getName(), op.getFunctionType());
        rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
        rewriter.eraseOp(op);
        return success();
    }
};

// toy return op lowering
struct ReturnOpLowering : public OpRewritePattern<toy::ReturnOp> {
    using OpRewritePattern<toy::ReturnOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(toy::ReturnOp op, PatternRewriter &rewriter) const final {
        // all fn calls should have been inlined, only main should remain
        if (op.hasOperand())
            return failure();

        rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
        return success();
    }
};

// toy transpose op lowering
struct TransposeOpLowering : public ConversionPattern {
    TransposeOpLowering(MLIRContext *ctx) 
        : ConversionPattern(toy::TransposeOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(
        Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
        
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter,
            [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs) {
                toy::TransposeOpAdaptor transposeAdaptor(memRefOperands);
                Value input = transposeAdaptor.getInput();

                SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                return builder.create<affine::AffineLoadOp>(loc, input, reverseIvs);
            });
        return success();
    }
};

// the actual toy lowering pass 
namespace {
struct ToyToAffineLoweringPass
    : public PassWrapper<ToyToAffineLoweringPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<
            affine::AffineDialect, 
            func::FuncDialect, 
            memref::MemRefDialect>();
    }

    void runOnOperation() final;

};
} // namespace 

// the actual lowering pass
void ToyToAffineLoweringPass::runOnOperation() {
    // instantiate the target for this lowering
    mlir::ConversionTarget target(getContext());

    // define the dialects we can legally lower to
    target.addLegalDialect<
        affine::AffineDialect,
        arith::ArithDialect,
        func::FuncDialect,
        memref::MemRefDialect
    >();

    // mark toy as illegal to catch lowerings that should have happened, but didn't
    // We do want toy.print to be excepted, as long as we update the operands
    target.addIllegalDialect<toy::ToyDialect>();
    target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
        // ensure that there are no tensor types left in print
        return llvm::none_of(op->getOperandTypes(), [](Type type) {
            return type.isa<TensorType>();
        });
    });

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<TransposeOpLowering>(&getContext());

    // attempt the conversion
    if (failed(
        applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();

}

