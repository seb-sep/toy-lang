#include <typeinfo>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "Toy/Dialect.h"

using namespace mlir;
using namespace toy;

namespace {
#include "Toy/ToyCombine.inc"
} // namespace

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

OpFoldResult StructConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

OpFoldResult StructAccessOp::fold(FoldAdaptor adaptor) {
    // getInput is a getter on the input argument to the struct access op
    auto structAttr = llvm::dyn_cast_if_present<ArrayAttr>(adaptor.getInput());
    if (!structAttr)
        return nullptr; // must coerce this into a real result
    
    size_t elementIndex = getIndex();
    return structAttr[elementIndex];
}



struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
    SimplifyRedundantTranspose(mlir::MLIRContext *context) 
    : OpRewritePattern<TransposeOp>(context) {}

    mlir::LogicalResult matchAndRewrite(
        TransposeOp op, 
        mlir::PatternRewriter &rewriter
    ) const override {

        // get the transpose input
        mlir::Value transposeInput = op.getOperand();
        TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

        // We only want to match when the input to the transpose is another transpose
        if (!transposeInputOp)
            return failure();

        rewriter.replaceOp(op, {transposeInputOp.getOperand()});
        return success();
    }
};

/*
Canonicalization: picking one form out of many equivalents to be the 
CANONICAL FORM, rewrite all others into the canonical form
https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html
Makes later optimizations simpler

*/
void TransposeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context
) {
    results.add<SimplifyRedundantTranspose>(context);
}

void ReshapeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context
) {
    
    results.add<
        ReshapeReshapeOptPattern, 
        RedundantReshapeOptPattern, 
        FoldConstantOpReshapePattern>(context);
}