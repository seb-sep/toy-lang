#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"

#include "Toy/Dialect.h"
#include "Toy/Passes.h"


using namespace mlir;

namespace {
class PrintOpLowering : public ConversionPattern {

public:
    // the explicit keyword prevents compiler from doing explicit type conversion w it
    explicit PrintOpLowering(MLIRContext *context)
        : ConversionPattern(toy::PrintOp::getOperationName(), 1, context) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        auto *context = rewriter.getContext();
        // we know that at this point, we should be passing memrefs only to print,
        // so we're okay with panicking here
        auto memRefType = llvm::cast<MemRefType>((*op->operand_type_begin()));
        auto memRefShape = memRefType.getShape();
        auto loc = op->getLoc();

        ModuleOp parentModule = op->getParentOfType<ModuleOp>();

        // get a ref to the printf fn
        auto printfRef = getOrInsertPrintf(rewriter, parentModule);
        Value formatSpecifierCst = getOrCreateGlobalString(
            loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
        Value newlineCst = getOrCreateGlobalString(
            loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);
        
        // loop over each dimension in the shape
        SmallVector<Value, 4> loopIvs;
        for (unsigned i = 0, e = memRefShape.size(); i!=e; ++i) {
            // lower and upper bounds are arith index ops
            auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
            auto upperBound = rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
            auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

            // so the sct for op takes in ops from arith
            auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

            for (Operation &nested : *loop.getBody())
                rewriter.eraseOp(&nested);
            // push back one value for each dimension???
            loopIvs.push_back(loop.getInductionVar());

            // set insertion point to the end of the loop block
            rewriter.setInsertionPointToEnd(loop.getBody());
            // call printf newline at the end of each inner loop
            if (i != e - 1)
                // an LLVM call op requires the function type, the function ref, and the value to call on
                rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef, newlineCst);
            rewriter.create<scf::YieldOp>(loc);
            // why do we need to move to the beginnign of the loop now?
            rewriter.setInsertionPointToStart(loop.getBody());
        }

        auto printOp = cast<toy::PrintOp>(op);
        // memref load needs an mlir value and loop values (likely from scf)??
        auto elementLoad = rewriter.create<memref::LoadOp>(loc, printOp.getInput(), loopIvs);
        // perhaps elementLoad is actually the element to print
        rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef, ArrayRef<Value>({formatSpecifierCst, elementLoad}));

        // we already created the llvm call, so remove the toy print that was there before
        rewriter.eraseOp(op);
        return success();
    }


private:

    static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
        auto llvmI32Ty = IntegerType::get(context, 32);
        auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
        auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy, true);
        return llvmFnType;
    }
    static FlatSymbolRefAttr getOrInsertPrintf(
        PatternRewriter &rewriter, 
        ModuleOp module
    ) {
        auto *context = module.getContext();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
            return SymbolRefAttr::get(context, "printf");

        // insert the printf
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", getPrintfType(context));
        return SymbolRefAttr::get(context, "printf");

    }

    // represent an access into a global string of the given name
    static Value getOrCreateGlobalString(
        Location loc, OpBuilder &builder, StringRef name, StringRef value, ModuleOp module) {
        
        LLVM::GlobalOp global;
        // if the global string does not exist, build it
        if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
            // RAII that resets the insertion point of the op builder when this is destroyed
            OpBuilder::InsertionGuard insertGuard(builder);
            builder.setInsertionPointToStart(module.getBody());
            auto type = LLVM::LLVMArrayType::get(
                IntegerType::get(builder.getContext(), 8), value.size());
            global = builder.create<LLVM::GlobalOp>(
                loc, type, true, LLVM::Linkage::Internal, name, builder.getStringAttr(value), 0);
        }

        // get the pointer to the first character
        Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
        Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(), builder.getIndexAttr(0));

        // for the llvm::getelementptr operation
        return builder.create<LLVM::GEPOp>(
            loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
            globalPtr, ArrayRef<Value>({cst0, cst0}));
    }
};
} // namespace 

namespace {
struct ToyToLLVMLoweringPass
    : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>> {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToLLVMLoweringPass)

        void getDependentDialects(DialectRegistry &registry) const override {
            registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
        }

        void runOnOperation() final;
};
} // namespace

void ToyToLLVMLoweringPass::runOnOperation() {
    LLVMConversionTarget target(getContext());
    // target.addLegalDialect<mlir::LLVMDialect>();
    target.addLegalOp<mlir::ModuleOp>();


    // Type converter to lower memrefs into their LLVM equivalents
    LLVMTypeConverter typeConverter(&getContext());

    // populate the patterns used for lowering affine and std dialects
    RewritePatternSet patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    // finally, add the print op lowering we made
    patterns.add<PrintOpLowering>(&getContext());

    // ensure you convert everything
    auto module = getOperation();
    // i guess the move here makes the pattern set frozen???
    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::toy::createLowerToLLVMPass() {
    return std::make_unique<ToyToLLVMLoweringPass>();
}