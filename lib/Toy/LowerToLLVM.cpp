#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "Toy/Dialect.h"


using namespace mlir;

namespace {
class PrintOpLowering : public ConversionPattern {

private:

    static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
        auto llvmI32Ty = IntegerType::get(context, 32);
        auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
        auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy, true);
        return llvmFnType;
    }
    static FlatSymbolRefAttr getOrInsertPrintf(
        PatternRewriter &rewriter, 
        ModuleOp module,
        LLVM::LLVMDialect *llvmDialect
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
};
} // namespace 