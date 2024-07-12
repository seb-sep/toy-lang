//===- toyc.cpp - The Toy Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "Toy/AST.h"
#include "Toy/Dialect.h"
#include "Toy/Lexer.h"
#include "Toy/MLIRGen.h"
#include "Toy/Parser.h"
#include "Toy/Passes.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/IR/Module.h"
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <iostream>

using namespace toy;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Toy, MLIR };
} // namespace
static cl::opt<enum InputType> inputType(
    "x", cl::init(Toy), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Toy, "toy", "load the input file as a Toy source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

namespace {
enum Action { None, DumpAST, DumpMLIR, DumpMLIRAffine, DumpMLIRLLVM, DumpLLVMIR, JitLLVM };
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine", "output the MLIR dump affine lowering")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm", "output the MLIR dump LLVM lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR lowering dump")),
    cl::values(clEnumValN(JitLLVM, "jit-llvm", "JIT compile and execute with the LLVM execution engine")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Handle '.toy' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).ends_with(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    module = mlirGen(context, *moduleAST);
    return !module ? 1 : 0;
  }

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

// Instantiate and run passes on the MLIR module.
llvm::Error loadAndProcessMLIR(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module) {
    // mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadMLIR(context, module))
    return llvm::make_error<llvm::StringError>(
      llvm::formatv("failed to load MLIR: {0}\n", error),
      llvm::inconvertibleErrorCode());

  mlir::PassManager pm(module.get()->getName());
  // apply generic pass manager cli options
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return llvm::make_error<llvm::StringError>("Failed to apply MLIR PM CLI options", llvm::inconvertibleErrorCode());

  bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
  bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;
  if (enableOpt || isLoweringToAffine) {

    // Inline all functions into main and then delete them.
    pm.addPass(mlir::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations.
    mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
    optPM.addPass(mlir::toy::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

  }

  if (isLoweringToAffine) {
    pm.addPass(mlir::toy::createLowerToAffinePass());

    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    if (enableOpt) {
      // this api of builder thunks returning pointers to a pass is pretty standard
      optPM.addPass(mlir::affine::createLoopFusionPass());
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    }
  }

  if (isLoweringToLLVM) {
    pm.addPass(mlir::toy::createLowerToLLVMPass());

    // necessary for debugging?
    pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  }
  
  if (mlir::failed(pm.run(*module)))
    return llvm::make_error<llvm::StringError>("Failed to run passes", llvm::inconvertibleErrorCode());

  return llvm::Error::success();
}

int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a Toy AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

// return the (potentially optimized) LLVM as an llvm module
llvm::Expected<std::unique_ptr<llvm::Module>> processLLVM(mlir::ModuleOp module) {
  // register the translation to LLVM with the mlir context
  // mlir::registerBuiltinDialectTranslation(*module->getContext());
  // mlir::registerLLVMDialectTranslation(*module->getContext());

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    return llvm::make_error<llvm::StringError>("failed to emit LLVM IR", llvm::inconvertibleErrorCode());
  }

  // from now on, we're in LLVM land
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    return llvm::make_error<llvm::StringError>("could not create LLVM TargetMachine Builder", llvm::inconvertibleErrorCode());
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    return llvm::make_error<llvm::StringError>("could not create TargetMachine", llvm::inconvertibleErrorCode());
  }
  // llvm::get() is kind of like rust's unwrap on an llvm expected
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(), tmOrError.get().get());

  // optional optimzier
  auto optPipeline = mlir::makeOptimizingTransformer(enableOpt ? 3 : 0, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    return llvm::make_error<llvm::StringError>(
      llvm::formatv("failed to call the optimizing pipeline: {0}\n", err),
      llvm::inconvertibleErrorCode());
  }

  // use std::move to wrap in a unique pointer
  // I guess this signature auto wraps in an llvm expected?
  return std::move(llvmModule);
}

int runLLVMJIT(mlir::ModuleOp module) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // register translations to LLVM IR
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());
  
  // register a transformer fn running passes over the LLVMIR
  auto optPipeline = mlir::makeOptimizingTransformer(enableOpt ? 3 : 0, 0, nullptr);
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;

  // create an execution engine with the options
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  if (!maybeEngine) {
    llvm::errs() << "failed to create an execution engine\n";
    return -1;
  }
  auto &engine = maybeEngine.get(); // remember that get 'unwraps' the llvm expected

  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "error invoking the main fn\n";
    return -1;
  }

  return 0;
}

int main(int argc, char **argv) {

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  // if we're dumping ast, no need to produce mlir module
  if (emitAction == Action::DumpAST) {
    return dumpAST();
  } else {
    // otherwise, we process the source into an MLIR module op which we reuse
    // across all compiler options

    // setup mlir context and register toy dialect
    mlir::DialectRegistry registry;
    mlir::func::registerAllExtensions(registry);
    mlir::MLIRContext context(registry);
    context.getOrLoadDialect<mlir::toy::ToyDialect>();

    // lifetime of the module op is tied ot that of the context, so once the context
    // goes out of scope, the module is destroyed
    // If you literally instantiate the module before the context, segfault on return
    // Think of the stack: last allocated item destroyed first when goes out of scope
    // So, need to destroy module before context
    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (auto err = loadAndProcessMLIR(context, module)) {
      llvm::handleAllErrors(std::move(err), [&](const llvm::ErrorInfoBase &error) {
        llvm::errs() << "Failed to load and process MLIR: " << error.message() << "\n";
      });
      return 1;
    }
    
    
    switch (emitAction) {
    case Action::DumpMLIR:
    case Action::DumpMLIRAffine:
    case Action::DumpMLIRLLVM: {
      std::cout << "dumping mlir \n";
      module->dump();
      std::cout << "dumped \n";
      return 0;
    }
    case Action::DumpLLVMIR: {
      auto expectedLLVM = processLLVM(*module);
      if (!expectedLLVM) {
        llvm::handleAllErrors(expectedLLVM.takeError(), [&](const llvm::ErrorInfoBase &error) {
          llvm::errs() << "Failed to process into LLVM: " << error.message() << "\n";
        });
        return 1;
      }
      expectedLLVM.get()->dump();
      return 0;
    }
    case Action::JitLLVM:
      return runLLVMJIT(*module);
    default:
      llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    }
  }
  return 0;
}