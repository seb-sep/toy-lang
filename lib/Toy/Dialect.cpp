//===- Dialect.cpp - Toy IR Dialect registration in MLIR ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the dialect for the Toy IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "Toy/Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/Hashing.h"
#include <algorithm>
#include <string>
#include <cstdint>

using namespace mlir;
using namespace mlir::toy;

#include "Toy/Dialect.cpp.inc"


// ToyInliner
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;


  // All call ops can be inlined
  bool isLegalToInline(
    Operation *call, 
    Operation *callable, 
    bool wouldBeCloned
  ) const final {
    return true;
  }


  // All ops withing a toy region can be inlined
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  // All regions (toy fns) can be inlined
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned, IRMapping &valueMapping) const final {
    return true;
  }

  void handleTerminator(
    Operation *op,
    ValueRange valuesToRepl
  ) const final {
    auto returnOp = mlir::cast<ReturnOp>(op);

    assert(returnOp.getNumOperands() == valuesToRepl.size());

    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  // I guess whenever there's a type mismatch between the value and result type,
  // seeks to inline an op which takes in only the input value and produces 
  // a single output value of the expected type
  Operation *materializeCallConversion(
    OpBuilder &builder, 
    Value input,
    Type resultType,
    Location conversionLoc
  ) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};

//===----------------------------------------------------------------------===//
// ToyDialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Toy/Ops.cpp.inc"
      >();
  
  // Register the toy inliner
  addInterfaces<ToyInlinerInterface>();
  addTypes<StructType>();
}

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}

static mlir::LogicalResult verifyConstantForType(
  mlir::Type type, 
  mlir::Attribute opaqueValue, 
  mlir::Operation *op
) {
  if (llvm::isa<TensorType>(type)) {
    auto attrValue = llvm::dyn_cast<mlir::DenseFPElementsAttr>(opaqueValue);
    if (!attrValue)
      return op->emitOpError("constant of TensorType must have a value" 
                          "of DenseElementsAttr, got ")
        << opaqueValue;
      
    // if the tensor shape is unranked, nothing to verify
    auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(type);
    if (!resultType)
      return success();
    
    // otherwise, it's ranked, so we have to make sure shapes match
    auto attrType = llvm::cast<mlir::RankedTensorType>(attrValue.getType());
    if (attrType.getRank() != resultType.getRank()) 
      return op->emitOpError("Rank of return type must match the rank of attribute: ")
        << resultType.getRank() << " != " << attrType.getRank();

    // then, check that each dim matches
    for (int dim=0, dimE = attrType.getRank(); dim<dimE; dim++) {
      if (attrType.getShape()[dim] != resultType.getShape()[dim])
        return op->emitOpError("Dimension of attribute ") 
          << attrType.getShape()[dim] << " mismatches dim of return "
          << resultType.getShape()[dim] << " at dimension " << dim << " \n";
    }

    return mlir::success();
  }

  // if the type is not a tensor, then it's a struct
  auto resultType = llvm::cast<StructType>(type);
  llvm::ArrayRef<mlir::Type> resultElementTypes = resultType.getElementTypes();

  // if it's a struct, attribute for the constant struct must be an array
  auto attrValue = llvm::dyn_cast<ArrayAttr>(opaqueValue);
  if (!attrValue || attrValue.getValue().size() != resultElementTypes.size())
    return op->emitOpError("constant struct must with ") << resultElementTypes.size()
      << " elements was initialized with ArrayAttr of " << attrValue.getValue().size() 
      << " elements\n";
  
  // verify each element of struct
  llvm::ArrayRef<mlir::Attribute> attrElementValues = attrValue.getValue();
  for (const auto it : llvm::zip(resultElementTypes, attrElementValues)) {
    // compare the result and attr element for each in struct
    if (failed(verifyConstantForType(std::get<0>(it), std::get<1>(it), op)))
      return mlir::failure();
  }
  return mlir::success();


}

/// Verifier for the constant operation. This corresponds to the
/// `let hasVerifier = 1` in the op definition.
mlir::LogicalResult ConstantOp::verify() {
  // we have a this context because this is a method on constantop
  return verifyConstantForType(getResult().getType(), getValue(), *this);
}

mlir::LogicalResult StructConstantOp::verify() {
  // we have a this context because this is a method on constantop
  return verifyConstantForType(getResult().getType(), getValue(), *this);
}



//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

void AddOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments) {
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee",
                     mlir::SymbolRefAttr::get(builder.getContext(), callee));
}


// get the callable func from within the op instance
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
    // -> operator is an overloadable field access operator 
    // for example, -> could dispatch to a struct you're wrapping
    // Remember: & gets an address, * dereferences, all can be overrode
    return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

// set the callable func for this call op
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

// when in doubt of what's in your toolkit, check your .inc files and the docs
Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

MutableOperandRange GenericCallOp::getArgOperandsMutable() { return getInputsMutable(); }


//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void MulOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

// set the type of the result of this op
void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult ReturnOp::verify() {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>((*this)->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!hasOperand())
    return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType || llvm::isa<mlir::UnrankedTensorType>(inputType) ||
      llvm::isa<mlir::UnrankedTensorType>(resultType))
    return mlir::success();

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resultType
                     << ")";
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

mlir::LogicalResult TransposeOp::verify() {
  // if the cast fails, one is not a tensor type or is just unranked, so null returned
  auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
  if (!inputType || !resultType) 
    return mlir::success();

  auto inputShape = inputType.getShape();
  if (!std::equal(inputShape.begin(), inputShape.end(),
                  resultType.getShape().rbegin())) { // rbegin is a reverse iterator
    return emitError()
           << "expected result shape to be a transpose of the input";
  }
  return mlir::success();
}

void TransposeOp::inferShapes() { 
  // get the ranked tensor type
  auto inputType = llvm::cast<RankedTensorType>(getOperand().getType());
  SmallVector<int64_t, 2> dims(llvm::reverse(inputType.getShape()));
  getResult().setType(RankedTensorType::get(dims, inputType.getElementType()));
}

// CastOp
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  // in our toy lang, we only cast one value at a time
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;

  // attempt to cast inputs to TensorType
  TensorType input = dyn_cast<TensorType>(inputs.front());
  TensorType output = dyn_cast<TensorType>(outputs.front());
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;

  // return false only if two ranked tensors of different rank
  return !input.hasRank() || !output.hasRank() || input == output;
  
}

void CastOp::inferShapes() { getResult().setType(getInput().getType()); }

// StructAccessOp
void StructAccessOp::build(
  mlir::OpBuilder &b, 
  mlir::OperationState &state, 
  // remmeber, these two match what we specified in ODS
  mlir::Value input, size_t index
) {
  // the return type can be taken from the input
  StructType structTy = llvm::cast<StructType>(input.getType());
  assert(index < structTy.getNumElementTypes());
  mlir::Type resultType = structTy.getElementTypes()[index];

  // build is autogenerated
  build(b, state, resultType, input, b.getI64IntegerAttr(index));
}

mlir::LogicalResult StructAccessOp::verify() {
  StructType structTy = llvm::cast<StructType>(getInput().getType());
  size_t indexValue = getIndex();
  if (indexValue >= structTy.getNumElementTypes())
    return emitOpError() << "index " << indexValue 
      << " must be less than num of elements " << structTy.getNumElementTypes() << "\n";
  
  // we can assume this exists because we already have the op
  mlir::Type resultType = getResult().getType();
  if (resultType != structTy.getElementTypes()[indexValue])
    return emitOpError() << "type of accessed value must match that in the struct\n";

  return mlir::success();
}


// Toy types

namespace mlir {
namespace toy {
namespace detail {

// Internal storage struct to hold the data of the Toy struct type.
struct StructTypeStorage : public mlir::TypeStorage {

  // used to unique instances of the type
  // each instances is uniqued by the elements it contains
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  llvm::ArrayRef<mlir::Type> elementTypes;

  // c++ constructor which just sets the value of elementTypes to what's passed
  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
    : elementTypes(elementTypes) {}

  // so you can == on llvm array refs?
  // so this is implementing == on StructTypeStorage with a KeyTy
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  // constructor for a new storage instance
  // The allocator MUST be used for dynamic allocations used to create type storage
  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {

    // copy the elements into the allocator
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    // This new () Struct() syntax means that the new StructTypeStorage
    // goes in the memory of (allocator.allocate())
    return new (allocator.allocate<StructTypeStorage>()) StructTypeStorage(elementTypes);
  }

};

} // namespace detail
} // namespace toy
} // namespace mlir

StructType StructType::get(llvm::ArrayRef<mlir::Type> elementTypes) {
  assert(!elementTypes.empty() && "expected at least 1 element type");

  // typebase get gives us a uniqued instance of the type???
  // need the right context to unique in
  // elements from the constructor go to the new instance
  mlir::MLIRContext *ctx = elementTypes.front().getContext();
  return Base::get(ctx, elementTypes);
}

llvm::ArrayRef<mlir::Type> StructType::getElementTypes() {
  // we get the getImpl() fn from the typebase, it knows
  // to hook up to our storage struct
  return getImpl()->elementTypes;
}

// parse the type
mlir::Type ToyDialect::parseType(mlir::DialectAsmParser &parser) const {
  // struct-type ::= `struct` `<` type (`,` type)* `>`

  // should this be !(parse struct || parse <)?
  // No it shold not, because mlir::logicalresult is truthy on FAILURE
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();

  // parse struct elements
  SmallVector<mlir::Type, 1> elementTypes;
  do {
    // having the location when we parse the next token is helpful for errors ig
    SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (!parser.parseType(elementType))
      return nullptr;

    // better be a tensor or struct
    if (!llvm::isa<TensorType, StructType>(elementType)) {
      parser.emitError(typeLoc, "element must be either a struct or tensor type, got: ")
        << elementType;
      return Type();
    }
    elementTypes.push_back(elementType);

  } while (succeeded(parser.parseOptionalComma()));

  // parse the >
  if (parser.parseGreater())
    return Type();
  return StructType::get(elementTypes);
}

void ToyDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
  // only type is struct
  StructType structType = llvm::cast<StructType>(type);

  printer << "struct<";
  // interleave comma iterates a function (printer) over a collection with a comma in between
  llvm::interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
}

// we're defining a type, so we inherit from mlir::TypeBase
// Derived types in MLIR are templated on concrete type, type base class, and storage class
//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Toy/Ops.cpp.inc"