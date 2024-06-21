
#ifndef TOY_OPS
#define TOY_OPS

// All op interfaces are written in tablegen, including the in-tree ones
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "Toy/ShapeInferenceInterface.td"

def Toy_Dialect: Dialect {
    let name = "toy";
    let summary = "A high level dialect for the Toy language";
    let cppNamespace = "::mlir::toy";
}

class Toy_Op<string mnemonic, list<Trait> traits = []>:
    Op<Toy_Dialect, mnemonic, traits>;

// Pure trait is an alias for speculative execution and no memory use traits:
// https://www.jeremykun.com/2023/09/07/mlir-using-traits/
def ConstantOp: Toy_Op<"constant", 
    [Pure]> {

    let summary = "constant operation";
    let description = [{
        Constant operation turns a literal into an SSA value. The data is attached
        to the operation as an attribute. For example:

        %0 = "toy.constant"()
            { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
            : () -> tensor<2x3xf64>
    }];

    // remember what an attribute is: a constant you can pass to an op between the brackets,
    // it is STATIC
    // we can't then use the constant op with a runtime value
    // the use of the attr makes it an attribute 
    let arguments = (ins F64ElementsAttr:$value);
    let results = (outs F64Tensor);

    // creates a mlir::LogicalResult::verify() stub in the corresponding generated c++ file
    let hasVerifier = 1;


    let builders = [
        OpBuilder<(ins "DenseElementsAttr":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>, // with the blob is codegenned in the .cpp.inc file
        OpBuilder<(ins "double":$value)> // without a code blob means we must implement ourselves
    ];

    let hasCustomAssemblyFormat = 1;
}

def AddOp: Toy_Op<"add", 
    [DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "element wise addition";
    let description = [{
        The "add" operation performs element-wise addition between two tensors.
        The shapes of the tensor operands are expected to match.
    }];

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];

    let hasCustomAssemblyFormat = 1;
}


// isolatedfromabove trait means that it does not capture values
// from the greater scope
// https://mlir.llvm.org/docs/Traits/#isolatedfromabove
def FuncOp: Toy_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
]> {

    let description = [{
        The "toy.func" operation represents a user defined function. These are
        callable SSA-region operations that contain toy computations.

        Example:

        ```mlir
        toy.func @main() {
        %0 = toy.constant dense<5.500000e+00> : tensor<f64>
        %1 = toy.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
        toy.print %1 : tensor<2x2xf64>
        toy.return
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name, // the name of the fn, pass as a symbol attr
        TypeAttrOf<FunctionType>:$function_type, //type of function ins and outs?
        OptionalAttr<DictArrayAttr>:$arg_attrs, 
        OptionalAttr<DictArrayAttr>:$res_attrs
    );

    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
        "StringRef":$name, "FunctionType":$type,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs // CArg is for a default value
    )>];

    // I guess extraClassDeclaration lets us add arbitrary cpp code to the class?
    let extraClassDeclaration = [{

        // implement FunctionOpInterface
        // functions have arg and result types, and a body, so we just define getters

        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

        Region *getCallableRegion() { return &getBody(); }
    }];

    // https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/
    // when do we not want to define the default builders?
    let skipDefaultBuilders = 1;

    let hasCustomAssemblyFormat = 1;
}

def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
    
    // we need a generic verison of call because we only sometimes need to do
    // shape inference
    let description = [{
        Generic calls represent calls to a user defined function that needs to
        be specialized for the shape of its arguments. The callee name is attached
        as a symbol reference via an attribute. The arguments list must match the
        arguments expected by the callee. For example:

        ```mlir
        %4 = toy.generic_call @my_func(%1, %3)
            : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
        ```

        This is only valid if a function named "my_func" exists and takes two
        arguments.
    }];

    // we're allowed to define the arguments like this because
    // the only type in Toy is an f64tensor, so all args will be a list of tensors
    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);

    let results = (outs F64Tensor);

    let builders = [
        OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
    ];

    let assemblyFormat = [{
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
    }];

}

def MulOp : Toy_Op<"mul",
    [DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
} 

def PrintOp : Toy_Op<"print"> {
    
    let description = [{
        The "print" builtin operation prints a given input tensor, and produces
        no results.
    }];
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);

    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def ReshapeOp : Toy_Op<"reshape"> {
    let description = [{
        Reshape operation is transforming its input tensor into a new tensor with
        the same number of elements but different shapes. For example:

        ```mlir
        %0 = toy.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
        ```
    }];

    let arguments = (ins F64Tensor:$input);

    // must be reshaping to a statically known type, 
    // the list in StaticShapeTensorOf is a list of allowed types to be in said tensor
    let results = (outs StaticShapeTensorOf<[F64]>);

    let hasCanonicalizer = 1;

    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];
}

def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
    let description = [{
        The "return" operation represents a return operation within a function.
        The operation takes an optional tensor operand and produces no results.
        The operand type must match the signature of the function that contains
        the operation. For example:

        ```mlir
        toy.func @foo() -> tensor<2xf64> {
            ...
            toy.return %0 : tensor<2xf64>
        }
        ```
  }];

  let arguments = (ins Variadic<F64Tensor>:$input);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let hasVerifier = 1;

  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";
}

def TransposeOp : Toy_Op<"transpose", 
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "transpose operation";

    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor);

    let builders = [
        OpBuilder<(ins "Value":$input)>
    ];

    let hasVerifier = 1;

    // when this is defined, we need an implementation of getCanonicalizationPatterns()
    // with the patterns we want to be added
    // https://mlir.llvm.org/docs/Canonicalization/
    let hasCanonicalizer = 1;

    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];
}

def CastOp : Toy_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    Pure,
    SameOperandsAndResultShape]
> {
    let description = [{
        The "cast" operation converts a tensor from one type to an equivalent type
        without changing any data elements. The source and destination types
        must both be tensor types with the same element type. If both are ranked,
        then shape is required to match. The operation is invalid if converting
        to a mismatching constant dimension.
    }];

    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor:$output);
    let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

}

// Adding the comment of TOY_OPS here is actually necessary to match with the #ifndef at the top
#endif // TOY_OPS