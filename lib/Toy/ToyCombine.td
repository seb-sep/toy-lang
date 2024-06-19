
#ifndef TOY_COMBINE
#define TOY_COMBINE

include "mlir/IR/PatternBase.td"
include "Toy/Ops.td"

// clear out multiple reshapes
// This can be done because a reshape defines the target shape, 
// so the shape is always the last reshape
def ReshapeReshapeOptPattern : Pat<
    (ReshapeOp(ReshapeOp $arg)),
    (ReshapeOp $arg)
>;


// def DumbPattern : Pat<
//     (ReshapeOp $arg),
//     (replaceWithValue $arg)
// >;

// don't reshape a tensor to the same shape
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
    (ReshapeOp:$res $arg), 
    (replaceWithValue $arg), 
    [(TypesAreIdentical $res, $arg)]
>;

// when reshaping a constant, just instantiate the constant in the right shape
// def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;
def ReshapeConstant : NativeCodeCall<"$0.reshape(::mlir::cast<mlir::ShapedType>($1.getType()))">;
def FoldConstantOpReshapePattern : Pat<
    (ReshapeOp:$res (ConstantOp $arg)),
    (ConstantOp (ReshapeConstant $arg, $res))
>;

#endif // TOY_COMBINE