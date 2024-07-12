toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00]]> : tensor<1x2xf64>
  toy.print %0 : tensor<1x2xf64>
  toy.return
}