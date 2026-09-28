package main

import "math/rand"

// Dense is a fully-connected layer: Y = activation(X·W + B).
//
// Shapes for a batch of n samples with `in` input features and `out` neurons:
//
//	X     n×in    input activations from the previous layer (or the data)
//	W     in×out  weights (learned)
//	B     1×out   bias, one per output neuron, broadcast across the batch (learned)
//	Z     n×out   cached pre-activation  (X·W + B), before the activation function
//	Y     n×out   Z after the activation, what Forward returns
//
// The d* fields are gradients, same shape as their target:
//
//	dW    in×out  dLoss/dW
//	dB    1×out   dLoss/dB
type Dense struct {
	W, dW Matrix // weights and their gradient
	B, dB Matrix // bias and its gradient
	X     Matrix // cached input (needed to compute dW in Backward)
	Z     Matrix // cached pre-activation (needed to compute the activation derivative)
	act   activation
	in    int
	out   int
}

// NewDense builds a layer mapping `in` features to `out` neurons.
// Weights are initialized small and random so neurons start in different
// places; bias starts at zero. Gradients are allocated with matching shapes.
func NewDense(in, out int, act activation) *Dense {
	w := NewMatrix(in, out)
	for i := range w.Rows() {
		for j := range w.Cols() {
			// scale by 1/sqrt(in) so pre-activations stay in a sane range;
			// this is a simple form of He/Xavier-style initialization.
			w[i][j] = rand.NormFloat64() / float64(in)
		}
	}

	return &Dense{
		W:   w,
		dW:  NewMatrix(in, out),
		B:   NewMatrix(1, out),
		dB:  NewMatrix(1, out),
		act: act,
		in:  in,
		out: out,
	}
}

// Forward computes Y = activation(X·W + B) and caches X and Z.
//
//	d.X = X                 so Backward can compute dW = Xᵀ·dZ
//	Z   = X·W + B           the pre-activation, saved for the derivative
//	Y   = activation(Z)     returned
func (d *Dense) Forward(X Matrix) Matrix {
	d.X = X // cache input for Backward (dW = Xᵀ·dZ needs it)

	// X·W: (n×in)·(in×out) = n×out. Then add the bias row to every row.
	Z := MatAddRow(MatMul(X, d.W), d.B)
	d.Z = Z // cache pre-activation for the activation derivative

	return MatApply(Z, d.act.function)
}

// Backward receives gradOut = dLoss/dY, the gradient of the loss with respect
// to this layer's output (shape n×out, handed down from the next layer).
// It computes dW and dB, and returns dLoss/dX to pass to the previous layer.
//
// The chain rule, step by step:
//
//	1. through the activation: dZ = gradOut ⊙ act'(Z)      (element-wise)
//	2. weight gradient:        dW = Xᵀ · dZ                (in×out, matches W)
//	3. bias gradient:          dB = column-sum of dZ       (1×out, matches B)
//	4. to previous layer:      dX = dZ · Wᵀ                (n×in, matches X)
//
// Nothing is subtracted here. This only produces gradients; the optimizer
// later does W = W - lr*dW.
func (d *Dense) Backward(gradOut Matrix) Matrix {
	// 1. dZ = gradOut ⊙ act'(Z). act'(Z) has Z's shape (n×out),
	//    matching gradOut, so the element-wise product is valid.
	actGrad := MatApply(d.Z, d.act.derivative)
	dZ := matMulElem(gradOut, actGrad)

	// 2. dW = Xᵀ·dZ. Xᵀ is (in×n), dZ is (n×out) -> in×out = shape of W.
	d.dW = MatMul(MatTrans(d.X), dZ)

	// 3. dB = sum of dZ down the batch: (n×out) -> 1×out = shape of B.
	d.dB = colSum(dZ)

	// 4. dX = dZ·Wᵀ. (n×out)·(out×in) = n×in = shape of X. This is what the
	//    previous layer receives as its gradOut.
	return MatMul(dZ, MatTrans(d.W))
}

// matMulElem is the element-wise (Hadamard) product:
// C[i][j] = A[i][j] * B[i][j]. Both matrices must have the same shape.
// Backprop uses it to multiply an incoming gradient by an activation
// derivative, point by point.
func matMulElem(a, b Matrix) Matrix {
	if a.Rows() != b.Rows() || a.Cols() != b.Cols() {
		panic("row or column missmatch")
	}

	m := NewMatrix(a.Rows(), a.Cols())
	for i := range a.Rows() {
		for j := range a.Cols() {
			m[i][j] = a[i][j] * b[i][j]
		}
	}
	return m
}

// colSum collapses a batch×cols matrix to 1×cols by summing each column
// down the batch axis. The bias is shared by every sample, so its gradient
// is the sum of the per-sample gradients.
func colSum(a Matrix) Matrix {
	m := NewMatrix(1, a.Cols())
	for i := range a.Rows() {
		for j := range a.Cols() {
			m[0][j] += a[i][j]
		}
	}
	return m
}
