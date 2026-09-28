package main

import (
	"math"
	"testing"
)

// TestDenseGradientCheck validates the analytic gradients (dW, dB, dX) against
// numeric gradients computed by finite differences. This is the gold-standard
// test for backprop: if the matrix shapes or transposes are wrong, it fails.
func TestDenseGradientCheck(t *testing.T) {
	const h = 1e-6 // finite-difference step

	d := NewDense(3, 2, Sigmoid)

	// loss = sum over all outputs (simple, so dLoss/dY = all ones).
	loss := func(X Matrix) float64 {
		Y := d.Forward(X)
		s := 0.0
		for i := range Y.Rows() {
			for j := range Y.Cols() {
				s += Y[i][j]
			}
		}
		return s
	}

	X := Matrix{{0.3, -0.2, 0.7}, {1.1, 0.4, -0.5}} // 2×3
	Y := d.Forward(X)

	// upstream gradient for loss = sum(Y) is all ones.
	gradOut := NewMatrix(Y.Rows(), Y.Cols())
	for i := range gradOut.Rows() {
		for j := range gradOut.Cols() {
			gradOut[i][j] = 1
		}
	}
	dX := d.Backward(gradOut)

	// --- check dW ---
	for i := range d.W.Rows() {
		for j := range d.W.Cols() {
			orig := d.W[i][j]
			d.W[i][j] = orig + h
			plus := loss(X)
			d.W[i][j] = orig - h
			minus := loss(X)
			d.W[i][j] = orig

			numeric := (plus - minus) / (2 * h)
			if math.Abs(numeric-d.dW[i][j]) > 1e-4 {
				t.Errorf("dW[%d][%d]: analytic %v, numeric %v", i, j, d.dW[i][j], numeric)
			}
		}
	}

	// --- check dB ---
	d.Forward(X) // refresh cache before re-backward
	d.Backward(gradOut)
	for j := range d.B.Cols() {
		orig := d.B[0][j]
		d.B[0][j] = orig + h
		plus := loss(X)
		d.B[0][j] = orig - h
		minus := loss(X)
		d.B[0][j] = orig

		numeric := (plus - minus) / (2 * h)
		if math.Abs(numeric-d.dB[0][j]) > 1e-4 {
			t.Errorf("dB[0][%d]: analytic %v, numeric %v", j, d.dB[0][j], numeric)
		}
	}

	// --- check dX ---
	for i := range X.Rows() {
		for j := range X.Cols() {
			orig := X[i][j]
			X[i][j] = orig + h
			plus := loss(X)
			X[i][j] = orig - h
			minus := loss(X)
			X[i][j] = orig

			numeric := (plus - minus) / (2 * h)
			if math.Abs(numeric-dX[i][j]) > 1e-4 {
				t.Errorf("dX[%d][%d]: analytic %v, numeric %v", i, j, dX[i][j], numeric)
			}
		}
	}
}

func TestDenseForwardShape(t *testing.T) {
	d := NewDense(3, 2, Sigmoid)
	X := Matrix{{1, 2, 3}, {4, 5, 6}} // 2×3
	Y := d.Forward(X)
	if Y.Rows() != 2 || Y.Cols() != 2 {
		t.Fatalf("Y shape = %dx%d, want 2x2", Y.Rows(), Y.Cols())
	}
}

func TestColSum(t *testing.T) {
	m := Matrix{{1, 2}, {3, 4}, {5, 6}} // 3×2
	got := colSum(m)
	want := Matrix{{9, 12}}
	if !eq(got, want) {
		t.Fatalf("colSum = %v, want %v", got, want)
	}
}

func TestMatMulElem(t *testing.T) {
	a := Matrix{{1, 2}, {3, 4}}
	b := Matrix{{5, 6}, {7, 8}}
	got := matMulElem(a, b)
	want := Matrix{{5, 12}, {21, 32}}
	if !eq(got, want) {
		t.Fatalf("matMulElem = %v, want %v", got, want)
	}
}

func TestMatAddRow(t *testing.T) {
	a := Matrix{{0, 0}, {0, 0}, {0, 0}}
	row := Matrix{{1, 2}}
	got := MatAddRow(a, row)
	want := Matrix{{1, 2}, {1, 2}, {1, 2}}
	if !eq(got, want) {
		t.Fatalf("MatAddRow = %v, want %v", got, want)
	}
}
