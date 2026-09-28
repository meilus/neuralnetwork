package main

import "testing"

func TestNetAdd(t *testing.T) {
	n := &Net{}
	n.Add(NewDense(2, 3, Sigmoid))
	n.Add(NewDense(3, 1, Sigmoid))

	if len(n.Layers) != 2 {
		t.Fatalf("len(Layers) = %d, want 2", len(n.Layers))
	}
}

func TestNetForwardShape(t *testing.T) {
	n := &Net{}
	n.Add(NewDense(2, 3, Sigmoid))
	n.Add(NewDense(3, 1, Sigmoid))

	X := Matrix{{0, 0}, {0, 1}, {1, 0}, {1, 1}} // 4×2
	Y := n.Forward(X)

	if Y.Rows() != 4 || Y.Cols() != 1 {
		t.Fatalf("Y shape = %dx%d, want 4x1", Y.Rows(), Y.Cols())
	}
}

// TestNetForwardChainsLayers checks Net.Forward equals calling each layer's
// Forward by hand.
func TestNetForwardChainsLayers(t *testing.T) {
	l1 := NewDense(2, 3, Sigmoid)
	l2 := NewDense(3, 1, Sigmoid)
	n := &Net{}
	n.Add(l1)
	n.Add(l2)

	X := Matrix{{0, 1}}
	got := n.Forward(X)

	// manual: fixes l1's weights, so use the same objects
	want := l2.Forward(l1.Forward(X))
	if !eq(got, want) {
		t.Fatalf("Net.Forward = %v, want %v", got, want)
	}
}

// TestNetBackwardPopulatesGradients checks every layer ends with non-nil,
// correctly shaped dW/dB after a forward+backward.
func TestNetBackwardPopulatesGradients(t *testing.T) {
	l1 := NewDense(2, 3, Sigmoid)
	l2 := NewDense(3, 1, Sigmoid)
	n := &Net{}
	n.Add(l1)
	n.Add(l2)

	X := Matrix{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	y := Matrix{{0}, {1}, {1}, {0}}

	yHat := n.Forward(X)
	g := MSEGrad(yHat, y)
	n.Backward(g)

	if l1.dW.Rows() != 2 || l1.dW.Cols() != 3 {
		t.Errorf("l1.dW shape = %dx%d, want 2x3", l1.dW.Rows(), l1.dW.Cols())
	}
	if l2.dW.Rows() != 3 || l2.dW.Cols() != 1 {
		t.Errorf("l2.dW shape = %dx%d, want 3x1", l2.dW.Rows(), l2.dW.Cols())
	}
	if l1.dB.Rows() != 1 || l1.dB.Cols() != 3 {
		t.Errorf("l1.dB shape = %dx%d, want 1x3", l1.dB.Rows(), l1.dB.Cols())
	}
}

// TestNetBackwardMatchesManualChain checks Net.Backward is exactly the manual
// reversed chain: l1.Backward(l2.Backward(gradOut)).
func TestNetBackwardMatchesManualChain(t *testing.T) {
	seedNet := func() (*Net, *Dense, *Dense) {
		l1 := NewDense(2, 3, Sigmoid)
		l2 := NewDense(3, 1, Sigmoid)
		n := &Net{}
		n.Add(l1)
		n.Add(l2)
		return n, l1, l2
	}

	X := Matrix{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	y := Matrix{{0}, {1}, {1}, {0}}

	// network version
	n1, a1, a2 := seedNet()
	yHat := n1.Forward(X)
	n1.Backward(MSEGrad(yHat, y))

	// manual version on identical layers
	n2, b1, b2 := seedNet()
	// copy weights so both nets are identical
	b1.W = a1.W
	b2.W = a2.W
	b1.B = a1.B
	b2.B = a2.B

	yHat2 := n2.Forward(X)
	g := MSEGrad(yHat2, y)
	b1.Backward(b2.Backward(g))

	if !eq(a1.dW, b1.dW) {
		t.Errorf("l1.dW mismatch: net %v vs manual %v", a1.dW, b1.dW)
	}
	if !eq(a2.dW, b2.dW) {
		t.Errorf("l2.dW mismatch: net %v vs manual %v", a2.dW, b2.dW)
	}
}

// TestNetTrainsXOR is the real end-to-end check: a 2->4->1 net trained with
// MSE + SGD should learn XOR.
func TestNetTrainsXOR(t *testing.T) {
	n := &Net{}
	n.Add(NewDense(2, 4, Sigmoid))
	n.Add(NewDense(4, 1, Sigmoid))

	X := Matrix{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	y := Matrix{{0}, {1}, {1}, {0}}

	for range 5000 {
		yHat := n.Forward(X)
		n.Backward(MSEGrad(yHat, y))
		SGDStep(n.Layers, 0.5)
	}

	yHat := n.Forward(X)
	for i := range yHat.Rows() {
		got := yHat[i][0]
		want := y[i][0]
		if (want == 0 && got > 0.1) || (want == 1 && got < 0.9) {
			t.Errorf("XOR row %d: got %.3f, want ~%.0f", i, got, want)
		}
	}
}
