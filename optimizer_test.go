package main

import "testing"

func TestSGDStepUpdatesW(t *testing.T) {
	l := &Dense{
		W:  Matrix{{1, 2}},
		dW: Matrix{{0.5, 0.5}},
		B:  NewMatrix(1, 2),
		dB: NewMatrix(1, 2),
	}

	SGDStep([]*Dense{l}, 0.1)

	want := Matrix{{0.95, 1.95}} // 1-0.1*0.5, 2-0.1*0.5
	if !eq(l.W, want) {
		t.Fatalf("W = %v, want %v", l.W, want)
	}
}

func TestSGDStepUpdatesB(t *testing.T) {
	l := &Dense{
		W:  NewMatrix(1, 2),
		dW: NewMatrix(1, 2),
		B:  Matrix{{1, 2}},
		dB: Matrix{{1, 1}},
	}

	SGDStep([]*Dense{l}, 0.5)

	want := Matrix{{0.5, 1.5}} // 1-0.5*1, 2-0.5*1
	if !eq(l.B, want) {
		t.Fatalf("B = %v, want %v", l.B, want)
	}
}

func TestSGDStepZeroGradientNoMove(t *testing.T) {
	l := &Dense{
		W:  Matrix{{1, 2}, {3, 4}},
		dW: NewMatrix(2, 2),
		B:  Matrix{{5, 6}},
		dB: NewMatrix(1, 2),
	}
	want := Matrix{{1, 2}, {3, 4}}

	SGDStep([]*Dense{l}, 0.9)

	if !eq(l.W, want) {
		t.Fatalf("W = %v, want unchanged %v", l.W, want)
	}
}

func TestSGDStepMultipleLayers(t *testing.T) {
	a := &Dense{W: Matrix{{1}}, dW: Matrix{{1}}, B: Matrix{{0}}, dB: Matrix{{0}}}
	b := &Dense{W: Matrix{{2}}, dW: Matrix{{4}}, B: Matrix{{1}}, dB: Matrix{{2}}}

	SGDStep([]*Dense{a, b}, 0.25)

	if !eq(a.W, Matrix{{0.75}}) { // 1 - 0.25*1
		t.Errorf("a.W = %v, want {{0.75}}", a.W)
	}
	if !eq(b.W, Matrix{{1}}) { // 2 - 0.25*4
		t.Errorf("b.W = %v, want {{1}}", b.W)
	}
	if !eq(b.B, Matrix{{0.5}}) { // 1 - 0.25*2
		t.Errorf("b.B = %v, want {{0.5}}", b.B)
	}
}

func TestSGDStepConvergesOnQuadratic(t *testing.T) {
	// Minimize f(w) = (w-3)^2 with gradient 2(w-3).
	// SGD from w=0 with lr=0.1 should walk toward 3.
	l := &Dense{W: Matrix{{0}}, dW: NewMatrix(1, 1), B: Matrix{{0}}, dB: NewMatrix(1, 1)}

	for range 50 {
		l.dW[0][0] = 2 * (l.W[0][0] - 3)
		SGDStep([]*Dense{l}, 0.1)
	}

	if l.W[0][0] < 2.9 || l.W[0][0] > 3.1 {
		t.Fatalf("W = %v, want ~3", l.W[0][0])
	}
}
