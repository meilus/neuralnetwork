package main

import (
	"math"
	"testing"
)

const eps = 1e-9

func approx(got, want float64) bool {
	return math.Abs(got-want) <= eps
}

func TestSigmoid(t *testing.T) {
	cases := []struct{ in, want float64 }{
		{0, 0.5},
		{1, 0.7310585786300049},
		{-1, 0.2689414213699951},
		{100, 1}, // saturates, no NaN
		{-100, 0},
	}
	for _, c := range cases {
		if got := sigmoid(c.in); !approx(got, c.want) {
			t.Errorf("sigmoid(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestSigmoidDer(t *testing.T) {
	cases := []struct{ in, want float64 }{
		{0, 0.25},
		{1, 0.19661193324148185},
	}
	for _, c := range cases {
		if got := sigmoidDer(c.in); !approx(got, c.want) {
			t.Errorf("sigmoidDer(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestTanh(t *testing.T) {
	cases := []struct{ in, want float64 }{
		{0, 0},
		{1, 0.7615941559557649},
		{-1, -0.7615941559557649},
	}
	for _, c := range cases {
		if got := tanh(c.in); !approx(got, c.want) {
			t.Errorf("tanh(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestTanhDer(t *testing.T) {
	cases := []struct{ in, want float64 }{
		{0, 1},
		{1, 0.4199743416140261},
	}
	for _, c := range cases {
		if got := tanhDer(c.in); !approx(got, c.want) {
			t.Errorf("tanhDer(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestReLU(t *testing.T) {
	if got := relu(-1); got != 0 {
		t.Errorf("relu(-1) = %v, want 0", got)
	}
	if got := relu(1); got != 1 {
		t.Errorf("relu(1) = %v, want 1", got)
	}
	if got := relu(2.5); got != 2.5 {
		t.Errorf("relu(2.5) = %v, want 2.5", got)
	}
}

func TestReLUDer(t *testing.T) {
	cases := []struct{ in, want float64 }{
		{-1, 0},
		{0, 0}, // convention f'(0) = 0
		{1, 1},
	}
	for _, c := range cases {
		if got := reluDer(c.in); got != c.want {
			t.Errorf("reluDer(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestLinear(t *testing.T) {
	if got := linear(-3.5); got != -3.5 {
		t.Errorf("linear(-3.5) = %v, want -3.5", got)
	}
	if got := linearDer(7); got != 1 {
		t.Errorf("linearDer(7) = %v, want 1", got)
	}
}

func TestActivationVarsWired(t *testing.T) {
	if got := Sigmoid.function(0); got != 0.5 {
		t.Errorf("Sigmoid.function(0) = %v, want 0.5", got)
	}
	if got := Sigmoid.derivative(0); got != 0.25 {
		t.Errorf("Sigmoid.derivative(0) = %v, want 0.25", got)
	}
	if got := ReLU.function(-2); got != 0 {
		t.Errorf("ReLU.function(-2) = %v, want 0", got)
	}
	if got := Linear.derivative(9); got != 1 {
		t.Errorf("Linear.derivative(9) = %v, want 1", got)
	}
}

func TestMatApplySigmoid(t *testing.T) {
	zeros := Matrix{{0, 0}, {0, 0}}
	got := MatApply(zeros, sigmoid)
	want := Matrix{{0.5, 0.5}, {0.5, 0.5}}
	if !eq(got, want) {
		t.Errorf("MatApply(zeros, sigmoid) = %v, want %v", got, want)
	}
}

func TestMatApplyShape(t *testing.T) {
	in := Matrix{{1, 2, 3}, {4, 5, 6}}
	got := MatApply(in, sigmoid)
	if got.Rows() != in.Rows() || got.Cols() != in.Cols() {
		t.Fatalf("shape = %dx%d, want %dx%d", got.Rows(), got.Cols(), in.Rows(), in.Cols())
	}
	if !approx(got[0][0], 0.7310585786300049) {
		t.Errorf("got[0][0] = %v, want 0.7310585786300049", got[0][0])
	}
}

func TestMatApplyIdentity(t *testing.T) {
	in := Matrix{{-1, 2}, {3, -4}}
	got := MatApply(in, linear)
	if !eq(got, in) {
		t.Errorf("MatApply(in, linear) = %v, want %v", got, in)
	}
}