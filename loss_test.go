package main

import (
	"math"
	"testing"
)

func TestMSE(t *testing.T) {
	cases := []struct {
		yHat, y Matrix
		want    float64
	}{
		{Matrix{{1}, {1}}, Matrix{{0}, {0}}, 1.0}, // mean of [1,1]
		{Matrix{{2}}, Matrix{{0}}, 4.0},           // single element: (2-0)^2
		{Matrix{{1, 2}}, Matrix{{1, 2}}, 0.0},     // perfect prediction
		{Matrix{{0, 0}}, Matrix{{1, 1}}, 1.0},     // mean of [1,1]
	}
	for _, c := range cases {
		if got := MSE(c.yHat, c.y); !approx(got, c.want) {
			t.Errorf("MSE(%v, %v) = %v, want %v", c.yHat, c.y, got, c.want)
		}
	}
}

func TestMSEGrad(t *testing.T) {
	cases := []struct {
		yHat, y Matrix
		want    Matrix
	}{
		// n=2, 2*(1-0)/2 = 1 per element
		{Matrix{{1}, {1}}, Matrix{{0}, {0}}, Matrix{{1}, {1}}},
		// single element, n=1: 2*(2-0)/1 = 4
		{Matrix{{2}}, Matrix{{0}}, Matrix{{4}}},
		// perfect prediction -> zero gradient
		{Matrix{{1, 2}}, Matrix{{1, 2}}, Matrix{{0, 0}}},
		// sign: under-predicted -> positive; over-predicted would be negative
		{Matrix{{0}}, Matrix{{1}}, Matrix{{-2}}}, // 2*(0-1)/1 = -2
	}
	for _, c := range cases {
		got := MSEGrad(c.yHat, c.y)
		if !eq(got, c.want) {
			t.Errorf("MSEGrad(%v, %v) = %v, want %v", c.yHat, c.y, got, c.want)
		}
	}
}

// TestMSEGradFiniteDifference checks the analytic gradient against the numeric
// derivative of MSE. This locks the /n consistency: if MSE and MSEGrad ever
// disagree on mean-vs-sum, this fails.
func TestMSEGradFiniteDifference(t *testing.T) {
	const h = 1e-6
	yHat := Matrix{{0.3, -0.2}, {0.7, 1.4}}
	y := Matrix{{0, 1}, {1, 0}}

	grad := MSEGrad(yHat, y)

	for i := range yHat.Rows() {
		for j := range yHat.Cols() {
			orig := yHat[i][j]

			yHat[i][j] = orig + h
			plus := MSE(yHat, y)
			yHat[i][j] = orig - h
			minus := MSE(yHat, y)
			yHat[i][j] = orig

			numeric := (plus - minus) / (2 * h)
			if math.Abs(numeric-grad[i][j]) > 1e-6 {
				t.Errorf("grad[%d][%d]: analytic %v, numeric %v", i, j, grad[i][j], numeric)
			}
		}
	}
}

func TestMSEPanicsOnMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on shape mismatch")
		}
	}()
	MSE(Matrix{{1, 2}}, Matrix{{1}})
}
