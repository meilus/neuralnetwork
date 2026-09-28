package main

func MSE(yHat, y Matrix) float64 {
	if yHat.Rows() != y.Rows() || yHat.Cols() != y.Cols() {
		panic("incompatible dimentions")
	}

	loss := 0.0
	n := float64(y.Rows() * y.Cols())
	for i := range y.Rows() {
		for j := range y.Cols() {
			diff := yHat[i][j] - y[i][j]
			loss += diff * diff
		}
	}
	return loss / n
}

func MSEGrad(yHat, y Matrix) Matrix {
	if yHat.Rows() != y.Rows() || yHat.Cols() != y.Cols() {
		panic("incompatible dimentions")
	}

	n := float64(y.Rows() * y.Cols())
	m := NewMatrix(y.Rows(), y.Cols())
	for i := range y.Rows() {
		for j := range y.Cols() {
			diff := yHat[i][j] - y[i][j]
			m[i][j] = 2 * diff / n
		}
	}
	return m
}
