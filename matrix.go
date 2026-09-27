package main

type Matrix [][]float64

func NewMatrix(r, c int) Matrix {
	m := make([][]float64, r)
	for i := range r {
		m[i] = make([]float64, c)
	}
	return Matrix(m)
}

func (m Matrix) Rows() int {
	return len(m)
}

func (m Matrix) Cols() int {
	if len(m) == 0 {
		return 0
	}
	return len(m[0])
}

func MatNeg(a Matrix) Matrix {
	m := NewMatrix(a.Rows(), a.Cols())
	for i := range a.Rows() {
		for j := range a.Cols() {
			m[i][j] = -a[i][j]
		}
	}
	return m
}

func MatAdd(a, b Matrix) Matrix {
	if a.Rows() != b.Rows() || a.Cols() != b.Cols() {
		panic("row or column missmatch")
	}

	m := NewMatrix(a.Rows(), a.Cols())
	for i := range a.Rows() {
		for j := range a.Cols() {
			m[i][j] = a[i][j] + b[i][j]
		}
	}
	return m
}

func MatSub(a, b Matrix) Matrix {
	return MatAdd(a, MatNeg(b))
}

func MatScale(a Matrix, s float64) Matrix {
	m := NewMatrix(a.Rows(), a.Cols())
	for i := range a.Rows() {
		for j := range a.Cols() {
			m[i][j] = a[i][j] * s
		}
	}
	return m
}

func MatMul(a, b Matrix) Matrix {
	if a.Cols() != b.Rows() {
		panic("row and col mismatch")
	}

	m := NewMatrix(a.Rows(), b.Cols())
	for i := range a.Rows() {
		for j := range b.Cols() {
			for k := range a.Cols() {
				m[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return m
}

func MatTrans(a Matrix) Matrix {
	m := NewMatrix(a.Cols(), a.Rows())
	for i := range a.Rows() {
		for j := range a.Cols() {
			m[j][i] = a[i][j]
		}
	}
	return m
}

func MatApply(a Matrix, f func(float64) float64) Matrix {
	m := NewMatrix(a.Rows(), a.Cols())
	for i := range a.Rows() {
		for j := range a.Cols() {
			m[i][j] = f(a[i][j])
		}
	}
	return m
}
