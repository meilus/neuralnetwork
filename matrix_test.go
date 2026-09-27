package main

import "testing"

func eq(a, b Matrix) bool {
	if a.Rows() != b.Rows() || a.Cols() != b.Cols() {
		return false
	}
	for i := range a.Rows() {
		for j := range a.Cols() {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}

var (
	a = Matrix{{1, 2, 3}, {4, 5, 6}}          // 2x3
	b = Matrix{{7, 8}, {9, 10}, {11, 12}}     // 3x2
	c = Matrix{{1, 2}, {3, 4}}                // 2x2
)

func TestNewMatrix(t *testing.T) {
	m := NewMatrix(2, 3)
	if m.Rows() != 2 || m.Cols() != 3 {
		t.Fatalf("shape = %dx%d, want 2x3", m.Rows(), m.Cols())
	}
	for i := range m.Rows() {
		for j := range m.Cols() {
			if m[i][j] != 0 {
				t.Fatalf("m[%d][%d] = %v, want 0", i, j, m[i][j])
			}
		}
	}
}

func TestMatMul(t *testing.T) {
	got := MatMul(a, b)
	want := Matrix{{58, 64}, {139, 154}}
	if !eq(got, want) {
		t.Fatalf("A*B = %v, want %v", got, want)
	}
}

func TestMatMulIdentity(t *testing.T) {
	id := Matrix{{1, 0}, {0, 1}}
	got := MatMul(c, id)
	if !eq(got, c) {
		t.Fatalf("C*I = %v, want %v", got, c)
	}
}

func TestMatMulPanicsOnMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on inner-dim mismatch")
		}
	}()
	MatMul(a, a) // 2x3 * 2x3, inner dims 3 != 2
}

func TestT(t *testing.T) {
	got := a.T()
	want := Matrix{{1, 4}, {2, 5}, {3, 6}}
	if !eq(got, want) {
		t.Fatalf("A.T = %v, want %v", got, want)
	}
}

func TestTInvolution(t *testing.T) {
	if !eq(a.T().T(), a) {
		t.Fatalf("(A.T).T != A")
	}
}

func TestMatAdd(t *testing.T) {
	got := MatAdd(c, c)
	want := Matrix{{2, 4}, {6, 8}}
	if !eq(got, want) {
		t.Fatalf("C+C = %v, want %v", got, want)
	}
}

func TestMatAddPanicsOnMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on shape mismatch")
		}
	}()
	MatAdd(a, b)
}

func TestMatSub(t *testing.T) {
	got := MatSub(c, c)
	want := NewMatrix(2, 2)
	if !eq(got, want) {
		t.Fatalf("C-C = %v, want zero", got)
	}
}

func TestMatNeg(t *testing.T) {
	got := MatNeg(c)
	want := Matrix{{-1, -2}, {-3, -4}}
	if !eq(got, want) {
		t.Fatalf("-C = %v, want %v", got, want)
	}
}

func TestMatScale(t *testing.T) {
	got := MatScale(c, 3)
	want := Matrix{{3, 6}, {9, 12}}
	if !eq(got, want) {
		t.Fatalf("3*C = %v, want %v", got, want)
	}
}