package main

func SGDStep(layers []*Dense, lr float64) {
	for _, l := range layers {
		scaledGradW := MatScale(l.dW, lr)
		scaledGradB := MatScale(l.dB, lr)

		l.W = MatSub(l.W, scaledGradW)
		l.B = MatSub(l.B, scaledGradB)
	}
}
