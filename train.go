package main

func Train(n *Net, X, y Matrix, epochs int, lr float64) []float64 {
	losses := make([]float64, 0, epochs)
	for range epochs {
		yHat := n.Forward(X)
		losses = append(losses, MSE(yHat, y))
		n.Backward(MSEGrad(yHat, y))
		SGDStep(n.Layers, lr)
	}
	return losses
}
