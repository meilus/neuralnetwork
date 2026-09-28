package main

import "fmt"

func main() {
	// XOR: classic non-linearly-separable problem. A single layer cannot
	// solve it; one hidden layer of 4 neurons with a sigmoid can.
	//
	//   x1 x2 | y
	//   0  0  | 0
	//   0  1  | 1
	//   1  0  | 1
	//   1  1  | 0
	X := Matrix{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	y := Matrix{
		{0},
		{1},
		{1},
		{0},
	}

	// 2 inputs -> 4 hidden neurons -> 1 output, all sigmoid.
	net := &Net{}
	net.Add(NewDense(2, 4, Sigmoid))
	net.Add(NewDense(4, 1, Sigmoid))

	const (
		epochs = 5000
		lr     = 0.5
	)

	fmt.Println("training XOR (2 -> 4 -> 1, sigmoid, MSE, SGD)...")
	losses := Train(net, X, y, epochs, lr)

	// print the loss at a few checkpoints so the descent is visible
	fmt.Println("loss:")
	for i, l := range losses {
		if i%(epochs/10) == 0 {
			fmt.Printf("  epoch %5d  %.6f\n", i, l)
		}
	}
	fmt.Printf("  epoch %5d  %.6f\n", epochs, losses[len(losses)-1])

	// final predictions vs targets
	fmt.Println("predictions:")
	yHat := net.Forward(X)
	for i := range X.Rows() {
		fmt.Printf("  %v xor %v = %.4f  (want %.0f)\n",
			int(X[i][0]), int(X[i][1]), yHat[i][0], y[i][0])
	}
}
