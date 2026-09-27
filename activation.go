package main

import "math"

type activation struct {
	function   func(float64) float64
	derivative func(float64) float64
}

var (
	Sigmoid = activation{sigmoid, sigmoidDer}
	Tanh    = activation{tanh, tanhDer}
	ReLU    = activation{relu, reluDer}
	Linear  = activation{linear, linearDer}
)

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidDer(x float64) float64 {
	sig := sigmoid(x)
	return sig * (1 - sig)
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}

func tanhDer(x float64) float64 {
	return 1 - math.Pow(tanh(x), 2)
}

func relu(x float64) float64 {
	return max(0, x)
}

func reluDer(x float64) float64 {
	if x > 0 {
		return 1
	} // i chose f'(0) = 0
	return 0
}

func linear(x float64) float64 {
	return x
}

func linearDer(x float64) float64 {
	return 1
}
