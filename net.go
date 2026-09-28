package main

type Net struct {
	Layers []*Dense
}

func (n *Net) Add(l *Dense) {
	n.Layers = append(n.Layers, l)
}

func (n *Net) Forward(X Matrix) Matrix {
	out := X
	for _, l := range n.Layers {
		out = l.Forward(out)
	}
	return out
}

func (n *Net) Backward(gradOut Matrix) {
	g := gradOut
	for i := len(n.Layers) - 1; i >= 0; i-- {
		g = n.Layers[i].Backward(g)
	}
}
