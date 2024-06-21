package main

import (
	"flag" 

	"gpu-simulator/nnet" // neural network module
	"gpu-simulator/expr" // experiment module2
)

func main() {
	flag.Parse()

	runner := new(runner.Runner).ParseFlag().Init()

	benchmark := nnet.NewBenchmark(runner.Driver())
	// set the length or any parameters

	runner.AddBenchmark(benchmark)

	runner.Run()
}
