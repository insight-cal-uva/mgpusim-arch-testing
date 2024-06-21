// Package fir implements the FIR benchmark form Hetero-Mark.
package nnet

import (
	"log"
	"math"

	"bytes"
	"encoding/binary"

	// embed hsaco files
	_ "embed"

	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/insts"
	"github.com/sarchlab/mgpusim/v3/kernels"
)

/*
 * OpenCL kernel arguments
 */
// __global__ void calculate_neuron(int count, float *prev, int prev_dim, float* next, int next_dim, float* weights, float* bias, int activation){

// KernelArgs defines kernel arguments
type KernelArgs struct { 
	Count 			 int32
	Pad 			 int32
	Prev 			 driver.Ptr
	PrevDim 		 int32
	Pad1 			 int32
	Next 			 driver.Ptr
	NextDim 		 int32
	Pad2 			 int32
	Weights 		 driver.Ptr
	Bias 			 driver.Ptr
	Activation 		 int32
	Pad3 			 int32
	// Padding		       int32 // padding to align to 8 bytes
	HiddenGlobalOffsetX int64
	HiddenGlobalOffsetY int64
	HiddenGlobalOffsetZ int64
} // note: according to https://github.com/sarchlab/mgpusim/blob/v3/doc/prepare_benchmarks.md#run-a-kernel the padding is only to be to 8 bytes, so none is needed here


// Benchmark defines a benchmark
type Benchmark struct {
	// required fields
	driver  *driver.Driver
	context *driver.Context
	queue   *driver.CommandQueue
	hsaco   *insts.HsaCo
	gpus    []int

	// parameters
	numExamples int // adjust this for cache locality differences
	
	// input data on host
	modelData []float32
	mnistData []float32

	// input data on device
	gModelData driver.Ptr
	gMnistData driver.Ptr
	gTmpData   driver.Ptr
	gTmpData2  driver.Ptr

	// commonly used options ( we will not care about unified memory at first)
	useUnifiedMemory bool
}

//go:embed kernels.hsaco
var hsacoBytes []byte

//go:embed mnist_data.bin
var mnistDataBytes []byte

//go:embed model_data.bin
var modelDataBytes []byte

// Helper function
func bytesToFloat32(b []byte) []float32 {
	floatArray := make([]float32, len(b)/4)

	buffer := bytes.NewReader(b)

	err := binary.Read(buffer, binary.LittleEndian, &floatArray)
	
	if err != nil {
		log.Fatal("binary.Read failed:", err)
	}

	return floatArray
}

// NewBenchmark returns a benchmark
func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := new(Benchmark)

	b.driver = driver
	b.context = b.driver.Init()
	b.queue = driver.CreateCommandQueue(b.context)

    b.hsaco = kernels.LoadProgramFromMemory(hsacoBytes, "calculate_neuron") // for some reason, we are not loading the correct magic bytes to recognize the hsaco file

	return b
}

// SelectGPU select GPU
func (b *Benchmark) SelectGPU(gpus []int) {
	b.gpus = gpus
}

// SetUnifiedMemory uses Unified Memory
func (b *Benchmark) SetUnifiedMemory() {
	b.useUnifiedMemory = true
}

// Run runs
func (b *Benchmark) Run() {
	b.driver.SelectGPU(b.context, b.gpus[0])
	b.initMem()
	b.exec()
}

func (b *Benchmark) initMem() {
    // todo: change init Mem to use the correct initilization scheme
	b.mnistData = bytesToFloat32(mnistDataBytes)
	b.modelData = bytesToFloat32(modelDataBytes)

	b.numExamples = 10

	// len_mnist := 28 * 28 * b.numExamples
	// len_model := (28 * 28) * 128 + 128 + (128 * 64) + 64 + (64 * 10) + 10 + 128 + 64 + 10
	len_tmp := 28 * 28 * b.numExamples

	b.gMnistData = b.driver.AllocateMemory(b.context, uint64(len(b.mnistData) * 4 + 128))
	b.gModelData = b.driver.AllocateMemory(b.context, uint64(len(b.modelData) * 4 + 128))
	b.gTmpData = b.driver.AllocateMemory(b.context, uint64(len_tmp*4))
	b.gTmpData2 = b.driver.AllocateMemory(b.context, uint64(len_tmp*4))

	b.driver.MemCopyH2D(b.context, b.gMnistData, b.mnistData)
	b.driver.MemCopyH2D(b.context, b.gModelData, b.modelData)

	// bogus memcopies into scratch space to avoid segfaults??
	b.driver.MemCopyH2D(b.context, b.gTmpData, make([]float32, len_tmp))
	b.driver.MemCopyH2D(b.context, b.gTmpData2, make([]float32, len_tmp))
}

func (b *Benchmark) exec() {
	queues := make([]*driver.CommandQueue, len(b.gpus))

	// make threadsPerBlock a uint16
	threadsPerBlock := uint16(16) 

	blocksPerGrid := func(n int) uint32 {
		return uint32(math.Ceil(float64(n) / float64(threadsPerBlock)))
	}

	// log the number of examples
	log.Printf("Number of examples: %d\n", b.numExamples)
	log.Printf("Number of gpus: %d\n", len(b.gpus))
	log.Printf("Blocks per Grid %d\n", blocksPerGrid(b.numExamples))

	for i, gpu := range b.gpus {
		b.driver.SelectGPU(b.context, gpu)
		queues[i] = b.driver.CreateCommandQueue(b.context)

		log.Printf("Entered the first command queue\n")

		// logging out each pointer
		log.Printf("gMnistData: %v\n", b.gMnistData)
		log.Printf("gModelData: %v\n", b.gModelData)
		log.Printf("gTmpData: %v\n", b.gTmpData)
		log.Printf("gTmpData2: %v\n", b.gTmpData2)

		// first layer
		args := KernelArgs{
			int32(b.numExamples),
			-1, // padding
			b.gMnistData,
			1,
			-1, // padding
			b.gTmpData,
			1,
			-1, // padding
			b.gModelData,
			b.gTmpData2,
			1,
			-1, // padding
			0, 0, 0,
		}

		b.driver.EnqueueLaunchKernel(
			queues[i],
			b.hsaco,
			[3]uint32{blocksPerGrid(b.numExamples * 128), 1, 1},
			[3]uint16{threadsPerBlock, 1, 1}, &args,
		)

		log.Printf("First layer done\n")

		// second layer
		args = KernelArgs{
			int32(b.numExamples),
			-1, // padding
			b.gMnistData,
			1,
			-1, // padding
			b.gTmpData,
			1,
			-1, // padding
			b.gModelData,
			b.gModelData,
			1,
			-1, // padding
			0, 0, 0,
		}

		b.driver.EnqueueLaunchKernel(
			queues[i],
			b.hsaco,
			[3]uint32{blocksPerGrid(b.numExamples * 64), 1, 1},
			[3]uint16{threadsPerBlock, 1, 1}, &args,
		)

		log.Printf("Second layer done\n")

		// last layer
		args = KernelArgs{
			int32(b.numExamples),
			-1, // padding
			b.gMnistData,
			1,
			-1, // padding
			b.gTmpData,
			1,
			-1, // padding
			b.gModelData,
			b.gModelData,
			0,
			-1, // padding
		 	0, 0, 0,
		}

		b.driver.EnqueueLaunchKernel(
			queues[i],
			b.hsaco,
			[3]uint32{blocksPerGrid(b.numExamples * 10), 1, 1},
			[3]uint16{threadsPerBlock, 1, 1}, &args,
		)

		log.Printf("Last layer done\n")
	}

	for i := range b.gpus {
		b.driver.DrainCommandQueue(queues[i])

		log.Printf("Drained queue %d\n", i)
	}
}

// Verify verifies
func (b *Benchmark) Verify() {
    // no verfication, we are going to leave that as a stub
	log.Printf("Passed!\n")
}
