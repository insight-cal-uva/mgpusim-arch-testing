// uses the cuda runtime api to: load the neural network + execute the neural network

#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "cuda_runtime.h"

#define ck(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__global__ void calculate_neuron(int count, float *prev, int prev_dim, float *next, int next_dim, float *weights, float *bias, int activation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= next_dim * count)
        return; // too far

    int next_index = idx % next_dim;

    float total = 0.0;
    for (int i = 0; i < prev_dim; i++) {
        // weights are a n x m matrix so when you flatten the ith row needs to be multiplied by next_dim and the jth column is the next_index
        float weight = weights[i * next_dim + next_index];
        float prior_node = prev[prev_dim * idx / next_dim + i];
        total += weight * prior_node;
    }

    total += bias[next_index];

    if (activation) {
        next[idx] = total > 0 ? total : 0; // relu activation function
    } else {
        next[idx] = total; // no activation function present
    }
}

/*
   Performs a feed forward inference pass on flattened MNIST data

   - `model` is the model weights (on gpu)
   - `mnist` is the input data (on gpu)
   - `tmp` is a large block of memory which the layer can be copied to (gpu)
*/
void feed_forward(float *model, float *mnist, float *tmp, int num_examples, int print_pred) {
    int threads_per_block = 256; // known good value
    int blocks_per_grid;

    int model_offset = 0;
    int data_offset = 0;

    // first layer
    blocks_per_grid = (128 * num_examples + threads_per_block - 1) / threads_per_block;
    calculate_neuron<<<blocks_per_grid, threads_per_block>>>(num_examples, mnist, 28 * 28, tmp, 128, model, model + 28 * 28 * 128 * sizeof(float), 1);
    model_offset += 28 * 28 * 128 + 128; // weights and the bias
    data_offset += 128 * num_examples;

    // second layer
    blocks_per_grid = (64 * num_examples + threads_per_block - 1) / threads_per_block;
    calculate_neuron<<<blocks_per_grid, threads_per_block>>>(num_examples, tmp, 128, tmp + data_offset * sizeof(float), 64,
                                                             model + model_offset * sizeof(float), model + (model_offset + 128 * 64) * sizeof(float),
                                                             1);
    model_offset += 128 * 64 + 64; // weights and the bias

    // last layer
    blocks_per_grid = (10 * num_examples + threads_per_block - 1) / threads_per_block;
    calculate_neuron<<<blocks_per_grid, threads_per_block>>>(
        num_examples, tmp + data_offset * sizeof(float), 64, tmp + (data_offset + 64 * num_examples) * sizeof(float), 10,
        model + model_offset * sizeof(float), model + (model_offset + 64 * 10) * sizeof(float), 0);

    // copy back to the cpu (skipping for now)
}

/*
   Loads a neural network from `model` file and performs inference using the entire MNIST database
 */
int main(int argc, char *argv[]) {
    // model parameters
    int first_layer = 28 * 28;
    int second_layer = 128;
    int third_layer = 64;
    int last_layer = 10;

    // calculating the model size
    int total_neurons = first_layer + second_layer + third_layer + last_layer;
    int weights = first_layer * second_layer + second_layer * third_layer + third_layer * last_layer;
    int bias = second_layer + third_layer + last_layer; // no bias on the first layer

    int model_size = (weights + bias) * sizeof(float); // the lenght of the binary string to read

    fprintf(stderr, "Loading a model of size %d..\n", model_size);

    // loading the model weights
    // note: managed memory was intentionally not used for greater memory control

    int fd = open("model", NULL);
    if (fd < 0) {
        fprintf(stderr, "Error opening `model`, exiting... \n");
        exit(1);
    }

    float *model;
    float *gpu_model;

    ck(cudaMallocHost(&model, model_size));

    if (model_size != read(fd, model, model_size)) {
        fprintf(stderr, "Did not read enough bytes, exiting...\n");
        exit(1);
    } else {
        close(fd);
    }

    ck(cudaMalloc((void **)&gpu_model, model_size));
    ck(cudaMemcpy(gpu_model, model, model_size, cudaMemcpyHostToDevice));
    fprintf(stderr, "Model loaded with data on the GPU\n");

    // Loading MNIST examples
    int mnist_fd = open("mnist_train_images.bin", NULL);
    if (mnist_fd < 0) {
        fprintf(stderr, "Error openining train images... \n");
        exit(1);
    }

    float *mnist;
    float *gpu_mnist;

    int num_mnist_images = 60000; // 60k
    int mnist_size = 28 * 28 * sizeof(float) * num_mnist_images;

    ck(cudaMallocHost(&mnist, mnist_size));

    if (mnist_size != read(fd, mnist, mnist_size)) {
        fprintf(stderr, "Did not read enough mnist bytes, exiting...\n");
        exit(1);
    }

    ck(cudaMalloc((void **)&gpu_mnist, mnist_size));
    ck(cudaMemcpy(gpu_mnist, mnist, mnist_size, cudaMemcpyHostToDevice));

    // Loading the ground truth labels
    int mnist_outputs_fd = open("sample_outputs.bin", NULL);
    if (mnist_outputs_fd < 0) {
        fprintf(stderr, "Error openining sample outputs... \n");
        exit(1);
    }

    float *mnist_outputs;

    int mnist_labels_size = 10 * sizeof(float) * num_mnist_images;

    ck(cudaMallocHost(&mnist_outputs, mnist_labels_size));

    if (0 != read(fd, mnist_outputs, mnist_labels_size)) {
        fprintf(stderr, "Did not read enough mnist labels bytes, exiting...\n");
        exit(1);
    }

    // Evaluating performance
    int max_samples = 10000;
    int step = 2;

    // mallocing enough temporary memory
    float *tmp;
    ck(cudaMalloc((void **)&tmp, max_samples * total_neurons * sizeof(float)));

    struct timespec start, end;
    printf("num_examples,time_taken\n"); // header
    for (int i = 5; i < max_samples; i += step) {
        clock_gettime(CLOCK_MONOTONIC, &start);

        // launching an inference pass
        feed_forward(gpu_model, gpu_mnist, tmp, i, 0);

        clock_gettime(CLOCK_MONOTONIC, &end);

        // validation of results (copy from tmp and compare to mnist_outputs)

        // results are in tmp + (128 + 64) * i * sizeof(float) for the last layer

        // get the data from the gpu
        float *results = (float *)malloc(10 * i * sizeof(float));
        ck(cudaMemcpy(results, tmp + (128 + 64) * i * sizeof(float), 10 * i * sizeof(float), cudaMemcpyDeviceToHost));

        // compare the results to 0.0001
        int correct = 0;
        for(int j=0; j < i * 10; j++) {
            float diff = results[j] - mnist_outputs[j];
            diff = diff > 0 ? diff : -diff;

            if (diff < 0.0001) {
                correct++;
            }
        }
        // print the number correct as a percentage
        fprintf(stderr, "Correct: %f\n", (float)correct / (i * 10));

        free(results);

        // elapsed nanoseconds
        long long elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);

        // log data to csv output format
        fprintf(stderr, "Completed %d\n", i);
        printf("%d,%lld\n", i, elapsed_ns);
    }

    // cleaning up memory
    cudaFreeHost(model);
    cudaFree(gpu_model);
}
