// OpenCL Kernel Generated from ChatGPT

__kernel void calculate_neuron(int count, __global float *prev, int prev_dim, __global float* next, int next_dim, __global float* weights, __global float* bias, int activation){
        
    return; // noop
    int idx = get_global_id(0);

    if(idx >= next_dim * count) return; // too far

    int next_index = idx % next_dim;

    float total = 0.0f;
    for(int i = 0; i < prev_dim; i++){
        // weights are a n x m matrix so when you flatten the ith row needs to be multiplied by next_dim and the jth column is the next_index
        float weight = weights[i * next_dim + next_index];
        float prior_node = prev[prev_dim * idx / next_dim + i];
        total += weight * prior_node;
    }

    total += bias[next_index];

    if(activation){
        next[idx] = total > 0.0f ? total : 0.0f; // relu activation function
    } else {
        next[idx] = total; // no activation function present
    }
}

