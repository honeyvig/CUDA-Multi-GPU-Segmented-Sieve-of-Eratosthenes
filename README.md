# CUDA-Multi-GPU-Segmented-Sieve-of-Eratosthenes
To implement the Segmented Sieve of Eratosthenes using CUDA with multiple GPUs, we need to break the problem into two major parts:

    Segmentation: The Segmented Sieve is a modification of the Sieve of Eratosthenes where we break the range of numbers into smaller blocks (segments), and we sieve each segment in parallel.
    Multi-GPU Setup: This involves setting up multiple CUDA devices (GPUs) and distributing the sieving tasks across them.

Steps:

    Segment the range: Break the range [2, N] into smaller segments.
    For each segment: Each segment will be sieved independently using known primes up to sqrt(N) which are found using a base sieve.
    Distribute across multiple GPUs: Each GPU handles one or more segments, and then we aggregate the results.

CUDA Code for Multi-GPU Segmented Sieve of Eratosthenes

The implementation will include:

    A base sieve to generate all primes up to sqrt(N).
    A segmented sieve that computes the primes in the range [L, R] (for each segment).
    Multiple GPUs will be used to process the segments in parallel.

Here is a simplified version of how we can write CUDA code for the multi-GPU segmented sieve of Eratosthenes.

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

#define MAX_PRIME 1000000

__global__ void segmented_sieve_kernel(int* primes, int* sieve, int L, int R, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int prime = primes[idx];
        int start = max(prime * prime, (L + prime - 1) / prime * prime);
        
        for (int i = start; i <= R; i += prime) {
            sieve[i - L] = 0;
        }
    }
}

// Function to generate primes up to sqrt(N) using the classic sieve of Eratosthenes
void base_sieve(int n, std::vector<int>& primes) {
    bool is_prime[n + 1];
    std::fill(is_prime, is_prime + n + 1, true);
    is_prime[0] = is_prime[1] = false;
    
    for (int p = 2; p * p <= n; ++p) {
        if (is_prime[p]) {
            for (int multiple = p * p; multiple <= n; multiple += p) {
                is_prime[multiple] = false;
            }
        }
    }

    for (int i = 2; i <= n; ++i) {
        if (is_prime[i]) {
            primes.push_back(i);
        }
    }
}

// Function to handle segmentation and call CUDA for segmented sieve
void segmented_sieve(int L, int R, std::vector<int>& primes) {
    int size = R - L + 1;
    int* d_sieve;
    int* d_primes;
    
    int num_primes = primes.size();
    cudaMalloc((void**)&d_sieve, size * sizeof(int));
    cudaMalloc((void**)&d_primes, num_primes * sizeof(int));
    
    int* sieve = new int[size];
    std::fill(sieve, sieve + size, 1);

    cudaMemcpy(d_sieve, sieve, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_primes, primes.data(), num_primes * sizeof(int), cudaMemcpyHostToDevice);
    
    // Define block size and grid size
    int block_size = 256;
    int grid_size = (num_primes + block_size - 1) / block_size;
    
    // Launch the kernel to process primes on GPU
    segmented_sieve_kernel<<<grid_size, block_size>>>(d_primes, d_sieve, L, R, num_primes);
    
    cudaMemcpy(sieve, d_sieve, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the primes in the range [L, R]
    for (int i = 0; i < size; ++i) {
        if (sieve[i]) {
            printf("%d ", L + i);
        }
    }

    delete[] sieve;
    cudaFree(d_sieve);
    cudaFree(d_primes);
}

int main() {
    int N = 1000000;
    int L = 1000000, R = 1001000;  // Define the range [L, R] for segmentation

    // Step 1: Generate base primes up to sqrt(N) using base sieve
    int limit = sqrt(N);
    std::vector<int> primes;
    base_sieve(limit, primes);

    // Step 2: Use CUDA to apply the segmented sieve for the range [L, R]
    segmented_sieve(L, R, primes);

    return 0;
}

Explanation:

    Base Sieve (base_sieve function): This generates all primes up to sqrt(N) using the classic Sieve of Eratosthenes. These primes will be used to mark non-prime numbers in each segment.
    Segmented Sieve Kernel (segmented_sieve_kernel): This kernel takes in the list of base primes and marks multiples of these primes as non-prime within the range [L, R] (the segment). It does this in parallel for each prime using CUDA threads.
    Multi-GPU: This code currently uses a single GPU. To scale this to multiple GPUs, you would need to distribute segments across GPUs, which requires setting up multiple CUDA contexts and handling synchronization. You would allocate different segments of the sieve range to each GPU, and each GPU would perform the sieving for its assigned segment.

Considerations for Multi-GPU Implementation:

    CUDA Streams: Each GPU can run in parallel using CUDA streams, which will allow you to manage multiple GPUs efficiently.
    Unified Memory: You can use CUDA Unified Memory to share memory between GPUs, avoiding the need for explicit memory copies.
    GPU Synchronization: After each GPU processes its segment, you would need to merge results from each GPU and handle memory management carefully.

Optimizations:

    You can enhance the performance of this algorithm by optimizing memory access patterns and minimizing unnecessary memory transfers between host and device.
    For large ranges, consider using techniques like tile-based sieving to handle memory more efficiently on the GPU.

Execution:

    Compiling: This program should be compiled using nvcc (NVIDIA CUDA Compiler).

nvcc -o segmented_sieve segmented_sieve.cu

Running: Execute the compiled binary on a machine with CUDA-capable GPUs.

    ./segmented_sieve

Notes:

    This is a basic structure. For multi-GPU handling, you'd need to extend it by managing multiple devices using CUDA APIs to assign different segments to different GPUs and then merge the results from each device.
