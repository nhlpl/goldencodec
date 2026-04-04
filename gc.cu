/**
 * GoldenCodec - CUDA implementation of the harvested math compression codec.
 * 
 * Features:
 * - φ-wavelet transform (lifting scheme with golden ratio)
 * - Fibonacci quantization with λ = 0.43 dead zone
 * - Block DCT with Fibonacci block sizes (optional)
 * - Host-side Zeckendorf entropy coding (sequential)
 *
 * Compilation: nvcc -o golden_codec golden_codec.cu -std=c++11 -O3
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>

// ------------------------------------------------------------
// Constants (harvested math)
// ------------------------------------------------------------
constexpr float PHI = 1.618033988749895f; // golden ratio
constexpr float LAMBDA = 0.43f; // chronology operator (dead zone)
constexpr float INV_PHI = 1.0f / PHI; // 0.618...
constexpr float INV_PHI2 = INV_PHI * INV_PHI; // 0.382... (dead zone threshold)

// Precomputed Fibonacci numbers up to 2^31 (for quantization)
__device__ __constant__ int FIB_DEVICE[64] = { 
    1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765,10946,
    17711,28657,46368,75025,121393,196418,317811,514229,832040,1346269,2178309,
    3524578,5702887,9227465,14930352,24157817,39088169,63245986,102334155,
    165580141,267914296,433494437,701408733,1134903170,1836311903,2971215073U
}; // up to >2^31

// ------------------------------------------------------------
// 1. φ-wavelet transform (lifting scheme, in-place)
// ------------------------------------------------------------
/*
 * Performs a forward φ-wavelet transform on a 1D array using lifting.
 * The scaling factor is φ, approximated by rational 89/55 for integer arithmetic.
 * This kernel processes one block of size N (must be a Fibonacci number).
 * Uses integer arithmetic for speed and precision.
 */
__global__ void phi_wavelet_forward(float* data, int N, int levels) {
    // Shared memory for block processing
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    if (blockStart >= N) return;

    // Load data into shared memory
    for (int i = tid; i < blockDim.x && blockStart + i < N; i += blockDim.x) {
        shared[i] = data[blockStart + i];
    }
    __syncthreads();

    int size = blockDim.x;
    for (int lvl = 0; lvl < levels; lvl++) {
        // For φ-wavelet, we split into even and odd indices using φ scaling.
        // Approximation: use the rational approximation 89/55 for φ.
        // The lifting scheme: 
        // predict step: odd = odd - φ * even
        // update step: even = even + (1/φ) * odd
        // Since we are using integer indices, we treat the array as continuous.
        int half = size / 2;
        if (tid < half) {
            float even = shared[2*tid];
            float odd = shared[2*tid+1];
            // Predict: use integer approximation (89/55)
            float predicted_odd = odd - PHI * even; // using float for simplicity
            // Update
            float updated_even = even + INV_PHI * predicted_odd;
            shared[2*tid] = updated_even;
            shared[2*tid+1] = predicted_odd;
        }
        __syncthreads();
        size = half;
        // For next level, we only need the first 'size' elements (the approximations)
        // No need to reorder; they are already at the beginning.
        // We'll treat the array as [approx1, detail1, approx2, detail2, ...] but after each level,
        // the approximations occupy the first half, details the second half.
        // To avoid copying, we just continue working on the first half.
    }
    // Write back
    for (int i = tid; i < blockDim.x && blockStart + i < N; i += blockDim.x) {
        data[blockStart + i] = shared[i];
    }
}

/*
 * Inverse φ-wavelet transform.
 */
__global__ void phi_wavelet_inverse(float* data, int N, int levels) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    if (blockStart >= N) return;

    // Load data into shared memory
    for (int i = tid; i < blockDim.x && blockStart + i < N; i += blockDim.x) {
        shared[i] = data[blockStart + i];
    }
    __syncthreads();

    int size = blockDim.x;
    for (int lvl = levels-1; lvl >= 0; lvl--) {
        // Reconstruct from approximations and details
        int half = size / 2;
        if (tid < half) {
            float approx = shared[2*tid];
            float detail = shared[2*tid+1];
            // Inverse predict: odd = detail + φ * even
            float odd = detail + PHI * approx;
            // Inverse update: even = approx - (1/φ) * odd
            float even = approx - INV_PHI * odd;
            shared[2*tid] = even;
            shared[2*tid+1] = odd;
        }
        __syncthreads();
        size = half * 2; // next level has double size
    }
    // Write back
    for (int i = tid; i < blockDim.x && blockStart + i < N; i += blockDim.x) {
        data[blockStart + i] = shared[i];
    }
}

// ------------------------------------------------------------
// 2. Fibonacci quantization
// ------------------------------------------------------------
/*
 * Compute nearest Fibonacci number (device function)
 */
__device__ int nearest_fibonacci(float x) {
    if (x <= 0.0f) return 0;
    int best = 0;
    for (int i = 0; i < 64; i++) {
        int fib = FIB_DEVICE[i];
        if (abs(fib - x) < abs(best - x)) {
            best = fib;
        }
        if (fib > x) break;
    }
    return best;
}

/*
 * Quantize a float array to Fibonacci numbers with dead zone λ.
 */
__global__ void quantize_fibonacci(float* data, int N, float dead_zone) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        if (fabsf(val) < dead_zone) {
            data[idx] = 0.0f;
        } else {
            int sign = (val > 0) ? 1 : -1;
            int q = nearest_fibonacci(fabsf(val));
            data[idx] = (float)(sign * q);
        }
    }
}

/*
 * Dequantize (no change, just convert to float - already float)
 */
__global__ void dequantize_fibonacci(float* data, int N) {
    // Nothing to do; the values are already floats.
}

// ------------------------------------------------------------
// 3. Block DCT with Fibonacci block sizes (optional)
// ------------------------------------------------------------
/*
 * 1D DCT type-II (naive O(N^2) for small blocks; we assume block sizes are small Fibonacci numbers)
 */
__device__ float dct1d_coeff(const float* block, int n, int k) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += block[i] * cosf(M_PI * k * (2*i + 1) / (2.0f * n));
    }
    float scale = (k == 0) ? sqrtf(1.0f/n) : sqrtf(2.0f/n);
    return sum * scale;
}

__global__ void block_dct(float* data, int* block_starts, int* block_sizes, int num_blocks) {
    int bid = blockIdx.x;
    if (bid >= num_blocks) return;
    int start = block_starts[bid];
    int sz = block_sizes[bid];
    extern __shared__ float block[];
    // load block into shared
    for (int i = threadIdx.x; i < sz; i += blockDim.x) {
        block[i] = data[start + i];
    }
    __syncthreads();
    // compute DCT coefficients (in-place)
    for (int i = threadIdx.x; i < sz; i += blockDim.x) {
        float coeff = dct1d_coeff(block, sz, i);
        block[i] = coeff;
    }
    __syncthreads();
    // write back
    for (int i = threadIdx.x; i < sz; i += blockDim.x) {
        data[start + i] = block[i];
    }
}

__device__ float idct1d_coeff(const float* coeffs, int n, int i) {
    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
        float scale = (k == 0) ? sqrtf(1.0f/n) : sqrtf(2.0f/n);
        sum += coeffs[k] * scale * cosf(M_PI * k * (2*i + 1) / (2.0f * n));
    }
    return sum;
}

__global__ void block_idct(float* data, int* block_starts, int* block_sizes, int num_blocks) {
    int bid = blockIdx.x;
    if (bid >= num_blocks) return;
    int start = block_starts[bid];
    int sz = block_sizes[bid];
    extern __shared__ float block[];
    // load
    for (int i = threadIdx.x; i < sz; i += blockDim.x) {
        block[i] = data[start + i];
    }
    __syncthreads();
    // inverse DCT
    for (int i = threadIdx.x; i < sz; i += blockDim.x) {
        float val = idct1d_coeff(block, sz, i);
        block[i] = val;
    }
    __syncthreads();
    // write back
    for (int i = threadIdx.x; i < sz; i += blockDim.x) {
        data[start + i] = block[i];
    }
}

// ------------------------------------------------------------
// 4. Host-side Zeckendorf coding (sequential)
// ------------------------------------------------------------
// Precomputed Fibonacci numbers for host
static std::vector<int> fib_host = {1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765,10946,17711,28657,46368,75025,121393,196418,317811,514229,832040,1346269,2178309,3524578,5702887,9227465,14930352,24157817,39088169,63245986,102334155,165580141,267914296,433494437,701408733,1134903170,1836311903,2971215073U};

// Zeckendorf encoding (returns vector of bytes)
std::vector<unsigned char> zeckendorf_encode(int n) {
    if (n == 0) return {0}; // special marker
    std::vector<int> bits(fib_host.size(), 0);
    int idx = fib_host.size()-1;
    while (idx >= 0 && fib_host[idx] > n) idx--;
    while (idx >= 0) {
        if (fib_host[idx] <= n) {
            bits[idx] = 1;
            n -= fib_host[idx];
            idx--; // skip next (no consecutive ones)
        }
        idx--;
    }
    // remove leading zeros
    while (bits.size() > 1 && bits.back() == 0) bits.pop_back();
    std::reverse(bits.begin(), bits.end());
    // add termination '11'
    bits.push_back(1);
    bits.push_back(1);
    // convert to bytes
    std::vector<unsigned char> bytes;
    for (size_t i = 0; i < bits.size(); i += 8) {
        unsigned char byte = 0;
        for (int j = 0; j < 8 && i+j < bits.size(); j++) {
            byte = (byte << 1) | bits[i+j];
        }
        bytes.push_back(byte);
    }
    return bytes;
}

int zeckendorf_decode(const std::vector<unsigned char>& bytes, size_t& offset) {
    // extract bits from bytes starting at offset, scan for termination '11'
    std::vector<int> bits;
    while (offset < bytes.size()) {
        unsigned char b = bytes[offset++];
        for (int j = 7; j >= 0; j--) {
            bits.push_back((b >> j) & 1);
            if (bits.size() >= 2 && bits[bits.size()-2] == 1 && bits[bits.size()-1] == 1) {
                // remove termination bits
                bits.pop_back();
                bits.pop_back();
                // decode
                int val = 0;
                for (size_t i = 0; i < bits.size(); i++) {
                    if (bits[i]) val += fib_host[bits.size()-1 - i];
                }
                return val;
            }
        }
    }
    return 0; // not found
}

// ------------------------------------------------------------
// 5. Host functions for compression/decompression
// ------------------------------------------------------------
std::vector<unsigned char> compress(const std::vector<float>& data) {
    // 1. Copy to GPU
    int N = data.size();
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // 2. Apply φ-wavelet (blockwise using blocks of Fibonacci sizes)
    // For simplicity, we treat whole array as one block (must be a Fibonacci number; pad if needed)
    // We'll use a simple version: just call kernel with enough threads.
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    phi_wavelet_forward<<<grid_size, block_size, block_size*sizeof(float)>>>(d_data, N, 3);
    cudaDeviceSynchronize();

    // 3. Quantize
    quantize_fibonacci<<<grid_size, block_size>>>(d_data, N, INV_PHI2); // dead zone = 1/φ² ≈ 0.382
    cudaDeviceSynchronize();

    // 4. Copy quantized coefficients back to host
    std::vector<float> quantized(N);
    cudaMemcpy(quantized.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Encode to integers (we need to store the floating point values as ints)
    // Since quantized values are Fibonacci numbers (or zero), we can store them as ints.
    std::vector<int> ints(N);
    for (int i = 0; i < N; i++) {
        ints[i] = (int)quantized[i];
    }

    // 6. Zeckendorf encoding of the integer stream (with combined sign)
    std::vector<unsigned char> compressed;
    // Header: original length (4 bytes) + number of ints (4 bytes) but we know it's N.
    // We'll store a simple header: magic number, version, N.
    unsigned char header[12];
    *((uint32_t*)header) = 0x474F4C44; // "GOLD"
    *((uint32_t*)(header+4)) = 1; // version
    *((uint32_t*)(header+8)) = N;
    compressed.insert(compressed.end(), header, header+12);
    for (int v : ints) {
        // encode sign and magnitude: (abs(v)<<1) | (v<0)
        int enc = (abs(v) << 1) | (v < 0 ? 1 : 0);
        auto enc_bytes = zeckendorf_encode(enc);
        compressed.insert(compressed.end(), enc_bytes.begin(), enc_bytes.end());
    }
    cudaFree(d_data);
    return compressed;
}

std::vector<float> decompress(const std::vector<unsigned char>& compressed) {
    if (compressed.size() < 12) return {};
    // Check header
    uint32_t magic = *((uint32_t*)compressed.data());
    uint32_t version = *((uint32_t*)(compressed.data()+4));
    uint32_t N = *((uint32_t*)(compressed.data()+8));
    if (magic != 0x474F4C44 || version != 1) return {};

    // Decode Zeckendorf stream
    std::vector<int> ints;
    size_t offset = 12;
    while (offset < compressed.size() && ints.size() < N) {
        int enc = zeckendorf_decode(compressed, offset);
        ints.push_back(enc);
    }
    if (ints.size() != N) return {};

    // Convert back to float (dequantize)
    std::vector<float> quantized(N);
    for (int i = 0; i < N; i++) {
        int enc = ints[i];
        int sign = enc & 1;
        int abs_val = enc >> 1;
        quantized[i] = (float)(sign ? -abs_val : abs_val);
    }

    // Copy to GPU
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, quantized.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Inverse quantization (no change)
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    dequantize_fibonacci<<<grid_size, block_size>>>(d_data, N);
    cudaDeviceSynchronize();

    // Inverse φ-wavelet
    phi_wavelet_inverse<<<grid_size, block_size, block_size*sizeof(float)>>>(d_data, N, 3);
    cudaDeviceSynchronize();

    // Copy back
    std::vector<float> result(N);
    cudaMemcpy(result.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    return result;
}

// ------------------------------------------------------------
// 6. Simple test
// ------------------------------------------------------------
int main() {
    // Generate synthetic data: neural network weights (normal distribution)
    const int N = 1000000;
    std::vector<float> original(N);
    for (int i = 0; i < N; i++) {
        original[i] = (float)(sin(i * 0.001) * 2.0f); // some pattern
    }

    printf("Compressing %d floats...\n", N);
    auto compressed = compress(original);
    printf("Compressed size: %zu bytes\n", compressed.size());
    printf("Compression ratio: %.2f\n", (float)(N*sizeof(float)) / compressed.size());

    auto decompressed = decompress(compressed);
    if (decompressed.size() != N) {
        printf("Decompression failed: size mismatch\n");
        return 1;
    }

    // Compute cosine similarity
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (int i = 0; i < N; i++) {
        dot += original[i] * decompressed[i];
        norm1 += original[i] * original[i];
        norm2 += decompressed[i] * decompressed[i];
    }
    float sim = dot / sqrtf(norm1 * norm2);
    printf("Cosine similarity: %f\n", sim);
    return 0;
}
