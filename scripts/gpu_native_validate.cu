// Small native CUDA correctness check for an authorized GPU allocation.
// Requires nvcc and NVIDIA driver only; no Python packages or downloads.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); \
        std::exit(1); \
    } \
} while (0)

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    if (count < 1) {
        std::fprintf(stderr, "No CUDA devices visible.\n");
        return 1;
    }
    CHECK(cudaSetDevice(0));
    cudaDeviceProp props{};
    CHECK(cudaGetDeviceProperties(&props, 0));
    size_t free_bytes = 0, total_bytes = 0;
    CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::printf("Visible CUDA devices: %d\nGPU: %s\n", count, props.name);
    std::printf("Compute capability: %d.%d\nFree memory: %.1f MiB\n",
                props.major, props.minor, free_bytes / 1048576.0);
    if (free_bytes < 512ULL * 1048576) {
        std::fprintf(stderr, "Insufficient free memory; skipping test.\n");
        return 1;
    }
    constexpr int n = 1 << 20;
    const size_t bytes = n * sizeof(float);
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = float(i % 100);
        b[i] = float(i % 37);
    }
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CHECK(cudaMalloc(&da, bytes));
    CHECK(cudaMalloc(&db, bytes));
    CHECK(cudaMalloc(&dc, bytes));
    CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i)
        add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));
    float elapsed = 0;
    CHECK(cudaEventElapsedTime(&elapsed, start, end));
    CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < n; ++i)
        if (c[i] != a[i] + b[i]) { correct = false; break; }
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
    CHECK(cudaFree(da));
    CHECK(cudaFree(db));
    CHECK(cudaFree(dc));
    std::printf("Elements: %d\nDevice arrays: %.1f MiB\n", n, 3 * bytes / 1048576.0);
    std::printf("Mean GPU kernel time: %.6f ms\n", elapsed / 10.0);
    std::printf("CPU/GPU correctness: %s\n", correct ? "PASS" : "FAIL");
    std::puts("Scope: vector addition only; not LLM or distributed inference.");
    return correct ? 0 : 1;
}
