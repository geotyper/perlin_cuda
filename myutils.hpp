#pragma once

#include <iostream>
#include <algorithm>

#define MUST(x) \
	do { \
		if (x != cudaSuccess) { \
			std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(x) << std::endl; \
			std::exit(1); \
		} \
	} while (false)

#define MUST_CRND(x) \
	do { \
		if (x != CURAND_STATUS_SUCCESS) { \
			std::cerr << "CURAND error at " << __FILE__ << ":" << __LINE__ << std::endl; \
			std::exit(1); \
		} \
	} while (false)

#define LIN(x, y, w) (y * w + x)

#define CUID(x) (blockDim.x * blockIdx.x + threadIdx.x)

#define CUDASSERT(cond, msg) \
	if (!(cond)) { \
		printf("assertion failed at " __FILE__ ":%d -> %s\n", __LINE__, msg); \
	}

template<typename T>
constexpr size_t nThreads(T n, size_t max = 1024) {
	return std::min(static_cast<size_t>(max), n);
}

template<typename T>
constexpr size_t nBlocks(T n, size_t max = 1024) {
	return static_cast<size_t>(std::ceil(n / nThreads(n, max)));
}

inline std::ostream& operator<<(std::ostream& o, dim3 d) {
	o << "(" << d.x << "," << d.y << "," << d.z << ")";
	return o;
}
