#include "myutils.hpp"
#include "noise.hpp"
#include "display.hpp"

#define SZ(x) static_cast<size_t>(x)

/*
 * Steps for the Perlin (2d) noise:
 *
 * 1. Fill the grid with random gradient vectors
 * 2. For each point (x,y) of the world, calculate the dot product between the point and each of the
 *    corners of the grid cell the point falls into
 * 3. Interpolate the 4 obtained values for each point
 */
int main(int argc, char **argv) {

	auto seed = 0LLU;
	if (argc > 1) {
		seed = atoi(argv[1]);
	}

	const World world(3, 3);
	const size_t GRID_SIZE = SZ(world.width) * SZ(world.height);
	const size_t GRID_MEM_SIZE = sizeof(float) * 2 * GRID_SIZE;
	const size_t GRID_MEM_HALF_SIZE = sizeof(float) * GRID_SIZE;

	auto step = .05f;
	PerlinDisplayer displayer(world, step);

	auto dGrid = fillGrid(SZ(world.width), SZ(world.height), seed);

	float *dResult;
	MUST(cudaMalloc(&dResult, sizeof(float) * SZ(world.width / step) * SZ(world.height / step)));

	dim3 threads(nThreads(SZ(world.width / step), 32), nThreads(SZ(world.height / step), 32));
	dim3 blocks(nBlocks(SZ(world.width / step), 32), nBlocks(SZ(world.height / step), 32));
	std::cout << "Total " << (world.width/step)*(world.height/step) << ": using " << blocks << " blocks and " << threads << " threads." << std::endl;
	perlin<<<blocks, threads>>>(world, dGrid, step, dResult);
	/*shit<<<b, t>>>();*/
	cudaDeviceSynchronize();

	/*return;*/

	displayer.display(dResult);

	MUST(cudaFree(dResult));
	freeGrid(dGrid);
}
