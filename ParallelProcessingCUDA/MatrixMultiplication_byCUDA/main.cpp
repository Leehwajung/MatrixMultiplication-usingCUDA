#include <iostream>
#include "MatrixMul.h"

#define BLOCKSIZE16	16
#define BLOCKSIZE32	32

using namespace std;

int compareCUDAandCPUMultiplycation(const int blockSize, const dim3& dimsA, const dim3& dimsB);

/**
 * Program main
 */
int main() {

	dim3 dimsL(10 * BLOCKSIZE32, 10 * BLOCKSIZE32, 1);
	dim3 dimsH(100 * BLOCKSIZE32, 100 * BLOCKSIZE32, 1);

	// Block size 32, matrix size 320
	int matrixResult = compareCUDAandCPUMultiplycation(BLOCKSIZE32, dimsL, dimsL);
	if (matrixResult) {
		exit(matrixResult);
	}

	cout << endl << "-----" << endl;

	// Block size 16, matrix size 320
	int matrixResult = compareCUDAandCPUMultiplycation(BLOCKSIZE16, dimsL, dimsL);
	if (matrixResult) {
		exit(matrixResult);
	}

	cout << endl << "-----" << endl;

	// Block size 32, matrix size 3200
	matrixResult = compareCUDAandCPUMultiplycation(BLOCKSIZE32, dimsH, dimsH);
	exit(matrixResult);
}

int compareCUDAandCPUMultiplycation(const int blockSize, const dim3& dimsA, const dim3& dimsB) {

	if (dimsA.x != dimsB.y) {
		cerr << "Error: outer matrix dimensions must be equal. (" << dimsA.x << " != " << dimsB.y << ")" << endl;
		exit(EXIT_FAILURE);
	}
	cout << "MatrixA(" << dimsA.x << "," << dimsA.y << "), MatrixB(" << dimsB.x << "," << dimsB.y << ")" << endl;

	int matrixResult = matrixMultiplyUsingCUDA(blockSize, dimsA, dimsB);
	if (matrixResult) {
		return(matrixResult);
	}

	cout << endl;

	return matrixMultiplyOnCPU(dimsA, dimsB);
}