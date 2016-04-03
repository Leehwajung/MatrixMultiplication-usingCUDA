#include <iostream>
#include "MatrixMul.h"

#define BLOCKSIZE	32

using namespace std;

int compareCUDAandCPUMultiplycation(const dim3& dimsA, const dim3& dimsB);

/**
 * Program main
 */
int main() {

	dim3 dimsL(1 * BLOCKSIZE, 1 * BLOCKSIZE, 1);
	dim3 dimsH(100 * BLOCKSIZE, 100 * BLOCKSIZE, 1);

	int matrixResult = compareCUDAandCPUMultiplycation(dimsL, dimsL);
	if (matrixResult) {
		exit(matrixResult);
	}

	cout << endl << "-----" << endl;

	matrixResult = compareCUDAandCPUMultiplycation(dimsH, dimsH);
	exit(matrixResult);
}

int compareCUDAandCPUMultiplycation(const dim3& dimsA, const dim3& dimsB) {

	if (dimsA.x != dimsB.y) {
		cerr << "Error: outer matrix dimensions must be equal. (" << dimsA.x << " != " << dimsB.y << ")" << endl;
		exit(EXIT_FAILURE);
	}
	cout << "MatrixA(" << dimsA.x << "," << dimsA.y << "), MatrixB(" << dimsB.x << "," << dimsB.y << ")" << endl;

	int matrixResult = matrixMultiplyOnCUDA(BLOCKSIZE, dimsA, dimsB);
	if (matrixResult) {
		return(matrixResult);
	}

	cout << endl;

	return matrixMultiplyOnCPU(dimsA, dimsB);
}