#include <iostream>
#include "MatrixMul.h"

using namespace std;

int MatrixMultiply(const dim3& dimsA, const dim3& dimsB);


/**
 * Program main
 */
int main() {

	dim3 dimsL(BLOCKSIZE, BLOCKSIZE, 1);
	dim3 dimsH(50 * BLOCKSIZE, 50 * BLOCKSIZE, 1);

	// matrix size 32 * 32
	int matrixResult = MatrixMultiply(dimsL, dimsL);
	if (matrixResult) {
		exit(matrixResult);
	}

	cout << endl << "-----" << endl;

	// matrix size 1600 * 1600
	matrixResult = MatrixMultiply(dimsH, dimsH);
	exit(matrixResult);
}

int MatrixMultiply(const dim3& dimsA, const dim3& dimsB) {

	if (dimsA.x != dimsB.y) {
		cerr << "Error: outer matrix dimensions must be equal. (" << dimsA.x << " != " << dimsB.y << ")" << endl;
		exit(EXIT_FAILURE);
	}
	cout << "MatrixA(" << dimsA.x << "," << dimsA.y << "), MatrixB(" << dimsB.x << "," << dimsB.y << ")" << endl;

	int matrixResult = matrixMultiplyUsingCUDA(dimsA, dimsB);
	if (matrixResult) {
		return(matrixResult);
	}

	cout << endl;

	matrixResult = matrixMultiplyUsingCPU(dimsA, dimsB);
	if (matrixResult) {
		return(matrixResult);
	}

	cout << endl;

	return matrixMultiplyUsingCUBLAS(dimsA, dimsB);
}