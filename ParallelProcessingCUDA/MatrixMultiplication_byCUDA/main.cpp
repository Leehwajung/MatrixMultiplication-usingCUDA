/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

#include <iostream>
#include "MatrixMul.h"

using namespace std;

/**
 * Program main
 */

int main() {

	int blockSize = 32;

	dim3 dimsA(5 * 2 * blockSize, 5 * 2 * blockSize, 1);
	dim3 dimsB(5 * 4 * blockSize, 5 * 2 * blockSize, 1);

	if (dimsA.x != dimsB.y) {
		cerr << "Error: outer matrix dimensions must be equal. (" << dimsA.x << " != " << dimsB.y << ")" << endl;
		exit(EXIT_FAILURE);
	}
	cout << "MatrixA(" << dimsA.x << "," << dimsA.y << "), MatrixB(" << dimsB.x << "," << dimsB.y << ")" << endl;

	int matrixResult = matrixMultiplyUsingCUDA(blockSize, dimsA, dimsB);
	if (matrixResult) {
		exit(matrixResult);
	}

	matrixResult = matrixMultiplyUsingCPU(dimsA, dimsB);

	exit(matrixResult);
}