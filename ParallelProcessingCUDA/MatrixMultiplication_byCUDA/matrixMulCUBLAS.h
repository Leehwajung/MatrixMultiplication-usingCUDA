#pragma once

#ifndef _matrixSize2
#define _matrixSize2
typedef struct _matrixSize2      // Optional Command-line multiplier for matrix sizes
{
	unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;


void initializeCUDA(int argc, char **argv, int &devID, int &iSizeMultiple, sMatrixSize &matrix_size);
int matrixMultiply(int argc, char **argv, int devID, sMatrixSize &matrix_size);
#endif