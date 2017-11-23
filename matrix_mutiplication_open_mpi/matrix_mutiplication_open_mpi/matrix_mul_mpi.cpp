#include "stdafx.h"
#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<omp.h>

#define _CRT_SECURE_NO_WARNINGS 1

MPI_Status status;
MPI_Request request;
double start_time, end_time;
#define NUM_THREADS 4
int omp_num_threads_used = 0;

struct matrix{
	unsigned int rows;
	unsigned int cols;
	double **data;
};

matrix * get_zero_filled_matrix(unsigned int dim_x, unsigned int dim_y) {
	matrix * m = (matrix*)malloc(sizeof(matrix));
	m->rows = dim_x;
	m->cols = dim_y;
	m->data = (double**)calloc(m->rows, sizeof(double*));
	for (unsigned int i = 0; i < m->rows; i++) {
		*(m->data + i) = (double*)calloc(m->cols, sizeof(double));
	}
	return m;
}

matrix * read_matrix(char file_path[]) {
	matrix *m = (matrix*)malloc(sizeof(matrix));
	FILE *file_handler = fopen(file_path, "r");

	if (NULL == file_handler) {
		printf("FILE MATRIX ERROR \n");
		exit(11);
	}
	m->rows = 0;
	m->cols = 0;
	int ch = 0;
	do {
		ch = fgetc(file_handler);
		if (m->rows == 0 && ch == '\t') {
			m->cols += 1;
		}
		if (ch == '\n') {
			m->rows += 1;
		}
	}while (ch != EOF);

	m->cols += 1;
	
	m->data = (double**)calloc(m->rows, sizeof(double*));
	for (unsigned int i = 0; i < m->rows; i++) {
		*(m->data + i) = (double*)calloc(m->cols, sizeof(double));
	}
	rewind(file_handler);
	
	for (unsigned int i = 0; i < m->rows; i++) {
		for (unsigned int j = 0; j < m->cols; j++) {
			if (!fscanf(file_handler, "%lf", *(m->data+i)+j)) {
				break;
			}
		}
	}

	fclose(file_handler);
	return m;
}

void serialMatrixMultiplication(matrix *matrixA, matrix *matrixB, matrix *matrixR) {
	unsigned k = 0;
	double start = MPI_Wtime();
	for (unsigned int i = 0; i < matrixA->rows; i++) {
		for (unsigned int j = 0; j < matrixA->cols; j++) {
			for (k = 0; k < matrixA->rows; k++) {
				*(*(matrixR->data + i) + j) += *(*(matrixA->data + i) + k) * *(*(matrixB->data + k) + j);
			}
		}
	}
	double end = MPI_Wtime();
	printf("Serial Calculation %f \n", end - start);
}

void parallelMatrixMultiplication(matrix *matrixA, matrix *matrixB, matrix *matrixR) {	
	unsigned k = 0;
	double start = MPI_Wtime();
	#pragma omp parallel for
	for (int i = 0; i < matrixA->rows; i++) {		
		for (unsigned int j = 0; j < matrixA->cols; j++) {
			for (k = 0; k < matrixA->rows; k++) {
				*(*(matrixR->data + i) + j) += *(*(matrixA->data + i) + k) * *(*(matrixB->data + k) + j);
			}
		}
	}
	double end = MPI_Wtime();
	//omp_num_threads_used = omp_get_num_threads();
	//printf("Expected to use %d threads but used %d \n", NUM_THREADS, omp_num_threads_used);
	printf("Parallel Time %f \n", end - start);
}

void parallelMatrixMultiplicationStaticThreads(matrix *matrixA, matrix *matrixB, matrix *matrixR, int threads) {	
	unsigned k = 0;
	double start = MPI_Wtime();
	#pragma omp parallel for schedule(static) num_threads(threads)
	for (int i = 0; i < matrixA->rows; i++) {
		for (unsigned int j = 0; j < matrixA->cols; j++) {
			for (k = 0; k < matrixA->rows; k++) {
				*(*(matrixR->data + i) + j) += *(*(matrixA->data + i) + k) * *(*(matrixB->data + k) + j);
			}
		}
	}
	double end = MPI_Wtime();
	//omp_num_threads_used = omp_get_num_threads();
	//printf("Expected to use %d threads but used %d \n", NUM_THREADS, omp_num_threads_used);
	printf("Parallel Time %f, threads %d \n", end - start, threads);
}


void parallelMatrixMultiplicationStaticChunk(matrix *matrixA, matrix *matrixB, matrix *matrixR, int chunk) {	
	unsigned k = 0;
	double start = MPI_Wtime();
	#pragma omp for schedule (static, chunk)
	for (int i = 0; i < matrixA->rows; i++) {
		for (unsigned int j = 0; j < matrixA->cols; j++) {
			for (k = 0; k < matrixA->rows; k++) {
				*(*(matrixR->data + i) + j) += *(*(matrixA->data + i) + k) * *(*(matrixB->data + k) + j);
			}
		}
	}
	double end = MPI_Wtime();
	//omp_num_threads_used = omp_get_num_threads();
	//printf("Expected to use %d threads but used %d \n", NUM_THREADS, omp_num_threads_used);
	printf("Parallel Time %f, chunk %d \n", end - start, chunk);
}

void parallelMatrixMultiplicationDynamicThreads(matrix *matrixA, matrix *matrixB, matrix *matrixR, int threads) {	
	unsigned k = 0;
	double start = MPI_Wtime();
	#define CHUNK 10
	#pragma omp parallel for schedule(dynamic, CHUNK) num_threads(threads)
	for (int i = 0; i < matrixA->rows; i++) {
		for (unsigned int j = 0; j < matrixA->cols; j++) {
			for (k = 0; k < matrixA->rows; k++) {
				*(*(matrixR->data + i) + j) += *(*(matrixA->data + i) + k) * *(*(matrixB->data + k) + j);
			}
		}
	}
	double end = MPI_Wtime();
	//omp_num_threads_used = omp_get_num_threads();
	//printf("Expected to use %d threads but used %d \n", NUM_THREADS, omp_num_threads_used);
	printf("Parallel Time %f, threads %d \n", end - start, threads);
}

void printMatrix(matrix * m) {
	for (unsigned int i = 0; i < m->rows; i++) {
		for (unsigned int j = 0; j < m->cols; j++) {
			printf("%f ", *(*(m->data + i) + j));
		}
		printf("\n");
	}
	printf("\n");
}

void freeMatrix(double **matrix, unsigned int ROWS) {
	for (unsigned int i = 0; i < ROWS; i++) {
		delete[] *(matrix + i);
	}
	delete[] matrix;
}



int main(int argc, char* argv[]) {
	char matrix_4[] = "C:\\Users\\uidj6605\\PycharmProjects\\dipi_tools\\4_4_random.txt";
	char matrix_50[] = "C:\\Users\\uidj6605\\PycharmProjects\\dipi_tools\\50_50_random.txt";
	char matrix_100[] = "C:\\Users\\uidj6605\\PycharmProjects\\dipi_tools\\100_100_random.txt";
	char matrix_500[] = "C:\\Users\\uidj6605\\PycharmProjects\\dipi_tools\\500_500_random.txt";
	char matrix_1k[] = "C:\\Users\\uidj6605\\PycharmProjects\\dipi_tools\\1000_1000_random.txt";
	char matrix_1_5k[] = "C:\\Users\\uidj6605\\PycharmProjects\\dipi_tools\\1500_1500_random.txt";
	char matrix_2k[] = "C:\\Users\\uidj6605\\PycharmProjects\\dipi_tools\\2000_2000_random.txt";
	char matrix_3k[] = "C:\\Users\\uidj6605\\PycharmProjects\\dipi_tools\\3000_3000_random.txt";

	matrix * m = read_matrix(matrix_1k);
	matrix * r = get_zero_filled_matrix(1000, 1000);

	MPI_Init(&argc, &argv);

	//parallelMatrixMultiplication(m, m, r);
	//serialMatrixMultiplication(m, m, r);	
	//printMatrix(r);
	int threadsNums[] = { 8, 16, 64, 128, 512, 1024 };	
	int chunckValues[] = { 1, 5, 10, 20 };
	for (int i = 0; i < 4; i++) {
		//parallelMatrixMultiplicationStaticThreads(m, m, r, threadsNums[i]);
		//parallelMatrixMultiplicationStaticChunk(m, m, r, chunckValues[i]); // but chunk sizes
		parallelMatrixMultiplicationDynamicThreads(m, m, r, chunckValues[i]);
	}			

	MPI_Finalize();
	
	/*	

	double ** matrixA, **matrixB, **matrixR;
	unsigned int COLS = 1000, ROWS = 1000;
	matrixA = new double*[ROWS];
	matrixB = new double*[ROWS];
	matrixR = new double*[ROWS];
	for(unsigned int i = 0; i < ROWS; i++) {
		*(matrixA + i) = new double[COLS];
		*(matrixB + i) = new double[COLS];
		*(matrixR + i) = new double[COLS];
		for (unsigned int j = 0; j < COLS; j++) {
			*(*(matrixA + i) + j) = j + 1;
			*(*(matrixB + i) + j) = j + 1;
			*(*(matrixR + i) + j) = j + 1;
		}
	}
	//printMatrix(matrixA, ROWS, COLS);
	//printMatrix(matrixB, ROWS, COLS);
	
	serialMatrixMultiplication(matrixA, matrixB, matrixR, ROWS, COLS);
	parallelMatrixMultiplication(matrixA, matrixB, matrixR, ROWS, COLS);

	//printMatrix(matrixR, ROWS, COLS);	
	freeMatrix(matrixA, ROWS);
	freeMatrix(matrixB, ROWS);
	freeMatrix(matrixR, ROWS);
	*/
	//getchar();
}