/* ENGR-E 517 High Performance Computing
*  Original Author : Matt Anderson (Serial Implementation 2D)
*  Name : Ninaad Joshi (Serial and Parallel Implementation 1D)
*  Project : Demonstration of the 2D Heat Distribution
*  			 Problem using CUDA programming model
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
/*****************************************************************/
/* set the DEBUG flag to 1 to display the values for every iteration, 
*  or set to 0 for measuring time for both CPU and GPU
*/
#ifndef DEBUG
	#define DEBUG 0
#endif

/* set the DISPLAY flag to 1 to display the final matrix for CPU and GPU
*/
#ifndef DISPLAY
	#define DISPLAY 0
#endif
/****************************************************************/
#define TEMP 50.0
#define EPS 1e-6
#define I_FIX 5
#define J_FIX 5


#ifndef COLS
	#define COLS 100
#endif
#ifndef ROWS
	#define ROWS 100
#endif

#ifndef BLOCK_SIZE_X
	#define BLOCK_SIZE_X 32
#endif

#ifndef BLOCK_SIZE_Y
	#define BLOCK_SIZE_Y 32
#endif

double* alloc_matrix(){
    double* matrix;
    matrix = (double*) malloc(ROWS * COLS * sizeof(double));
    return matrix;
}

void init_matrix(double* matrix){
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++) {
            matrix[i * COLS + j] = 0.0;
        }
    matrix[I_FIX * COLS + J_FIX] = TEMP;
}

void print_matrix(double* matrix){
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++)
            printf("%3.7lf ", matrix[i * COLS + j]);
        printf("\n");
    }
}

void copy_matrix(double* dest, double* source) {
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++)
            dest[i * COLS + j] = source[i * COLS + j];
}

double max_abs(double* m1, double* m2){
    double max_val = DBL_MIN;
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++){
            if (fabs(m1[i * COLS + j] - m2[i * COLS + j]) > max_val) {
                max_val = fabs(m1[i * COLS + j] - m2[i * COLS + j]);
            }
        }
    return max_val;
}

/***********CPU***********/
void compute_new_values(double* old_matrix, double* new_matrix){
    for (int i = 1; i < ROWS-1; i++)
        for (int j= 1; j < COLS-1; j++)
            new_matrix[i * COLS + j] = 0.25 * (old_matrix[(i-1) * COLS + j] 
											+ old_matrix[(i+1) * COLS + j] 
											+ old_matrix[i * COLS + (j-1)] 
											+ old_matrix[i * COLS + (j+1)]);
    new_matrix[I_FIX * COLS + J_FIX] = TEMP;
}
/***********CPU***********/

/***********GPU***********/
__global__ void compute_new_values_gpu(const double* __restrict__ d_old_matrix,
	double* __restrict__ d_new_matrix){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i == I_FIX && j == J_FIX)
    	d_new_matrix[I_FIX * COLS + J_FIX] = TEMP;
	else if (0 < i && i < ROWS - 1 && 0 < j && j < COLS - 1)
		d_new_matrix[i * COLS + j] = 0.25 * (d_old_matrix[(i-1) * COLS + j] 
										+ d_old_matrix[(i+1) * COLS + j] 
										+ d_old_matrix[i * COLS + (j-1)] 
										+ d_old_matrix[i * COLS + (j+1)]);
}
/***********GPU***********/

/* Round the value of a / b to nearest higher integer value
*/
int divideUp(int n1, int n2) { 
	return (n1 % n2 != 0) ? (n1 / n2 + 1) : (n1 / n2); 
}

int main(int argc, char *argv[]) {
	//CPU
    double *a_old = alloc_matrix(); //allocate memory for the matrices
    double *a_new = alloc_matrix();
	struct timeval a_start, a_end;
	double tos_serial;
	// GPU
	long int iterations = 0, i = 0;
    double *h_in = alloc_matrix(); //allocate memory for the matrices
    double *h_out = alloc_matrix();
	int error;
	double *d_in;
	double *d_out;
	struct timeval h_start, h_end;
	double tos_cuda;

	printf("DISPLAY = %d DEBUG = %d ROWS = %d COLS = %d\n", DISPLAY, DEBUG, ROWS, COLS);
	printf("BLOCK_SIZE_X = %d BLOCK_SIZE_Y = %d\n", BLOCK_SIZE_X, BLOCK_SIZE_Y);
/*************************CPU**************************/
    init_matrix(a_old); //initialize the matrices
    init_matrix(a_new);
	printf("CPU: Starting the serial heat distribution\n");
 	
	if (DISPLAY || DEBUG){
		printf("CPU:The initial heat distribution matrix is:\n");
    	print_matrix(a_old);
	}

	gettimeofday(&a_start, NULL);
    while (1) {

        if (DEBUG)
            printf("\nCPU:Performing a new iteration...%ld\n", iterations);

        //compute new values and put them into a_new
        compute_new_values(a_old, a_new);

        if (DEBUG) {
            printf("CPU:a_old is:\n"); //output matrix to screen
            print_matrix(a_old);

            printf("CPU:a_new is:\n");
            print_matrix(a_new);
        }

        //calculate the maximum absolute differences among pairwise
        // differences of old and new matrix elements
        double max_diff = max_abs(a_old, a_new);

        if (DEBUG)
            printf("CPU:Max diff is: %f\n", max_diff);

        if (max_diff < EPS)
            break;

        copy_matrix(a_old, a_new); //assign values of a_new to a_old

        if (DEBUG)
            printf("CPU:End of iteration...%ld\n", iterations);
		++iterations;
    }
	gettimeofday(&a_end, NULL);
	tos_serial =  (a_end.tv_sec - a_start.tv_sec) + \
	(a_end.tv_usec - a_start.tv_usec)/1000000.0;

	printf("CPU:Time required is %.3e\n", tos_serial);
    if (DISPLAY || DEBUG){
		printf("CPU:The final heat distribution matrix is:\n");
    	print_matrix(a_new);
	}
	printf("The iterations performed by the serial code are %ld\n", iterations);

/*************************GPU**********************/
	printf("GPU:Starting the parallel heat distribution on CUDA\n");

    init_matrix(h_in); //initialize the matrices
    init_matrix(h_out);

	cudaMalloc((void **)&d_in, (size_t) ROWS * COLS * sizeof(double));
	error = cudaGetLastError();
	if (DEBUG)
		printf("GPU:d_in cudaMalloc error = %d\n", error);
   
	cudaMalloc((void **)&d_out, (size_t) ROWS * COLS * sizeof(double));
	error = cudaGetLastError();
	if (DEBUG)
		printf("GPU:d_out cudaMalloc error = %d\n", error);

	// copy data from host memory to device memory
	cudaMemcpy(d_in, h_in, ROWS * COLS * sizeof(double), cudaMemcpyHostToDevice);
	// copy data from device memory to device memory
	cudaMemcpy(d_out, d_in, ROWS * COLS * sizeof(double), cudaMemcpyDeviceToDevice);

    // block and grid dimensions
    dim3 blocks(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grids(divideUp(ROWS, BLOCK_SIZE_X), divideUp(COLS, BLOCK_SIZE_Y));
	
	gettimeofday(&h_start, NULL);
   	for(i = 0; i < iterations + 1; ++i) {


		//compute new values and put them into d_out
		compute_new_values_gpu<<<grids, blocks>>>(d_in, d_out);
		if (DEBUG){

            printf("GPU:Performing a new iteration...%ld\n", i);
			// copy data from device memory to host memory
			cudaMemcpy(h_in, d_in, ROWS * COLS * sizeof(double), cudaMemcpyDeviceToHost);
    		cudaMemcpy(h_out, d_out, ROWS * COLS * sizeof(double), cudaMemcpyDeviceToHost);
        
			printf("GPU:d_in is:\n"); //output d_in to screen
            print_matrix(h_in);

            printf("GPU:d_out is:\n"); //output d_out to screen
            print_matrix(h_out);
        	
			//calculate the maximum absolute differences among pairwise
        	// differences of old and new matrix elements
        	double max_diff = max_abs(h_in, h_out);

            printf("GPU:Max diff is: %f\n", max_diff);
        	if (max_diff < EPS)
            	break;
            printf("GPU:End of iteration...%ld\n", i);
        }
		// make the current d_out as d_in
		cudaMemcpy(d_in, d_out, ROWS * COLS * sizeof(double), cudaMemcpyDeviceToDevice);
    }
	gettimeofday(&h_end, NULL);

	// copy data from device memory to host memory
	cudaMemcpy(h_out, d_out, ROWS * COLS * sizeof(double), cudaMemcpyDeviceToHost);
	tos_cuda = (h_end.tv_sec - h_start.tv_sec) + \
	(h_end.tv_usec - h_start.tv_usec)/1000000.0;
	printf("GPU:Time required is %.3e seconds\n", tos_cuda);
	
	if (DISPLAY || DEBUG){
    	printf("GPU:The final heat distribution matrix is:\n");
    	print_matrix(h_out);
	}
	//calculate the maximum absolute differences among pairwise
	// differences of old and new matrix elements
    double max_diff = max_abs(h_out, a_new);

	printf("GPU:Max diff between serial and CUDA implementation is: %f\n",\
		max_diff);
	
	printf("Speed Up achieved is : %.3lf\n", tos_serial/tos_cuda);
	free(h_in);
	free(h_out);
	cudaFree(d_in);
	cudaFree(d_out);
    return 0;
}
