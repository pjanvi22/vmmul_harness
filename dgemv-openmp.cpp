#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

const char* dgemv_desc = "OpenMP dgemv.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */

void my_dgemv(int n, double* A, double* x, double* y) {

   #pragma omp parallel
   {
      int nthreads = omp_get_num_threads();
      int thread_id = omp_get_thread_num();
   
      int chunk_size = n/nthreads;
      int start = thread_id*chunk_size;
      int end = (thread_id==nthreads-1)?n:start+chunk_size;
   
       for(int i=start;i<end;i++){
         double sum = 0.0;
         for(int j=0;j<n;j++){
            sum+=A[i*n+j]*x[j];
         }
         #pragma omp atomic
         y[i]+=sum;
      }
   }

   // insert your dgemv code here. you may need to create additional parallel regions,
   // and you may want to comment out the above parallel code block that prints out
   // nthreads and thread_id so as to not taint your timings

}

