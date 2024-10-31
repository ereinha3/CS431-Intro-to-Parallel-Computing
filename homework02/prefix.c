#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include "common.h"


void usage(int argc, char** argv);
void verify(int* sol, int* ans, int n);
void prefix_sum(int* src, int* prefix, int n);
void prefix_sum_p1(int* src, int* prefix, int n);
void prefix_sum_p2(int* src, int* prefix, int n);


int main(int argc, char** argv)
{
    // get inputs
    uint32_t n = 1048576;
    unsigned int seed = time(NULL);
    if(argc > 2) {
        n = atoi(argv[1]); 
        seed = atoi(argv[2]);
    } else if (argc > 1){
        usage(argc, argv);
	n = atoi(argv[1]);
        printf("using %"PRIu32" elements and time as seed\n", n);
    }


    // set up data 
    int* prefix_array = (int*) AlignedMalloc(sizeof(int) * n);  
    int* input_array = (int*) AlignedMalloc(sizeof(int) * n);
    srand(seed);
    for(int i = 0; i < n; i++) {
        input_array[i] = rand() % 100;
    }


    // set up timers
    uint64_t start_t;
    uint64_t end_t;
    InitTSC();


    // execute serial prefix sum and use it as ground truth
    start_t = ReadTSC();
    prefix_sum(input_array, prefix_array, n);
    end_t = ReadTSC();
    printf("Time to do O(N-1) prefix sum on a %"PRIu32" elements: %g (s)\n", 
           n, ElapsedTime(end_t - start_t));


    // execute parallel prefix sum which uses a NlogN algorithm
    int* input_array1 = (int*) AlignedMalloc(sizeof(int) * n);  
    int* prefix_array1 = (int*) AlignedMalloc(sizeof(int) * n);  
    memcpy(input_array1, input_array, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p1(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do O(NlogN) //prefix sum on a %"PRIu32" elements: %g (s)\n",
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);

    
    // execute parallel prefix sum which uses a 2(N-1) algorithm
    memcpy(input_array1, input_array, sizeof(int) * n);
    memset(prefix_array1, 0, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p2(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do 2(N-1) //prefix sum on a %"PRIu32" elements: %g (s)\n", 
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);


    // free memory
    AlignedFree(prefix_array);
    AlignedFree(input_array);
    AlignedFree(input_array1);
    AlignedFree(prefix_array1);


    return 0;
}

void usage(int argc, char** argv)
{
    fprintf(stderr, "usage: %s <# elements> <rand seed>\n", argv[0]);
}


void verify(int* sol, int* ans, int n)
{
    int err = 0;
    for(int i = 0; i < n; i++) {
        if(sol[i] != ans[i]) {
            err++;
        }
    }
    if(err != 0) {
        fprintf(stderr, "There was an error: %d\n", err);
    } else {
        fprintf(stdout, "Pass\n");
    }
}

void prefix_sum(int* src, int* prefix, int n)
{
    prefix[0] = src[0];
    for(int i = 1; i < n; i++) {
        prefix[i] = src[i] + prefix[i - 1];
    }
}

// Helpful explanation: https://en.wikipedia.org/wiki/Prefix_sum
void prefix_sum_p1(int* src, int* prefix, int n)
{
    // Two cases to store current and previous array
    int *twoDivides = (int *)malloc(n * sizeof(int));
    int *twoNotDivides = (int *)malloc(n * sizeof(int));
    #pragma omp parallel for
    for (int i=0; i<n; i++){
	// initializing
	prefix[i] = src[i];
    }
    for(int i = 0; i<=floor(log2(n)); i++){
	// index from previously moved (as we are in log2(n))
	int twoToI = 1 << (i);
	// use prev or curr based on conditional
	int* using = (i % 2) ? twoNotDivides : twoDivides;
	#pragma omp parallel for
	for(int j = 0; j<n; j++){
	    // update based on conditional (funneling to only update bigger than 2**i)
	    using[j] = (j < twoToI) ? prefix[j] : prefix[j] + prefix[j-twoToI];
	}
	#pragma omp parallel for
	for(int j =0; j<n; j++){
	    // updating original array
	    prefix[j] = using[j];
	}	
    }
    free(twoDivides);
    free(twoNotDivides);
}
void prefix_sum_p2(int* src, int* prefix, int n) {
    // can check if exists y in Z such that n = 2^y
    // if n = 2^y then n = 1... followed by y zeroes
    // so n = 1...0...0 & n-1 = 0...1...1 (0 followed by y 1s) = 
    // 1 0 0 0 ... 0 0 0 
    // 			&
    // 0 1 1 1 ... 1 1 1
    // -----------------
    // 0 0 0 0 0 0 0 0 0
    if (n & (n - 1)) {
        fprintf(stderr, "Error: n must be a power of 2.\n");
        return;
    }

    // allocate space for an array to store temp array and update each cycle so that there wont be WAR conflicts
    int* temp = malloc(n * sizeof(int));

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
	// initializing
        prefix[i] = src[i];
        temp[i] = src[i];
    }

    // n = 2^y so y steps
    int steps = log2(n);
    for (int i = 0; i < steps; i++) {
	// number of iterations will decline logarithmically
        int currentMax = 1 << (steps - i - 1);
        #pragma omp parallel for
        for (int j = 0; j < currentMax; j++) {
	    // currIndex is calculated using logarithmic indexing (every other, every fourth, every 8th element ...)
            int currIndex = (j + 1) * (1 << (i + 1)) - 1;
	    // if accessing every other , we know previous is -1 from curr; if accessing every fourth we know previous is -2 from curr; every 8th prev -4 from curr
            int prevIndex = currIndex - (1 << i);
	    // update temp to avoid collision in prefix
	    temp[currIndex] += prefix[prevIndex];
        }
	// embarassingly parallel so easy
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            prefix[j] = temp[j];
        }
    }
    // root of tree already summed so iterating over children to catch all unsummed remainders
    for (int i =0; i<steps-1; i++) {
	// 2 ^ (i+1) -1 cleanups at each temporal step based on previous nested fors
    	int currentMax = (1 << (i + 1)) - 1;
	#pragma omp parallel for
        for (int j = 0; j < currentMax; j++) {
	    // index where cumulative term comes from
            int prevIndex = (j + 1) * (1 << (steps - i - 1)) - 1;
	    // where accumulation term is stored
            int currIndex = prevIndex + (1 << (steps - i - 2));
	    // update temp to avoid collision in prefix
	    temp[currIndex] += prefix[prevIndex];
        }        
	// embarassingly parallel so easy
	#pragma omp parallel for
        for (int j = 0; j < n; j++) {
            prefix[j] = temp[j];
        }

    }

    free(temp);
}
