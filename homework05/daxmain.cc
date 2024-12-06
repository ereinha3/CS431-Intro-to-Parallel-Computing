#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>
#include "main.h"

#define MAX_FILENAME 256
#define MAX_NUM_LENGTH 100

#define NUM_TIMERS       7
#define LOAD_TIME        0
#define VEC_BCAST_TIME   1
#define MAT_SCATTER_TIME 2
#define LOCK_INIT_TIME   3
#define SPMV_COO_TIME    4
#define RES_REDUCE_TIME  5
#define STORE_TIME       6
#define NUM_ITER         20


int main(int argc, char** argv)
{
   // Read the sparse matrix and store it in row_ind, col_ind, and val,
   // also known as co-ordinate format (COO).
   int m;
   int n;
   int nnz;
   int* row_ind;
   int* col_ind;
   double* val;
   double* vector_x;
   double* res1;
   double* res2;
   double* res3;

   // timer
   double start;
   double end;

   // program info
   usage(argc, argv);


   double timer[NUM_TIMERS];
   for(unsigned int i = 0; i < NUM_TIMERS; i++) {
       timer[i] = 0.0;
   }

   // Initialize MPI
   MPI_Init(NULL, NULL);

   // Current rank's ID
   int world_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
   // Total number of ranks
   int world_size;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);

   // Rank 0 loads the sparse matrix from a file and distributes it
   if(world_rank == 0) {
       start = MPI_Wtime();

       // Read the sparse matrix file name
       char matrixName[MAX_FILENAME];
       strcpy(matrixName, argv[1]);
       int is_symmetric = 0;
       read_info(matrixName, &is_symmetric);

       int ret;
       MM_typecode matcode;

       // load and expand sparse matrix from file (if symmetric)
       fprintf(stdout, "Matrix file name: %s ... ", matrixName);

       ret = mm_read_mtx_crd(matrixName, &m, &n, &nnz, &row_ind, &col_ind, 
                             &val, &matcode);
       check_mm_ret(ret);
       if(is_symmetric) {
           expand_symmetry(m, n, &nnz, &row_ind, &col_ind, &val);
       }

       // Load the input vector file
       char vectorName[MAX_FILENAME];
       strcpy(vectorName, argv[2]);
       fprintf(stdout, "Vector file name: %s ... ", vectorName);
       unsigned int vector_size;
       read_vector(vectorName, &vector_x, &vector_size);
       assert((unsigned int) n == vector_size);
       fprintf(stdout, "file loaded\n");

       end = MPI_Wtime();
       timer[LOAD_TIME] = end - start;

   } 


   #if 0
   // Calculate SpMV using naive distributed COO
   spmv_coo_naive(world_rank, world_size, row_ind, col_ind, val, m, n, nnz, 
                  vector_x, &res1, timer);

   // Store the calculated vector in a file, one element per line.
   if(world_rank == 0) {
       start = MPI_Wtime();
       char resName[MAX_FILENAME];
       strcpy(resName, argv[3]); 
       fprintf(stdout, "Result file name: %s ... ", resName);
       store_result(resName, res1, m);
       fprintf(stdout, "file saved\n");
       end = MPI_Wtime();
       timer[STORE_TIME] = end - start;
   }

   // print timer
   if(world_rank == 0) {
       print_time(timer);
   }
   #endif

   #if 1
   // Clculate SpMV using 2-D grid of processors and COO
   for (int i = 0; i < NUM_ITER; i++){
       spmv_coo_2d(world_rank, world_size, row_ind, col_ind, val, m, n, nnz,
               vector_x, &res2, timer, i);
   }
   // Store the calculated vector in a file, one element per line.
   if(world_rank == 0) {
       char resName[MAX_FILENAME];
       strcpy(resName, argv[3]); 
       fprintf(stdout, "Result file name: %s ... ", resName);
   start = MPI_Wtime();
   store_result(resName, res2, m);
   end = MPI_Wtime();
   timer[STORE_TIME] = end - start;
       fprintf(stdout, "file saved\n");
       
   print_time(timer);
       free(res2);
   }
   #endif



   MPI_Finalize();

   return 0;
}


/* This function checks the number of input parameters to the program to make 
  sure it is correct. If the number of input parameters is incorrect, it 
  prints out a message on how to properly use the program.
  input parameters:
      int    argc
      char** argv 
  return parameters:
      none
*/
void usage(int argc, char** argv)
{
   if(argc < 4) {
       fprintf(stderr, "usage: %s <matrix> <vector> <result>\n", argv[0]);
       exit(EXIT_FAILURE);
   } 
}

/* This function prints out information about a sparse matrix
  input parameters:
      char*       fileName    name of the sparse matrix file
      MM_typecode matcode     matrix information
      int         m           # of rows
      int         n           # of columns
      int         nnz         # of non-zeros
  return paramters:
      none
*/
void print_matrix_info(char* fileName, MM_typecode matcode, 
                      int m, int n, int nnz)
{
   fprintf(stdout, "-----------------------------------------------------\n");
   fprintf(stdout, "Matrix name:     %s\n", fileName);
   fprintf(stdout, "Matrix size:     %d x %d => %d\n", m, n, nnz);
   fprintf(stdout, "-----------------------------------------------------\n");
   fprintf(stdout, "Is matrix:       %d\n", mm_is_matrix(matcode));
   fprintf(stdout, "Is sparse:       %d\n", mm_is_sparse(matcode));
   fprintf(stdout, "-----------------------------------------------------\n");
   fprintf(stdout, "Is complex:      %d\n", mm_is_complex(matcode));
   fprintf(stdout, "Is real:         %d\n", mm_is_real(matcode));
   fprintf(stdout, "Is integer:      %d\n", mm_is_integer(matcode));
   fprintf(stdout, "Is pattern only: %d\n", mm_is_pattern(matcode));
   fprintf(stdout, "-----------------------------------------------------\n");
   fprintf(stdout, "Is general:      %d\n", mm_is_general(matcode));
   fprintf(stdout, "Is symmetric:    %d\n", mm_is_symmetric(matcode));
   fprintf(stdout, "Is skewed:       %d\n", mm_is_skew(matcode));
   fprintf(stdout, "Is hermitian:    %d\n", mm_is_hermitian(matcode));
   fprintf(stdout, "-----------------------------------------------------\n");

}


/* This function checks the return value from the matrix read function, 
  mm_read_mtx_crd(), and provides descriptive information.
  input parameters:
      int ret    return value from the mm_read_mtx_crd() function
  return paramters:
      none
*/
void check_mm_ret(int ret)
{
   switch(ret)
   {
       case MM_COULD_NOT_READ_FILE:
           fprintf(stderr, "Error reading file.\n");
           exit(EXIT_FAILURE);
           break;
       case MM_PREMATURE_EOF:
           fprintf(stderr, "Premature EOF (not enough values in a line).\n");
           exit(EXIT_FAILURE);
           break;
       case MM_NOT_MTX:
           fprintf(stderr, "Not Matrix Market format.\n");
           exit(EXIT_FAILURE);
           break;
       case MM_NO_HEADER:
           fprintf(stderr, "No header information.\n");
           exit(EXIT_FAILURE);
           break;
       case MM_UNSUPPORTED_TYPE:
           fprintf(stderr, "Unsupported type (not a matrix).\n");
           exit(EXIT_FAILURE);
           break;
       case MM_LINE_TOO_LONG:
           fprintf(stderr, "Too many values in a line.\n");
           exit(EXIT_FAILURE);
           break;
       case MM_COULD_NOT_WRITE_FILE:
           fprintf(stderr, "Error writing to a file.\n");
           exit(EXIT_FAILURE);
           break;
       case 0:
           fprintf(stdout, "file loaded.\n");
           break;
       default:
           fprintf(stdout, "Error - should not be here.\n");
           exit(EXIT_FAILURE);
           break;

   }
}

/* This function reads information about a sparse matrix using the 
  mm_read_banner() function and printsout information using the
  print_matrix_info() function.
  input parameters:
      char*       fileName    name of the sparse matrix file
  return paramters:
      none
*/
void read_info(char* fileName, int* is_sym)
{
   FILE* fp;
   MM_typecode matcode;
   int m;
   int n;
   int nnz;

   if((fp = fopen(fileName, "r")) == NULL) {
       fprintf(stderr, "Error opening file: %s\n", fileName);
       exit(EXIT_FAILURE);
   }

   if(mm_read_banner(fp, &matcode) != 0)
   {
       fprintf(stderr, "Error processing Matrix Market banner.\n");
       exit(EXIT_FAILURE);
   } 

   if(mm_read_mtx_crd_size(fp, &m, &n, &nnz) != 0) {
       fprintf(stderr, "Error reading size.\n");
       exit(EXIT_FAILURE);
   }

   print_matrix_info(fileName, matcode, m, n, nnz);
   *is_sym = mm_is_symmetric(matcode);

   fclose(fp);
}

/* This function converts a sparse matrix stored in COO format to CSR format.
  input parameters:
      int* row_ind     list or row indices (per non-zero)
      int* col_ind     list or col indices (per non-zero)
      double*  val     list or values  (per non-zero)
      int  m       # of rows
      int  n       # of columns
      int  n       # of non-zeros
  output parameters:
      unsigned int**   csr_row_ptr pointer to row pointers (per row)
      unsigned int**   csr_col_ind pointer to column indices (per non-zero)
      double**     csr_vals    pointer to values (per non-zero)
  return paramters:
      none
*/
void convert_coo_to_csr(int* row_ind, int* col_ind, double* val, 
                       int m, int n, int nnz,
                       unsigned int** csr_row_ptr, unsigned int** csr_col_ind,
                       double** csr_vals)

{
}

/* Reads in a vector from file.
  input parameters:
      char*    fileName    name of the file containing the vector
  output parameters:
      double** vector      pointer to the vector
      int* vecSize     pointer to # elements in the vector
  return parameters:
      none
*/
void read_vector(char* fileName, double** vector, unsigned int* vecSize)
{
   FILE* fp = fopen(fileName, "r");
   assert(fp);
   char line[MAX_NUM_LENGTH];    
   fgets(line, MAX_NUM_LENGTH, fp);
   fclose(fp);

   unsigned int vector_size = atoi(line);
   double* vector_ = (double*) malloc(sizeof(double) * vector_size);

   fp = fopen(fileName, "r");
   assert(fp); 
   // first read the first line to get the # elements
   fgets(line, MAX_NUM_LENGTH, fp);

   unsigned int index = 0;
   while(fgets(line, MAX_NUM_LENGTH, fp) != NULL) {
       vector_[index] = atof(line); 
       index++;
   }

   fclose(fp);
   assert((unsigned int) index == vector_size);

   *vector = vector_;
   *vecSize = vector_size;
}

/* SpMV function for COO stored sparse matrix
*/
void spmv_coo(int* row_ind, int* col_ind, double* vals, 
             int m, int n, int nnz, double* vector_x, double *res, 
             omp_lock_t* writelock)
{
   // first initialize res to 0
   #pragma omp parallel for schedule(static)
   for(int i = 0; i < m; i++) {
       res[i] = 0.0;
   }

   // calculate spmv
   #pragma omp parallel for schedule(static)
   for(int i = 0; i < nnz; i++) {
       double tmp = vals[i] * vector_x[col_ind[i] - 1];
       omp_set_lock(&(writelock[row_ind[i] - 1]));
       res[row_ind[i] - 1] += tmp;
       omp_unset_lock(&(writelock[row_ind[i] - 1]));
   }
}


/* Save result vector in a file
*/
void store_result(char *fileName, double* res, int m)
{
   FILE* fp = fopen(fileName, "w");
   assert(fp);

   fprintf(fp, "%d\n", m);
   for(int i = 0; i < m; i++) {
       fprintf(fp, "%0.10f\n", res[i]);
   }

   fclose(fp);
}

/* Print timing information 
*/
void print_time(double timer[])
{
   fprintf(stdout, "Module\t\tTime\n");
   fprintf(stdout, "Load\t\t");
   fprintf(stdout, "%f\n", timer[LOAD_TIME]);
   fprintf(stdout, "Vec Bcast\t");
   fprintf(stdout, "%f\n", timer[VEC_BCAST_TIME]/NUM_ITER);
   fprintf(stdout, "Mat Scatter\t");
   fprintf(stdout, "%f\n", timer[MAT_SCATTER_TIME]/NUM_ITER);
   fprintf(stdout, "Lock Init\t");
   fprintf(stdout, "%f\n", timer[LOCK_INIT_TIME]/NUM_ITER);
   fprintf(stdout, "COO SpMV\t");
   fprintf(stdout, "%f\n", timer[SPMV_COO_TIME]/NUM_ITER);
   fprintf(stdout, "Res Reduce\t");
   fprintf(stdout, "%f\n", timer[RES_REDUCE_TIME]/NUM_ITER);
   fprintf(stdout, "Store\t\t");
   fprintf(stdout, "%f\n", timer[STORE_TIME]);
}

void expand_symmetry(int m, int n, int* nnz_, int** row_ind, int** col_ind, 
                    double** val)
{
   fprintf(stdout, "Expanding symmetric matrix ... ");
   int nnz = *nnz_;

   // first, count off-diagonal non-zeros
   int not_diag = 0;
   for(int i = 0; i < nnz; i++) {
       if((*row_ind)[i] != (*col_ind)[i]) {
           not_diag++;
       }
   }

   int* _row_ind = (int*) malloc(sizeof(int) * (nnz + not_diag));
   assert(_row_ind);
   int* _col_ind = (int*) malloc(sizeof(int) * (nnz + not_diag));
   assert(_col_ind);
   double* _val = (double*) malloc(sizeof(double) * (nnz + not_diag));
   assert(_val);

   memcpy(_row_ind, *row_ind, sizeof(int) * nnz);
   memcpy(_col_ind, *col_ind, sizeof(int) * nnz);
   memcpy(_val, *val, sizeof(double) * nnz);
   int index = nnz;
   for(int i = 0; i < nnz; i++) {
       if((*row_ind)[i] != (*col_ind)[i]) {
           _row_ind[index] = (*col_ind)[i];
           _col_ind[index] = (*row_ind)[i];
           _val[index] = (*val)[i];
           index++;
       }
   }
   assert(index == (nnz + not_diag));

   free(*row_ind);
   free(*col_ind);
   free(*val);

   *row_ind = _row_ind;
   *col_ind = _col_ind;
   *val = _val;
   *nnz_ = nnz + not_diag;

   fprintf(stdout, "done\n");
   fprintf(stdout, "  Total # of non-zeros is %d\n", nnz + not_diag);
}

void init_locks(omp_lock_t** locks, int m)
{
   omp_lock_t* _locks = (omp_lock_t*) malloc(sizeof(omp_lock_t) * m);
   assert(_locks);
   for(int i = 0; i < m; i++) {
       omp_init_lock(&(_locks[i]));
   }
   *locks = _locks;
}

void destroy_locks(omp_lock_t* locks, int m)
{
   assert(locks);
   for(int i = 0; i < m; i++) {
       omp_destroy_lock(&(locks[i]));
   }
   free(locks);
}



void spmv_coo_naive(int world_rank, int world_size, int* row_ind, int* col_ind,                     double* val, int m, int n, int nnz, double* vector_x, 
                   double** res, double timer[])
{
   double start;
   double end;


   // Rank 0 now determines how work will be distributed among the ranks
   int nnz_per_rank = 0;
   if(world_rank == 0) {
       nnz_per_rank = (nnz + world_size - 1) / world_size;
   }
   start = MPI_Wtime();
   // Broadcast this to everyone
   MPI_Bcast(&nnz_per_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
   // Also broadcast m and n
   MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
   // Lastly, send the input vector x
   if(world_rank != 0) {
       vector_x = (double*) malloc(sizeof(double) * n);
       assert(vector_x);
   }
   MPI_Bcast(vector_x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   end = MPI_Wtime();
   timer[VEC_BCAST_TIME] = end - start;


   // Now, let's send the sparse matrix
   // First, pad the data so that we can use MPI_Scatter instead of 
   // MPI_Scatterv
   if(world_rank == 0) {
       // new_nnz will be larger than nnz
       int new_nnz = nnz_per_rank * world_size;

       int* row_ind_tmp = (int*) malloc(sizeof(int) * new_nnz);
       assert(row_ind_tmp);
       memset(row_ind_tmp, 0, sizeof(int) * new_nnz);

       int* col_ind_tmp = (int*) malloc(sizeof(int) * new_nnz);
       assert(col_ind_tmp);
       memset(col_ind_tmp, 0, sizeof(int) * new_nnz);

       double* val_tmp = (double*) malloc(sizeof(double) * new_nnz);
       assert(val_tmp);
       memset(val_tmp, 0, sizeof(double) * new_nnz);

       memcpy(row_ind_tmp, row_ind, sizeof(int) * nnz);
       memcpy(col_ind_tmp, col_ind, sizeof(int) * nnz);
       memcpy(val_tmp, val, sizeof(double) * nnz);

       free(row_ind);
       free(col_ind);
       free(val);

       row_ind = row_ind_tmp;
       col_ind = col_ind_tmp;
       val = val_tmp;
   } else {
       // Everyone else should get ready to receive the appropriate 
       // amount of data
       // Each process will be responsible for nnz_per_rank non-zero elements
       row_ind = (int*) malloc(sizeof(int) * nnz_per_rank);
       assert(row_ind);

       col_ind = (int*) malloc(sizeof(int) * nnz_per_rank);
       assert(col_ind);

       val = (double*) malloc(sizeof(double) * nnz_per_rank);
       assert(val);
   }

   start = MPI_Wtime();    
   // Scatter the data to each node
   MPI_Scatter(row_ind, nnz_per_rank, MPI_INT, row_ind, nnz_per_rank, MPI_INT,
               0, MPI_COMM_WORLD);
   MPI_Scatter(col_ind, nnz_per_rank, MPI_INT, col_ind, nnz_per_rank, MPI_INT,
               0, MPI_COMM_WORLD);
   MPI_Scatter(val, nnz_per_rank, MPI_DOUBLE, val, nnz_per_rank, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
   end = MPI_Wtime();
   timer[MAT_SCATTER_TIME] = end - start;


   // Now, calculate SpMV using COO
   // First set up some locks
   start = MPI_Wtime();
   omp_lock_t* writelock; 
   init_locks(&writelock, m);
   end = MPI_Wtime();
   timer[LOCK_INIT_TIME] = end - start;

   // set up result vector
   start = MPI_Wtime();
   double* res_coo = (double*) malloc(sizeof(double) * m);;
   assert(res_coo);

   fprintf(stdout, "Calculating COO SpMV ... ");
   // Calculate SPMV using COO
   spmv_coo(row_ind, col_ind, val, m, n, nnz_per_rank, vector_x, res_coo, 
            writelock);
   fprintf(stdout, "done\n");
   end = MPI_Wtime();
   timer[SPMV_COO_TIME] = end - start;
   // Make sure everyone's finished before doing any communication
   MPI_Barrier(MPI_COMM_WORLD);


   // Each rank has partial result - reduce to get the final result to rank 0
   double* res_coo_final = NULL;
   if(world_rank == 0) {
       res_coo_final = (double*) malloc(sizeof(double) * m);
       assert(res_coo_final);
       memset(res_coo_final, 0, sizeof(double) * m);
   }
   start = MPI_Wtime();
   MPI_Reduce(res_coo, res_coo_final, m, MPI_DOUBLE, MPI_SUM, 0, 
              MPI_COMM_WORLD);
   end = MPI_Wtime();
   timer[RES_REDUCE_TIME] = end - start;

   *res = res_coo_final;

   free(res_coo);
   if(world_rank == 0) {
       free(res_coo_final);
       free(vector_x);
       free(row_ind);
       free(col_ind);
       free(val);
   }
   destroy_locks(writelock, m);

}

void spmv_coo_2d(int world_rank, int world_size, int* row_ind, int* col_ind, 
                double* val, int m, int n, int nnz, double* vector_x,
                double** res, double timer[],int iter)
{

   double start;
   double end;

   // Rank 0 now determines how work will be distributed among the ranks
   int nnz_per_grid_element = (nnz + world_size - 1) / world_size;
       
   start = MPI_Wtime();
   // Also broadcast m and n
   MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
   // Lastly, send the input vector x
   if(world_rank != 0) {
       vector_x = (double*) malloc(sizeof(double) * n);
       assert(vector_x);
   }
   MPI_Bcast(vector_x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   end = MPI_Wtime();
   timer[VEC_BCAST_TIME] += end - start;


   // First, pad the data so that we can use MPI_Scatter instead of 
   // MPI_Scatterv
   if(world_rank == 0) {
       // new_nnz will be larger than nnz
       int new_nnz = nnz_per_grid_element * world_size;

       int* row_ind_tmp = (int*) malloc(sizeof(int) * new_nnz);
       assert(row_ind_tmp);
       memset(row_ind_tmp, 0, sizeof(int) * new_nnz);

       int* col_ind_tmp = (int*) malloc(sizeof(int) * new_nnz);
       assert(col_ind_tmp);
       memset(col_ind_tmp, 0, sizeof(int) * new_nnz);

       double* val_tmp = (double*) malloc(sizeof(double) * new_nnz);
       assert(val_tmp);
       memset(val_tmp, 0, sizeof(double) * new_nnz);

       memcpy(row_ind_tmp, row_ind, sizeof(int) * nnz);
       memcpy(col_ind_tmp, col_ind, sizeof(int) * nnz);
       memcpy(val_tmp, val, sizeof(double) * nnz);

    //   free(row_ind);
    //   free(col_ind);
    //   free(val);

       row_ind = row_ind_tmp;
       col_ind = col_ind_tmp;
       val = val_tmp;
   } else {
       // Everyone else should get ready to receive the appropriate 
       // amount of data
       // Each process will be responsible for nnz_per_grid_element non-zero elements
       row_ind = (int*) malloc(sizeof(int) * nnz_per_grid_element);
       assert(row_ind);

       col_ind = (int*) malloc(sizeof(int) * nnz_per_grid_element);
       assert(col_ind);

       val = (double*) malloc(sizeof(double) * nnz_per_grid_element);
       assert(val);
   }

   start = MPI_Wtime();    
   // Scatter the data to each node
   MPI_Scatter(row_ind, nnz_per_grid_element, MPI_INT, row_ind, nnz_per_grid_element, MPI_INT,
               0, MPI_COMM_WORLD);
   MPI_Scatter(col_ind, nnz_per_grid_element, MPI_INT, col_ind, nnz_per_grid_element, MPI_INT,
               0, MPI_COMM_WORLD);
   MPI_Scatter(val, nnz_per_grid_element, MPI_DOUBLE, val, nnz_per_grid_element, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
   end = MPI_Wtime();
   timer[MAT_SCATTER_TIME] += end - start;

   int q,p;
   q = sqrt(world_size);
   while(world_size % q != 0){
       q--;
   };
   p = world_size/q;
   //printf("%d %d \n",p,q);

   //Create the comm split grid
   int grid_row = world_rank / q;  // This is the row and grid indexs
   int grid_col = world_rank % q;  // For n=6, this should give
                                   //3 rows by 2 
   if ( iter == 0){
   char processor_name[MPI_MAX_PROCESSOR_NAME];
   int name_len;
   MPI_Get_processor_name(processor_name, &name_len);
   printf("MPI Task %d is running on node: %s\n", world_rank, processor_name);
   printf("World Rank:%d -> Row: %d Col: %d \n",world_rank, grid_row, grid_col);
   }
   MPI_Comm row_comm;
   MPI_Comm col_comm;
   MPI_Comm_split(MPI_COMM_WORLD, grid_row, world_rank, &row_comm);
   MPI_Comm_split(MPI_COMM_WORLD, grid_col, world_rank, &col_comm);

   // set up result vector
   double* res_grid_element = (double*) malloc(sizeof(double) * m);
   assert(res_grid_element);
   memset(res_grid_element, 0, sizeof(double) * m);

   double* res_coo_final = NULL;
   if (world_rank == 0){
       //fprintf(stdout, "Calculating 2D COO SpMV ... \n");
       res_coo_final = (double*) malloc(sizeof(double) * m);
       assert(res_coo_final);
       memset(res_coo_final, 0, sizeof(double) * m);
   }
   double* res_coo_row_final = NULL;
   if (grid_col == 0){
       res_coo_row_final = (double*) malloc(sizeof(double) * m);
       assert(res_coo_row_final);
       memset(res_coo_row_final, 0, sizeof(double) * m);
   }
   //So now every grid element has a result vector, row_ind, col_ind, vals
   //With the 0,0 element row/col/vals being nnz_per_grid_element*p*q long
   //while the others are nnz_per_grid_element long
   //every row has a result vector
   //And the grid has a final result vector

   MPI_Barrier(MPI_COMM_WORLD);

   if (world_rank == 0){
       start = MPI_Wtime();
   }
   // Calculate SPMV using COO over the grid elements
   for(int i = 0; i<nnz_per_grid_element;i++){
   res_grid_element[row_ind[i] - 1] += val[i] * vector_x[col_ind[i] -1];
   }

   if (world_rank == 0){
   end = MPI_Wtime();
   timer[SPMV_COO_TIME] += end - start;
   }
   
   //if (grid_col == 0){
   //      printf("Row: %d has completed spmv \n", grid_row);
   //}

   MPI_Barrier(MPI_COMM_WORLD);
   if (world_rank == 0){
       start = MPI_Wtime();
   }
   //Reduce into the row
   MPI_Reduce(res_grid_element, res_coo_row_final, m, MPI_DOUBLE, MPI_SUM, 0, row_comm);
   //if (grid_col == 0){
   //  printf("Row: %d has completed row reduction \n", grid_row);
   //  }
       
   if(grid_col == 0){ 
           //Only grid elements in the first column call this, 
           //and the grid comm breaks the grid into columns, so this 
           //mpi reduce only occurs amongst the first column.
       MPI_Reduce(res_coo_row_final, res_coo_final, m, MPI_DOUBLE, MPI_SUM, 0, col_comm);
   }

   if (world_rank == 0){
   end = MPI_Wtime();
   timer[RES_REDUCE_TIME] += end - start;

   *res = res_coo_final;
   }

   //Everyone frees this
   free(res_grid_element);
   if (world_rank != 0){
   free(vector_x);};
   free(row_ind);
   free(col_ind);
   free(val);

   if(grid_col == 0) {
       free(res_coo_row_final);
   }
}