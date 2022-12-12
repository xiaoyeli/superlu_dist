#include "superlu_sdefs.h"

int
sread_binary(FILE *fp, int_t *m, int_t *n, int_t *nnz, 
	     float **nzval, int_t **rowind, int_t **colptr)
{
    int_t isize = sizeof(int_t), dsize = sizeof(float);
    int_t nnz_read;
    int_t i,j;
    fread(n, isize, 1, fp);
    fread(nnz, isize, 1, fp);
    printf("fread n " IFMT "\tnnz " IFMT "\n", *n, *nnz);
    *m = *n;
    *colptr = intMalloc_dist(*n+1);
    *rowind = intMalloc_dist(*nnz);
    *nzval  = floatMalloc_dist(*nnz);
    fread(*colptr, isize, (int_t) (*n + 1), fp);
    fread(*rowind, isize, (int_t) *nnz, fp);
    nnz_read = fread(*nzval, dsize, (int_t) (*nnz), fp);
    printf("# of floats fread: " IFMT "\n", nnz_read);
    return 0;
}

int
swrite_binary(int_t n, int_t nnz,
	      float *values, int_t *rowind, int_t *colptr)
{       
      FILE  *fp1;
      int nnz_written;
      size_t isize = sizeof(int_t), dsize = sizeof(float);
      fp1 = fopen("matrix.bin", "wb");
      fwrite(&n, isize, 1, fp1);
      fwrite(&nnz, isize, 1, fp1);
      fwrite(colptr, isize, n+1, fp1);
      fwrite(rowind, isize, nnz, fp1);
      nnz_written = fwrite(values, dsize, nnz, fp1);
      printf("n " IFMT ", # of float: " IFMT "\n", n, nnz);
      printf("dump binary file ... # of float fwrite: %d\n", nnz_written);
      assert(nnz_written==nnz);
      fclose(fp1);
      return 0;
}
