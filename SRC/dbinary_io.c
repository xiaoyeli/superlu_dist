#include "superlu_ddefs.h"

int
dread_binary(FILE *fp, int_t *m, int_t *n, int_t *nnz, 
	     double **nzval, int_t **rowind, int_t **colptr)
{
    size_t isize = sizeof(int_t), dsize = sizeof(double);
    int nnz_read;
    fread(n, isize, 1, fp);
    fread(nnz, isize, 1, fp);
    printf("fread n " IFMT "\tnnz " IFMT "\n", *n, *nnz);
    *m = *n;
    *colptr = intMalloc_dist(*n+1);
    *rowind = intMalloc_dist(*nnz);
    *nzval  = doubleMalloc_dist(*nnz);
    fread(*colptr, isize, (size_t) (*n + 1), fp);
    fread(*rowind, isize, (size_t) *nnz, fp);
    nnz_read = fread(*nzval, dsize, (size_t) (*nnz), fp);
    printf("# of doubles fread: %d\n", nnz_read);
    return 0;
}

int
dwrite_binary(int_t n, int_t nnz,
	      double *values, int_t *rowind, int_t *colptr)
{       
      FILE  *fp1;
      int nnz_written;
      size_t isize = sizeof(int_t), dsize = sizeof(double);
      fp1 = fopen("matrix.bin", "wb");
      fwrite(&n, isize, 1, fp1);
      fwrite(&nnz, isize, 1, fp1);
      fwrite(colptr, isize, n+1, fp1);
      fwrite(rowind, isize, nnz, fp1);
      nnz_written = fwrite(values, dsize, nnz, fp1);
      printf("n " IFMT ", # of double: " IFMT "\n", n, nnz);
      printf("dump binary file ... # of double fwrite: %d\n", nnz_written);
      assert(nnz_written==nnz);
      fclose(fp1);
      return 0;
}
