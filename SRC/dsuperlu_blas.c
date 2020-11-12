

#ifdef _CRAY
_fcd ftcs = _cptofcd("N", strlen("N"));
_fcd ftcs1 = _cptofcd("L", strlen("L"));
_fcd ftcs2 = _cptofcd("N", strlen("N"));
_fcd ftcs3 = _cptofcd("U", strlen("U"));
#endif

int superlu_dgemm(const char *transa, const char *transb,
                  int m, int n, int k, double alpha, double *a, 
                  int lda, double *b, int ldb, double beta, double *c, int ldc)
{
#ifdef _CRAY
    _fcd ftcs = _cptofcd(transa, strlen(transa));
    _fcd ftcs1 = _cptofcd(transb, strlen(transb));
    return SGEMM(ftcs, ftcs1, &m, &n, &k,
                 &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#elif defined(USE_VENDOR_BLAS)
    return dgemm_(transa, transb, &m, &n, &k,
                  &alpha, a, &lda, b, &ldb, &beta, c, &ldc, 1, 1);
#else
    return dgemm_(transa, transb, &m, &n, &k,
                  &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#endif
}


// dtrsm_ ("R", "U", "N", "N", &len, &nsupc, &alpha,
//   179  			ublk_ptr, &ld_ujrow, &lusup[off], &nsupr,

//    dtrsv_ ("L", "N", "U", &segsize, &lusup[luptr], &nsupr