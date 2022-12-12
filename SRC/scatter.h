#ifndef _SCATTER_H_
#define _SCATTER_H_

#ifdef CLEAN_SCATTER
#define SCATTER_L_CPU  scatter_l
#define SCATTER_U_CPU  scatter_u
#else
#define SCATTER_L_CPU  scatter_l
#define SCATTER_U_CPU  scatter_u

#endif

void
scatter_l (int_t ib,
           int_t ljb,
           int_t nsupc,
           int_t iukp,
           int_t *xsup,
           int_t klst,
           int_t nbrow,
           int_t lptr,
           int_t temp_nbrow,
           int_t *usub,
           int_t *lsub,
           double *tempv,
           int_t *indirect_thread, int_t *indirect2,
           int_t **Lrowind_bc_ptr, double **Lnzval_bc_ptr, gridinfo_t *grid);

void
scatter_u (int_t ib,
           int_t jb,
           int_t nsupc,
           int_t iukp,
           int_t *xsup,
           int_t klst,
           int_t nbrow,
           int_t lptr,
           int_t temp_nbrow,
           int_t *lsub,
           int_t *usub,
           double *tempv,
           int_t *indirect,
           int_t **Ufstnz_br_ptr, double **Unzval_br_ptr, gridinfo_t *grid);

void
arrive_at_ublock (int_t j,      //block number
                  int_t *iukp,  // output
                  int_t *rukp, int_t *jb,   /* Global block number of block U(k,j). */
                  int_t *ljb,   /* Local block number of U(k,j). */
                  int_t *nsupc,     /*supernode size of destination block */
                  int_t iukp0,  //input
                  int_t rukp0, int_t *usub,     /*usub scripts */
                  int_t *perm_u,    /*permutation matrix */
                  int_t *xsup,  /*for SuperSize and LBj */
                  gridinfo_t *grid);


void
block_gemm_scatter( int_t lb, int_t j,
                    Ublock_info_t *Ublock_info,
                    Remain_info_t *Remain_info,
                    double *L_mat, int_t ldl,
                    double *U_mat, int_t ldu,
                    double *bigV,
                    // int_t jj0,
                    int_t knsupc,  int_t klst,
                    int_t *lsub, int_t *usub, int_t ldt,
                    int_t thread_id,
                    int_t *indirect,
                    int_t *indirect2,
                    int_t **Lrowind_bc_ptr, double **Lnzval_bc_ptr,
                    int_t **Ufstnz_br_ptr, double **Unzval_br_ptr,
                    int_t *xsup, gridinfo_t *grid,
                    SuperLUStat_t *stat
#ifdef SCATTER_PROFILE
                    , double *Host_TheadScatterMOP, double *Host_TheadScatterTimer
#endif
                  );


/*this version uses a lock to prevent multiple thread updating the same block*/
void
block_gemm_scatter_lock( int_t lb, int_t j,
                         omp_lock_t* lock,
                         Ublock_info_t *Ublock_info,
                         Remain_info_t *Remain_info,
                         double *L_mat, int_t ldl,
                         double *U_mat, int_t ldu,
                         double *bigV,
                         // int_t jj0,
                         int_t knsupc,  int_t klst,
                         int_t *lsub, int_t *usub, int_t ldt,
                         int_t thread_id,
                         int_t *indirect,
                         int_t *indirect2,
                         int_t **Lrowind_bc_ptr, double **Lnzval_bc_ptr,
                         int_t **Ufstnz_br_ptr, double **Unzval_br_ptr,
                         int_t *xsup, gridinfo_t *grid
#ifdef SCATTER_PROFILE
                         , double *Host_TheadScatterMOP, double *Host_TheadScatterTimer
#endif
                       );

int_t block_gemm_scatterTopLeft( int_t lb,  int_t j,
                                 double* bigV, int_t knsupc,  int_t klst, int_t* lsub,
                                 int_t * usub, int_t ldt,  int_t* indirect, int_t* indirect2,
                                 HyP_t* HyP,
                                 LUstruct_t *LUstruct,
                                 gridinfo_t* grid,
                                 SCT_t*SCT, SuperLUStat_t *stat
                               );
int_t block_gemm_scatterTopRight( int_t lb,  int_t j,
                                  double* bigV, int_t knsupc,  int_t klst, int_t* lsub,
                                  int_t * usub, int_t ldt,  int_t* indirect, int_t* indirect2,
                                  HyP_t* HyP,
                                  LUstruct_t *LUstruct,
                                  gridinfo_t* grid,
                                  SCT_t*SCT, SuperLUStat_t *stat
                                );
int_t block_gemm_scatterBottomLeft( int_t lb,  int_t j,
                                    double* bigV, int_t knsupc,  int_t klst, int_t* lsub,
                                    int_t * usub, int_t ldt,  int_t* indirect, int_t* indirect2,
                                    HyP_t* HyP,
                                    LUstruct_t *LUstruct,
                                    gridinfo_t* grid,
                                    SCT_t*SCT, SuperLUStat_t *stat
                                  );
int_t block_gemm_scatterBottomRight( int_t lb,  int_t j,
                                     double* bigV, int_t knsupc,  int_t klst, int_t* lsub,
                                     int_t * usub, int_t ldt,  int_t* indirect, int_t* indirect2,
                                     HyP_t* HyP,
                                     LUstruct_t *LUstruct,
                                     gridinfo_t* grid,
                                     SCT_t*SCT, SuperLUStat_t *stat
                                   );

void gather_u(int_t num_u_blks,
              Ublock_info_t *Ublock_info, int_t * usub,
              double *uval,  double *bigU,  int_t ldu,
              int_t *xsup, int_t klst                /* for SuperSize */
             );

void gather_l( int_t num_LBlk, int_t knsupc,
               Remain_info_t *L_info,
               double * lval, int_t LD_lval,
               double * L_buff );
#endif
