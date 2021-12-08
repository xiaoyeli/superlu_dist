/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * \brief Macro definitions
 *
 * <pre>
 * -- Distributed SuperLU routine (version 6.2) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * November 12, 2019
 *
 * Last update: February 19, 2020
 * </pre>
 */

#ifndef __SUPERLU_FCNAMES /* allow multiple inclusions */
#define __SUPERLU_FCNAMES

/* These are the functions defined in F90 wraper */
#define f_create_gridinfo_handle       FC_GLOBAL(f_create_gridinfo_handle,F_CREATE_GRIDINFO_HANDLE)
#define f_create_gridinfo3d_handle     FC_GLOBAL(f_create_gridinfo3d_handle,F_CREATE_GRIDINFO3D_HANDLE)
#define f_create_options_handle        FC_GLOBAL(f_create_options_handle,F_CREATE_OPTIONS_HANDLE)
#define f_create_SuperMatrix_handle    FC_GLOBAL(f_create_supermatrix_handle,F_CREATE_SUPERMATRIX_HANDLE)
#define f_destroy_gridinfo_handle      FC_GLOBAL(f_destroy_gridinfo_handle,F_DESTROY_GRIDINFO_HANDLE)
#define f_destroy_options_handle       FC_GLOBAL(f_destroy_options_handle,F_DESTROY_OPTIONS_HANDLE)
#define f_destroy_ScalePerm_handle     FC_GLOBAL(f_destroy_scaleperm_handle,F_DESTROY_SCALEPERM_HANDLE)
#define f_destroy_LUstruct_handle      FC_GLOBAL(f_destroy_lustruct_handle,F_DESTROY_LUSTRUCT_HANDLE)
#define f_destroy_SOLVEstruct_handle   FC_GLOBAL(f_destroy_solvestruct_handle,F_DESTROY_SOLVESTRUCT_HANDLE)
#define f_destroy_SuperMatrix_handle   FC_GLOBAL(f_destroy_supermatrix_handle,F_DESTROY_SUPERMATRIX_HANDLE)
#define f_create_SuperLUStat_handle    FC_GLOBAL(f_create_superlustat_handle,F_CREATE_SUPERLUSTAT_HANDLE)
#define f_destroy_SuperLUStat_handle   FC_GLOBAL(f_destroy_superlustat_handle,F_DESTROY_SUPERLUSTAT_HANDLE)
#define f_get_gridinfo                 FC_GLOBAL(f_get_gridinfo,F_GET_GRIDINFO)
#define f_get_gridinfo3d               FC_GLOBAL(f_get_gridinfo3d,F_GET_GRIDINFO3D)
#define f_get_SuperMatrix              FC_GLOBAL(f_get_supermatrix,F_GET_SUPERMATRIX)
#define f_set_SuperMatrix              FC_GLOBAL(f_set_supermatrix,F_SET_SUPERMATRIX)
#define f_get_CompRowLoc_Matrix        FC_GLOBAL(f_get_comprowloc_matrix,F_GET_COMPROWLOC_MATRIX)
#define f_set_CompRowLoc_Matrix        FC_GLOBAL(f_set_comprowloc_matrix,F_SET_COMPROWLOC_MATRIX)
#define f_get_superlu_options          FC_GLOBAL(f_get_superlu_options,F_GET_SUPERLU_OPTIONS)
#define f_set_superlu_options          FC_GLOBAL(f_set_superlu_options,F_SET_SUPERLU_OPTIONS)
#define f_set_default_options          FC_GLOBAL(f_set_default_options,F_SET_DEFAULT_OPTIONS)
#define f_superlu_gridinit             FC_GLOBAL(f_superlu_gridinit,F_SUPERLU_GRIDINIT)
#define f_superlu_gridinit3d           FC_GLOBAL(f_superlu_gridinit3d,F_SUPERLU_GRIDINIT3D)
#define f_superlu_gridmap              FC_GLOBAL(f_superlu_gridmap,F_SUPERLU_GRIDMAP)
#define f_superlu_gridexit             FC_GLOBAL(f_superlu_gridexit,F_SUPERLU_GRIDEXIT)
#define f_PStatInit                    FC_GLOBAL(f_pstatinit,F_PSTATINIT)
#define f_PStatFree                    FC_GLOBAL(f_pstatfree,F_PSTATFREE)
#define f_Destroy_CompRowLoc_Mat_dist  FC_GLOBAL(f_destroy_comprowloc_mat_dist,F_DESTROY_COMPROWLOC_MAT_DIST)
#define f_Destroy_SuperMat_Store_dist  FC_GLOBAL(f_destroy_supermat_store_dist,F_DESTROY_SUPERMAT_STORE_DIST)
#define f_check_malloc                 FC_GLOBAL(f_check_malloc,F_CHECK_MALLOC)

////// double
#define f_dcreate_ScalePerm_handle     FC_GLOBAL(f_dcreate_scaleperm_handle,F_DCREATE_SCALEPERM_HANDLE)
#define f_dcreate_LUstruct_handle      FC_GLOBAL(f_dcreate_lustruct_handle,F_DCREATE_LUSTRUCT_HANDLE)
#define f_dcreate_SOLVEstruct_handle   FC_GLOBAL(f_dcreate_solvestruct_handle,F_DCREATE_SOLVESTRUCT_HANDLE)
#define f_dScalePermstructInit         FC_GLOBAL(f_dscalepermstructinit,F_DSCALEPERMSTRUCTINIT)
#define f_dScalePermstructFree         FC_GLOBAL(f_dscalepermstructfree,F_DSCALEPERMSTRUCTFREE)
#define f_dLUstructInit                FC_GLOBAL(f_dlustructinit,F_DLUSTRUCTINIT)
#define f_dLUstructFree                FC_GLOBAL(f_dlustructfree,F_DLUSTRUCTFREE)
#define f_dDestroy_LU_SOLVE_struct     FC_GLOBAL(f_ddestroy_lu_solve_struct,F_DDESTROY_LU_SOLVE_STRUCT)
#define f_dDestroy_LU_SOLVE_struct_3d  FC_GLOBAL(f_ddestroy_lu_solve_struct_3d,F_DDESTROY_LU_SOLVE_STRUCT_3D)
#define f_dDestroy_A3d_gathered_on_2d  FC_GLOBAL(f_ddestroy_a3d_gathered_on_2d,F_DDESTROY_A3D_GATHERED_ON_2D)

#define f_dCreate_CompRowLoc_Mat_dist  FC_GLOBAL(f_dcreate_comprowloc_mat_dist,F_DCREATE_COMPROWLOC_MAT_DIST)
#define f_dSolveFinalize               FC_GLOBAL(f_dsolvefinalize,F_DSOLVEFINALIZE)
#define f_pdgssvx                      FC_GLOBAL(f_pdgssvx,F_PDGSSVX)
#define f_pdgssvx3d                    FC_GLOBAL(f_pdgssvx3d,F_PDGSSVX3D)
#define f_dcreate_dist_matrix          FC_GLOBAL(f_dcreate_dist_matrix,F_DCREATE_DIST_MATRIX)
#define f_dcreate_matrix_x_b           FC_GLOBAL(f_dcreate_matrix_x_b,F_DCREATE_MATRIX_X_B)
#define f_dcreate_matrix_x_b_3d        FC_GLOBAL(f_dcreate_matrix_x_b_3d,F_DCREATE_MATRIX_X_B_3D)

////// complex16
#define f_zcreate_ScalePerm_handle     FC_GLOBAL(f_zcreate_scaleperm_handle,F_ZCREATE_SCALEPERM_HANDLE)
#define f_zcreate_LUstruct_handle      FC_GLOBAL(f_zcreate_lustruct_handle,F_ZCREATE_LUSTRUCT_HANDLE)
#define f_zcreate_SOLVEstruct_handle   FC_GLOBAL(f_zcreate_solvestruct_handle,F_ZCREATE_SOLVESTRUCT_HANDLE)
#define f_zScalePermstructInit         FC_GLOBAL(f_zscalepermstructinit,F_ZSCALEPERMSTRUCTINIT)
#define f_zScalePermstructFree         FC_GLOBAL(f_zscalepermstructfree,F_ZSCALEPERMSTRUCTFREE)
#define f_zLUstructInit                FC_GLOBAL(f_zlustructinit,F_ZLUSTRUCTINIT)
#define f_zLUstructFree                FC_GLOBAL(f_zlustructfree,F_ZLUSTRUCTFREE)
#define f_zDestroy_LU_SOLVE_struct     FC_GLOBAL(f_zdestroy_lu_solve_struct,F_ZDESTROY_LU_SOLVE_STRUCT)
#define f_zDestroy_LU_SOLVE_struct_3d  FC_GLOBAL(f_zdestroy_lu_solve_struct_3d,F_ZDESTROY_LU_SOLVE_STRUCT_3D)
#define f_zDestroy_A3d_gathered_on_2d  FC_GLOBAL(f_zdestroy_a3d_gathered_on_2d,F_ZDESTROY_A3D_GATHERED_ON_2D)

#define f_zCreate_CompRowLoc_Mat_dist  FC_GLOBAL(f_zcreate_comprowloc_mat_dist,F_ZCREATE_COMPROWLOC_MAT_DIST)
#define f_zSolveFinalize               FC_GLOBAL(f_zsolvefinalize,F_ZSOLVEFINALIZE)
#define f_pzgssvx                      FC_GLOBAL(f_pzgssvx,F_PZGSSVX)
#define f_pzgssvx3d                    FC_GLOBAL(f_pzgssvx3d,F_PZGSSVX3D)
#define f_zcreate_matrix_x_b           FC_GLOBAL(f_zcreate_matrix_x_b,F_ZCREATE_MATRIX_X_B)
#define f_zcreate_matrix_x_b_3d        FC_GLOBAL(f_zcreate_matrix_x_b_3d,F_ZCREATE_MATRIX_X_B_3D)

/* BLAS */
#define sasum_    FC_GLOBAL(sasum,SASUM)
#define isamax_   FC_GLOBAL(isamax,ISAMAX)
#define scopy_    FC_GLOBAL(scopy,SCOPY)
#define sscal_    FC_GLOBAL(sscal,SSCAL)
#define sger_     FC_GLOBAL(sger,SGER)
#define snrm2_    FC_GLOBAL(snrm2,SNRM2)
#define ssymv_    FC_GLOBAL(ssymv,SSYMV)
#define sdot_     FC_GLOBAL(sdot,SDOT)
#define saxpy_    FC_GLOBAL(saxpy,SAXPY)
#define ssyr2_    FC_GLOBAL(ssyr2,SSYR2)
#define srot_     FC_GLOBAL(srot,SROT)
#define sgemv_    FC_GLOBAL(sgemv,SGEMV)
#define strsv_    FC_GLOBAL(strsv,STRSV)
#define sgemm_    FC_GLOBAL(sgemm,SGEMM)
#define strsm_    FC_GLOBAL(strsm,STRSM)

#define dasum_    FC_GLOBAL(dasum,DASUM)
#define idamax_   FC_GLOBAL(damax,DAMAX)
#define dcopy_    FC_GLOBAL(dcopy,DCOPY)
#define dscal_    FC_GLOBAL(dscal,DSCAL)
#define dger_     FC_GLOBAL(dger,DGER)
#define dnrm2_    FC_GLOBAL(dnrm2,DNRM2)
#define dsymv_    FC_GLOBAL(dsymv,DSYMV)
#define ddot_     FC_GLOBAL(ddot,DDOT)
#define daxpy_    FC_GLOBAL(daxpy,DAXPY)
#define dsyr2_    FC_GLOBAL(dsyr2,DSYR2)
#define drot_     FC_GLOBAL(drot,DROT)
#define dgemv_    FC_GLOBAL(dgemv,DGEMV)
#define dtrsv_    FC_GLOBAL(dtrsv,DTRSV)
#define dgemm_    FC_GLOBAL(dgemm,DGEMM)
#define dtrsm_    FC_GLOBAL(dtrsm,DTRSM)

#define scasum_   FC_GLOBAL(scasum,SCASUM)
#define icamax_   FC_GLOBAL(icamax,ICAMAX)
#define ccopy_    FC_GLOBAL(ccopy,CCOPY)
#define cscal_    FC_GLOBAL(cscal,CSCAL)
#define scnrm2_   FC_GLOBAL(scnrm2,SCNRM2)
#define caxpy_    FC_GLOBAL(caxpy,CAXPY)
#define cgemv_    FC_GLOBAL(cgemv,CGEMV)
#define ctrsv_    FC_GLOBAL(ctrsv,CTRSV)
#define cgemm_    FC_GLOBAL(cgemm,CGEMM)
#define ctrsm_    FC_GLOBAL(ctrsm,CTRSM)
#define cgerc_    FC_GLOBAL(cgerc,CGERC)
#define chemv_    FC_GLOBAL(chemv,CHEMV)
#define cher2_    FC_GLOBAL(cher2,CHER2)

#define dzasum_   FC_GLOBAL(dzasum,DZASUM)
#define izamax_   FC_GLOBAL(izamax,IZAMAX)
#define zcopy_    FC_GLOBAL(zcopy,ZCOPY)
#define zscal_    FC_GLOBAL(zscal,ZSCAL)
#define dznrm2_   FC_GLOBAL(dznrm2,DZNRM2)
#define zaxpy_    FC_GLOBAL(zaxpy,ZAXPY)
#define zgemv_    FC_GLOBAL(zgemv,ZGEMV)
#define ztrsv_    FC_GLOBAL(ztrsv,ZTRSV)
#define zgemm_    FC_GLOBAL(zgemm,ZGEMM)
#define ztrsm_    FC_GLOBAL(ztrsm,ZTRSM)
#define zgerc_    FC_GLOBAL(zgerc,ZGERC)
#define zhemv_    FC_GLOBAL(zhemv,ZHEMV)
#define zher2_    FC_GLOBAL(zher2,ZHER2)
#define zgeru_    FC_GLOBAL(zgeru,ZGERU)

/* LAPACK */
#define strtri_   FC_GLOBAL(strtri,STRTRI)
#define dtrtri_   FC_GLOBAL(dtrtri,DTRTRI)
#define ctrtri_   FC_GLOBAL(ctrtri,CTRTRI)
#define ztrtri_   FC_GLOBAL(ztrtri,ZTRTRI)

/*
#define mc64id_dist         mc64id_dist
#define mc64ad_dist         mc64ad_dist
*/

#define c_bridge_dgssv_               FC_GLOBAL(c_bridge_dgssv,C_BRIDGE_DGSSV)
#define c_fortran_slugrid_            FC_GLOBAL(c_fortran_slugrid,C_FORTRAN_SLUGRID)
#define c_fortran_pdgssvx_            FC_GLOBAL(c_fortran_pdgssvx,C_FORTRAN_PDGSSVX)
#define c_fortran_pdgssvx_ABglobal_   FC_GLOBAL(c_fortran_pdgssvx_ABglobal,C_FORTRAN_PDGSSVX_ABGLOBAL)
#define c_fortran_pzgssvx_            FC_GLOBAL(c_fortran_pzgssvx,C_FORTRAN_PZGSSVX)
#define c_fortran_pzgssvx_ABglobal_   FC_GLOBAL(c_fortran_pzgssvx_ABglobal,C_FORTRAN_PZGSSVX_ABGLOBAL)


#endif /* __SUPERLU_FCNAMES */
