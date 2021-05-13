!> @file
!! \brief This module contains some parameter used in SuperLU for
!! Fortran90 user.
!
!! <pre>
!! -- Distributed SuperLU routine (version 7.0) --
!! Lawrence Berkeley National Lab, Univ. of California Berkeley.
!! Last update: December 31, 2020
!! </pre>

module superlupara_mod

!----------------------------------------------------
! This module contains some parameter used in SUPERLU for Fortran90 user.
!----------------------------------------------------


implicit none
public superlu_ptr

!----------------------------------------------------
! kind of integer to hold a SuperLU pointer.  Use 64-bit integer.
! This might need to be changed on systems with large memory.
! If changed, be sure to change it in superlu_c2f_wrap.c too.
!
! integer, parameter :: superlu_ptr = kind(0) ! default integer size: 32-bit
integer, parameter :: superlu_ptr = 8 ! 64-bit

!----------------------------------------------------
! The following parameters are defined:

! These values come from superlu_enum_consts.h. If the values in there
! change, then they need to be changed here, too.

integer, parameter, public :: &
                      NO                      = 0, & ! yes_no_t
                      YES                     = 1, &
                      DOFACT                  = 0, & ! fact_t
                      SamePattern             = 1, &
                      SamePattern_SameRowPerm = 2, &
                      FACTORED                = 3, &
                      NOROWPERM               = 0, & ! rowperm_t
                      LargeDiag_MC64          = 1, &
                      LargeDiag_HWPM          = 2, &
                      MY_PERMR                = 3, &
                      NATURAL                 = 0, & ! colperm_t
                      MMD_ATA                 = 1, &
                      MMD_AT_PLUS_A           = 2, &
                      COLAMD                  = 3, &
                      METIS_AT_PLUS_A         = 4, &
                      PARMETIS                = 5, &
                      ZOLTAN                  = 6, &
                      MY_PERMC                = 7, &
                      NOTRANS                 = 0, & ! trans_t
                      TRANS                   = 1, &
                      CONJ                    = 2, &
                      NOEQUIL                 = 0, & ! DiagScale_t  Need?
                      ROW                     = 1, &
                      COL                     = 2, &
                      BOTH                    = 3, &
                      NOREFINE                = 0, & ! IterRefine_t
                      SLU_SINGLE              = 1, &
                      SLU_DOUBLE              = 2, &
                      SLU_EXTRA               = 3, &
                      USUB                    = 0, & ! MemType
                      LSUB                    = 1, &
                      UCOL                    = 2, &
                      LUSUP                   = 3, &
                      LLVL                    = 4, &
                      ULVL                    = 5, &
                      NO_MEMTYPE              = 6, &
                      SYSTEM                  = 0, & ! LU_space_t
                      USER                    = 1, & 
                      SILU                    = 0, & ! milu_t
                      SMILU_1                 = 1, &
                      SMILU_2                 = 2, &
                      SMILU_3                 = 3
                      
! These values come from supermatrix.h. If the values in there
! change, then they need to be changed here, too.

integer, parameter, public :: &
                      SLU_NC                  = 0, & ! Stype_t
                      SLU_NCP                 = 1, &
                      SLU_NR                  = 2, &
                      SLU_SC                  = 3, &
                      SLU_SCP                 = 4, &
                      SLU_SR                  = 5, &
                      SLU_DN                  = 6, &
                      SLU_NR_loc              = 7, &
                      SLU_S                   = 0, & ! Dtype_t
                      SLU_D                   = 1, &
                      SLU_C                   = 2, &
                      SLU_Z                   = 3, &
                      SLU_GE                  = 0, & ! Mtype_t
                      SLU_TRLU                = 1, &
                      SLU_TRUU                = 2, &
                      SLU_TRL                 = 3, &
                      SLU_TRU                 = 4, &
                      SLU_SYL                 = 5, &
                      SLU_SYU                 = 6, &
                      SLU_HEL                 = 7, &
                      SLU_HEU                 = 8

!----------------------------------------------------

end module superlupara_mod
