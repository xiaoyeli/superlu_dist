!
! -- Distributed SuperLU routine (version 2.0) --
! Lawrence Berkeley National Lab, Univ. of California Berkeley.
! July 29, 2003
!
!
      program f_pddrive
      use superlu_mod
      include 'mpif.h'
      implicit none
      integer maxn, maxnz, maxnrhs
      parameter ( maxn = 10000, maxnz = 100000, maxnrhs = 10 )
      integer rowind(maxnz), colptr(maxn)
      real*8  values(maxnz), b(maxn), berr(maxnrhs)
      integer n, m, nnz, nrhs, ldb, i, ierr, info, iam, m_loc, nnz_loc, fst_row
      integer nprow, npcol
      integer init

      integer(superlu_ptr) :: grid
      integer(superlu_ptr) :: options
      integer(superlu_ptr) :: ScalePermstruct
      integer(superlu_ptr) :: LUstruct
      integer(superlu_ptr) :: SOLVEstruct
      integer(superlu_ptr) :: A

      integer(superlu_ptr) :: stat


! Default process rows
      nprow = 1  
! Default process columns
      npcol = 1 
! Number of right-hand side 
      nrhs = 1  

! INITIALIZE MPI ENVIRONMENT 
      call mpi_init(ierr)

! Check Malloc
      call f_check_malloc(iam)

! create C structures used in superlu
      call f_create_gridinfo(grid)
      call f_create_options(options)
      call f_create_ScalePermstruct(ScalePermstruct)
      call f_create_LUstruct(LUstruct)
      call f_create_SOLVEstruct(SOLVEstruct)
      call f_create_SuperMatrix(A)

! initialize the SuperLU process grid
      nprow = 2
      npcol = 2
      call f_superlu_gridinit(MPI_COMM_WORLD, nprow, npcol, grid)

! Bail out if I do not belong in the grid. 
      call get_GridInfo(grid, iam=iam)
      if ( iam >= nprow * npcol ) then 
         go to 100
      endif
      if ( iam == 0 ) then 
         write(*,*) ' Process grid ', nprow, ' X ', npcol
      endif


! Read Harwell-Boeing matrix
      if ( iam == 0 ) then 
         call hbcode1(m, n, nnz, values, rowind, colptr)
      endif


! Distribute the matrix to the gird
      call  f_dcreate_matrix_dis(A, m, n, nnz, values, rowind, colptr, grid)

! Get m_loc
      call  get_CompRowLoc_Matrix(A, nrow_loc=m_loc);

! Setup the right hand side
      nrhs = 1
      ldb = m_loc
      do i = 1, ldb
         b(i) = 1.0
      enddo


! set the default input options
      call f_set_default_options(options)

! set one or more option
!      call set_superlu_options(options,Fact=FACTORED)


! initialize ScalePermstruct and LUstruct

! get the m and n 
      call get_SuperMatrix(A,nrow=m,ncol=n)
      call f_ScalePermstructInit(m, n, ScalePermstruct)
      call f_LUstructInit(m, n, LUstruct)

! initialize the statistics variables
      call f_create_SuperLUStat(stat)
      call f_PStatInit(stat)


! call the linear equation solver
      call f_pdgssvx(options, A, &
               ScalePermstruct, b, &
               ldb, nrhs, grid, &
               LUstruct, SOLVEstruct, berr, &
               stat, info)

      if (info == 0) then
         write (*,*) 'Backward error: ', (berr(i), i = 1, nrhs)
      else
         write(*,*) 'INFO from f_pdgssvx = ', info
      endif


! free memory
      call f_PStatFree(stat)
      call f_destroy_SuperLUStat(stat)

! deallocate SuperLU allocated storage

      call f_Destroy_CompRowLoc_Matrix_dis(A)
      call f_ScalePermstructFree(ScalePermstruct)
!      call get_SuperMatrix(A,ncol=n)
      call f_Destroy_LU(n, grid, LUstruct)
      call f_LUstructFree(LUstruct)
      call get_superlu_options(options, SolveInitialized=init)
      if (init == YES) then
         call f_dSolveFinalize(options, SOLVEstruct)
      endif


! release the SuperLU process grid
100   call f_superlu_gridexit(grid)

! destroy C structures in superlu_matrix_type
      call f_destroy_gridinfo(grid)
      call f_destroy_options(options)
      call f_destroy_ScalePermstruct(ScalePermstruct)
      call f_destroy_LUstruct(LUstruct)
      call f_destroy_SOLVEstruct(SOLVEstruct)
      call f_destroy_SuperMatrix(A)

! TERMINATES THE MPI EXECUTION ENVIRONMENT
      call mpi_finalize(ierr)
!

! Check Malloc
      call f_check_malloc(iam)


      stop
      end




