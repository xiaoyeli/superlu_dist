!
! -- Distributed SuperLU routine (version 2.0) --
! Lawrence Berkeley National Lab, Univ. of California Berkeley.
! July 10, 2003
!
!
      program f_pddrive_ABglobal
      include 'mpif.h'
      implicit none
      integer maxn, maxnz, maxnrhs
      parameter ( maxn = 10000, maxnz = 100000, maxnrhs = 10 )
      integer rowind(maxnz), colptr(maxn)
      real*8  values(maxnz), b(maxn), berr(maxnrhs)
      integer n, nnz, nrhs, ldb, i, ierr, info, iopt
      integer nprow, npcol
      integer factors_handle(8), grid_handle(8)
!
      call mpi_init(ierr)
!
!     Read Harwell-Boeing matrix
      call hbcode1(n, n, nnz, values, rowind, colptr)
!
!     Adjust to 0-based indexing which is required by the C routines.
      do i = 1, n+1
         colptr(i) = colptr(i) - 1;
      end do
      do i = 1, nnz
         rowind(i) = rowind(i) - 1;
      end do

      nrhs = 1
      ldb = n
      do i = 1, n
         b(i) = 1.0
      enddo
!
      iopt = 1
      nprow = 2
      npcol = 2
      call c_fortran_slugrid(iopt, MPI_COMM_WORLD, nprow, npcol,
     $     grid_handle)
!
! Only performs LU factorization
!
      iopt = 1
      call c_fortran_pdgssvx_ABglobal(iopt, n, nnz, nrhs,
     $     values, rowind, colptr, b, ldb, grid_handle, berr,
     $     factors_handle, info)
!
! Now performs triangular solve with the existing factors
!
      iopt = 3
      call c_fortran_pdgssvx_ABglobal(iopt, n, nnz, nrhs,
     $     values, rowind, colptr, b, ldb, grid_handle, berr,
     $     factors_handle, info)
!
      if (info .eq. 0) then
         write (*,*) 'Backward error: ', (berr(i), i = 1, nrhs)
      else
         write(*,*) 'INFO from c_fortran_pdgssvx_ABglobal = ', info
      endif
!
! Now free the storage associated with the handles
!
      iopt = 4
      call c_fortran_pdgssvx_ABglobal(iopt, n, nnz, nrhs,
     $     values, rowind, colptr, b, ldb, grid_handle, berr,
     $     factors_handle, info)
      iopt = 2
      call c_fortran_slugrid(iopt, MPI_COMM_WORLD, nprow, npcol,
     $     grid_handle)
!
      call mpi_finalize(ierr)
!
      stop
      end
