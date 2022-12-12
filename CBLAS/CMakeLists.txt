set(headers 
    f2c.h
)
set(sources input_error_dist.c)

if (enable_single)
    list(APPEND sources
      isamax.c
      sasum.c
      saxpy.c
      scopy.c
      sdot.c
      snrm2.c
      srot.c
      sscal.c
      sgemv.c
      ssymv.c
      strsv.c
      sger.c
      ssyr2.c
      sgemm.c
      strsm.c
    )
endif()

if (enable_double)
    list(APPEND sources
      idamax.c
      dasum.c
      daxpy.c
      dcopy.c
      ddot.c
      dnrm2.c
      drot.c
      dscal.c
      dgemv.c
      dsymv.c
      dtrsv.c
      dger.c
      dsyr2.c
      dgemm.c
      dtrsm.c
    )
endif()

if (enable_complex16)
    list(APPEND sources
      izamax.c
      dzasum.c
      zaxpy.c
      zcopy.c
      dznrm2.c
      zscal.c
      dcabs1.c
      z_internal.c
      zgemv.c
      zhemv.c
      ztrsv.c
      zgerc.c
      zgeru.c
      zher2.c
      zgemm.c
      ztrsm.c
    )
endif()

add_library(blas ${sources} ${HEADERS})

install(TARGETS blas DESTINATION "${INSTALL_LIB_DIR}")
install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
