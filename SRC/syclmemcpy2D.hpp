#ifndef __SYCLMEMCPY2D /* allow multiple inclusions */
#define __SYCLMEMCPY2D

#pragma once
#include <CL/sycl.hpp>

/// Asynchronously copies 2D matrix specified by \p x and \p y from the address
/// specified by \p from_ptr to the address specified by \p to_ptr, while \p
/// \p from_pitch and \p to_pitch are the range of dim x in bytes of the matrix
/// specified by \p from_ptr and \p to_ptr. The value of \p direction is used to
/// set the copy direction, it can be \a host_to_host, \a host_to_device, \a
/// device_to_host, \a device_to_device or \a automatic. The return of the
/// function does NOT guarantee the copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param to_pitch Range of dim x in bytes of destination matrix.
/// \param from_ptr Pointer to source memory address.
/// \param from_pitch Range of dim x in bytes of source matrix.
/// \param x Range of dim x of matrix to be copied.
/// \param y Range of dim y of matrix to be copied.
/// \param direction Direction of the copy.
/// \param q Queue to execute the copy task.
/// \returns no return value.

extern inline void
sycl_memcpy2D_async(cl::sycl::queue &q,
		    void *to_ptr, size_t to_pitch,
		    const void *from_ptr, size_t from_pitch,
		    size_t x, size_t y) {

    sycl::range<3> to_range   = sycl::range<3>(to_pitch, y, 1);
    sycl::range<3> from_range = sycl::range<3>(from_pitch, y, 1);
    sycl::range<3> size       = sycl::range<3>(x, y, 1);

    size_t to_slice = to_range.get(1) * to_range.get(0);
    size_t from_slice = from_range.get(1) * from_range.get(0);
    unsigned char *to_surface = (unsigned char *)to_ptr;
    const unsigned char *from_surface =	(const unsigned char *)from_ptr;

    for (size_t z = 0; z < size.get(2); ++z) {
	unsigned char *to_ptr = to_surface;
	const unsigned char *from_ptr = from_surface;
	for (size_t y = 0; y < size.get(1); ++y) {
	    q.memcpy(to_ptr, from_ptr, size.get(0)).wait();
	    to_ptr += to_range.get(0);
	    from_ptr += from_range.get(0);
	}
	to_surface += to_slice;
	from_surface += from_slice;
    }
}
// /// memcpy 2D matrix with pitch.
// 		static inline cl::sycl::vector_class<cl::sycl::event>
// 		    dpct_memcpy(cl::sycl::queue &q, void *to_ptr, const void *from_ptr,
// 				size_t to_pitch, size_t from_pitch, size_t x, size_t y,
// 				memcpy_direction direction = automatic) {
// 		    return dpct_memcpy(q, to_ptr, from_ptr, cl::sycl::range<3>(to_pitch, y, 1),
// 				       cl::sycl::range<3>(from_pitch, y, 1),
// 				       cl::sycl::id<3>(0, 0, 0), cl::sycl::id<3>(0, 0, 0),
// 				       cl::sycl::range<3>(x, y, 1), direction);
// 		}
// /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
// /// and \p from_range to another specified by \p to_ptr and \p to_range.
// 		static inline cl::sycl::vector_class<cl::sycl::event>
// 		    dpct_memcpy(cl::sycl::queue &q, void *to_ptr, const void *from_ptr,
// 				cl::sycl::range<3> to_range, cl::sycl::range<3> from_range,
// 				cl::sycl::range<3> size, memcpy_direction direction) {

// 		    size_t to_slice = to_range.get(1) * to_range.get(0),
// 			from_slice = from_range.get(1) * from_range.get(0);
// 		    unsigned char *to_surface = (unsigned char *)to_ptr;
// 		    const unsigned char *from_surface = (const unsigned char *)from_ptr;

// 		    for (size_t z = 0; z < size.get(2); ++z) {
// 			unsigned char *to_ptr = to_surface;
// 			const unsigned char *from_ptr = from_surface;
// 			for (size_t y = 0; y < size.get(1); ++y) {
// 			    dpct_memcpy(q, to_ptr, from_ptr, size.get(0), direction);
// 			    to_ptr += to_range.get(0);
// 			    from_ptr += from_range.get(0);
// 			}
// 			to_surface += to_slice;
// 			from_surface += from_slice;
// 		    }
// 		}

#endif // SYCLMEMCPY2D
