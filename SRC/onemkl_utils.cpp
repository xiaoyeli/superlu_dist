/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
#include "superlu_defs.h"

#ifdef HAVE_SYCL  // enable SYCL

#include <iostream>
#include "onemkl_utils.hpp"

void DisplayHeader()
{
    const int kb = 1024;
    const int mb = kb * kb;
    // cout << "NBody.GPU" << endl << "=========" << endl << endl;

    int devCount=0;

    sycl::platform platform(sycl::default_selector{});
    auto const& gpu_devices = platform.get_devices();
    for (int i = 0; i < gpu_devices.size(); i++) {
      if (gpu_devices[i].is_gpu()) {
	// if(gpu_devices[i].get_info<sycl::info::device::partition_max_sub_devices>() > 0) {
	//   auto subDevicesDomainNuma = gpu_devices[i].create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);
	//   devCount += subDevicesDomainNuma.size();
	// }
	// else {
	  devCount++;
	  //}
      }
    }
    std::cout << "SYCL version:    v" << gpu_devices[0].get_info<sycl::info::device::version>() << std::endl;
    std::cout << "SYCL Devices: \n \n";

    for(int i = 0; i < devCount; ++i)
    {
      if(gpu_devices[i].is_gpu()) {
	std::cout << i << " : " << gpu_devices[i].get_info<sycl::info::device::name>() << std::endl;
	std::cout << "  Global memory (mb):   " << gpu_devices[i].get_info<sycl::info::device::global_mem_size>() / mb << std::endl;
	std::cout << "  Shared memory (kb):   " << gpu_devices[i].get_info<sycl::info::device::local_mem_size>() / kb << std::endl;
	std::cout << "  Constant memory (kb): " << gpu_devices[i].get_info<sycl::info::device::max_constant_buffer_size>() / kb << std::endl;
      }
    }
}

#endif  // enable SYCL
