#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <sys/syscall.h>
#include <unistd.h>

#if __has_include(<sycl/sycl.hpp>)
 #include <sycl/sycl.hpp>
#else
 #include <CL/sycl.hpp>
 namespace sycl = cl::sycl;
#endif

auto asyncHandler = [](sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const &e) {
      std::cerr << "Caught asynchronous SYCL exception:" << std::endl
		<< e.what() << std::endl
		<< "Exception caught at file:" << __FILE__
		<< ", line:" << __LINE__ << std::endl;
    }
  }
};

class device_ext: public sycl::device {
public:
  device_ext(): sycl::device() {}
  ~device_ext() { std::lock_guard<std::mutex> lock(m_mutex); }
  device_ext(const sycl::device& base): sycl::device(base) {}

private:
  mutable std::mutex m_mutex;
};

static inline int get_tid() { return syscall(SYS_gettid); }

class dev_mgr {
public:
  int current_device() {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto                        it = _thread2dev_map.find(get_tid());
    if(it != _thread2dev_map.end()) {
      check_id(it->second);
      return it->second;
    }
    printf("WARNING: no SYCL device found in the map, returning DEFAULT_DEVICE_ID\n");
    return DEFAULT_DEVICE_ID;
  }
  sycl::queue* current_queue() {
    return _queues[current_device()];
  }

  void select_device(int id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    check_id(id);
    _thread2dev_map[get_tid()] = id;
  }
  int device_count() { return _queues.size(); }

  /// Returns the instance of device manager singleton.
  static dev_mgr& instance() {
    static dev_mgr d_m;
    return d_m;
  }
  dev_mgr(const dev_mgr&)            = delete;
  dev_mgr& operator=(const dev_mgr&) = delete;
  dev_mgr(dev_mgr&&)                 = delete;
  dev_mgr& operator=(dev_mgr&&)      = delete;

private:
  mutable std::mutex m_mutex;

  dev_mgr() {
    sycl::device dev(sycl::gpu_selector_v);
    _queues.push_back(new sycl::queue(dev, asyncHandler, sycl::property_list{sycl::property::queue::in_order{}}));
  }

  void check_id(int id) const {
    if(id >= _queues.size()) { throw std::runtime_error("invalid device id"); }
  }

  // Note: only 1 out-of-order SYCL queue is created per device
  std::vector<sycl::queue*> _queues;

  /// DEFAULT_DEVICE_ID is used, if current_device() can not find current
  /// thread id in _thread2dev_map, which means default device should be used
  /// for the current thread.
  const int DEFAULT_DEVICE_ID = 0;
  /// thread-id to device-id map.
  std::map<int, int> _thread2dev_map;
};

/// Util function to get the current device (in int).
static inline void syclGetDevice(int* id) { *id = dev_mgr::instance().current_device(); }

/// Util function to get the current queue
static inline sycl::queue* sycl_get_queue() {
  return dev_mgr::instance().current_queue();
}

/// Util function to set a device by id. (to _thread2dev_map)
static inline void syclSetDevice(int id) { dev_mgr::instance().select_device(id); }

/// Util function to get number of GPU devices (default: explicit scaling)
static inline void syclGetDeviceCount(int* id) { *id = dev_mgr::instance().device_count(); }
