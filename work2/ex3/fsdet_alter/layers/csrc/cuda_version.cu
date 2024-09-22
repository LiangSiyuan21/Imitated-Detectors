#include <cuda_runtime_api.h>

namespace fsdet_alter {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace fsdet_alter
