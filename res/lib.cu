#include <cuComplex.h>
#include <cuda.h>
#include <cstdint>

#define MAX_ITER 255

extern "C" __global__ void render_line(uint8_t *out, int x_offset, int y,
                                       float scale) {
  int px_index = blockIdx.x * blockDim.x + threadIdx.x;
  cuFloatComplex c =
      cuCmulf(make_cuFloatComplex((float)px_index / 2 + x_offset, y),
              make_cuFloatComplex(scale, 0));
  cuFloatComplex z = c;
  cuFloatComplex last_z = z;

  for (uint8_t i = 1; i < MAX_ITER; i += 2) {
    last_z = z;
    z = cuCaddf(cuCmulf(z, z), c);

    if (isnan(z.x)) {
      if (isnan(last_z.x)) {
        out[px_index] = MAX_ITER - i - 1;
        return;
      }
      out[px_index] = MAX_ITER - i;
      return;
    }
  }

  out[px_index] = 0;
}
