#include <cuComplex.h>
#include <cuda.h>
#include <stdint.h>

#define MAX_ITER 255

extern "C" __global__ void render_line(uint8_t *out, int x_offset, int y,
                                       float scale) {
  int px_index = blockIdx.x * blockDim.x + threadIdx.x;
  cuFloatComplex c =
      cuCmulf(make_cuFloatComplex((float)px_index / 2 + x_offset, y),
              make_cuFloatComplex(scale, 0));
  cuFloatComplex z = c;

  for (uint8_t i = 1; i < MAX_ITER; ++i) {
    z = cuCaddf(cuCmulf(z, z), c);

    if (isnan(z.x) || isnan(z.y)) {
      out[px_index] = MAX_ITER - i;
      return;
    }
  }

  out[px_index] = 0;
}
