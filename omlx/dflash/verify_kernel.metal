#include <metal_stdlib>
using namespace metal;

// SiliconFlash placeholder kernel.
//
// The optimized implementation will target the DFlash verify path where the
// target model validates a fixed draft block (typically M=16). Keeping this
// file in-tree now lets us wire packaging, source discovery, and future MLX
// kernel compilation without changing the surrounding module layout.
//
// Deliverable #2 will replace this with a real batched-GEMV / verify kernel.

kernel void dflash_verify_placeholder(
    device const half* in0 [[buffer(0)]],
    device half* out0 [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    out0[gid] = in0[gid];
}
