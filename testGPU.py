
import tensorflow as tf

# tf.test.is_gpu_available(
#     cuda_only=False, min_cuda_compute_capability=None
# )
tf.compat.v1.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)