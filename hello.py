import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from utils import draw_points, load_image

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger()
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def infer(engine, context, h_input, d_input, h_output, d_output, stream, input_data):
    np.copyto(h_input, input_data.ravel())
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return h_output.reshape(engine.get_binding_shape(1))

def main():

    engine = load_engine("./weights/xfeet/xfeat.engine")
    context = engine.create_execution_context()
    h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)

    img = cv2.imread("assets/ref.png")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = np.array([img_rgb.transpose(2, 0, 1)], dtype=np.float32) / 255.0

    output = infer(engine, context, h_input, d_input, h_output, d_output, stream, img_tensor)
    print(f"Output shape: {output.shape}")

    img_out = draw_points(img, output)
    plt.imshow(img_out[..., ::-1])
    plt.show()
    cv2.imwrite("assets/ref_res_trt.png", img_out)

if __name__ == "__main__":
    main()
