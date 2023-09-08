# pytorch-CycleGAN-and-pix2pix-TensorRT
Performance optimizing of pytorch-CycleGAN-and-pix2pix onto NVIDIA GPUs, using TensorRT, CV-CUDA, etc.
# Overview
Original source code copied from here, https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix, thanks to junyanz for this great work. I perform GPU inference optimizing onto this baseline, via NVIDIA TensorRT, CV-CUDA and other SDKs.
Currently, I just almost finished pix2pix inference pipeline optimization. and put all my optimizing steps in this repo too for reference, for who want to know how does the inference optimizing goes from scratch.
I am still new to CUDA coding and inference optimization, any suggestions are all welcome.
Please follow here to run the baseline, [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/README.md)https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/README.md.

# System config
This is my system config scripts for reference to run my code:
> cd /thor/projects/TRT_DeepSpeed/pytorch-CycleGAN-and-pix2pix/
> pip install -r requirements.txt
> pip install onnx_graphsurgeon
> pip install nvidia-pyindex
> pip install polygraphy
> pip install nvtx
> cp ../../usr_lib_x86_64-linux-gnu/* /usr/lib/x86_64-linux-gnu/
> pip install opencv-python=4.2.0.32 opencv-contrib-python
> apt-get update -y && apt-get install -y libbsd-dev
> pip install moviepy
> pip install onnxruntime onnxruntime-gpu
> #config cvcuda
> apt install software-properties-common
> add-apt-repository  ppa:ubuntu-toolchain-r/test
> apt install gcc-11
> update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov  gcov /usr/bin/gcov-11
> gcc --version
> g++ --version
> tar -xvf nvcv-lib-0.4.0_beta-cuda11-x86_64-linux.tar.xz 
> tar -xvf nvcv-dev-0.4.0_beta-cuda11-x86_64-linux.tar.xz 
> tar -xvf nvcv-python3.8-0.4.0_beta-cuda11-x86_64-linux.tar.xz 
> cd export LD_LIBRARY_PATH=/thor/Downloads/opt/nvidia/cvcuda0/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
> export LD_LIBRARY_PATH=/thor/Downloads/opt/nvidia/cvcuda0/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
> export PYTHONPATH=/thor/Downloads/opt/nvidia/cvcuda0/lib/x86_64-linux-gnu/python/:$PYTHONPATH
> pip install nvcv_python-0.4.0_beta-cp38-cp38-linux_x86_64.whl 
> #run test.py
> clear && python test.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained --export_onnx "./onnx_trt/pix2pix.onnx"
> trtexec --onnx=./onnx_trt/pix2pix.onnx --saveEngine=./onnx_trt/pix2pix.engine --workspace=4096  --exportOutput=output.json
> trtexec --onnx=./onnx_trt/pix2pix.onnx --fp16 --saveEngine=./onnx_trt/pix2pix_bs1_fp16_6000.engine --workspace=4096  --exportOutput=output.json

# Optimizing steps
