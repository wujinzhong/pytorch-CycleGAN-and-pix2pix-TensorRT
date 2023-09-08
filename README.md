# pytorch-CycleGAN-and-pix2pix-TensorRT
Performance optimizing of pytorch-CycleGAN-and-pix2pix onto NVIDIA GPUs, using TensorRT, CV-CUDA, etc.
# Overview
Original source code copied from here, https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix, thanks to junyanz for this great work. I perform GPU inference optimizing onto this baseline, via NVIDIA TensorRT, CV-CUDA and other SDKs.

Currently, I just almost finished pix2pix inference pipeline optimization. and put all my optimizing steps in this repo too for reference, for who want to know how does the inference optimizing goes from scratch.

I am still new to CUDA coding and inference optimization, any suggestions are all welcome.

Please follow here to run the baseline, [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/README.md)https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/README.md.

# System config
This is my system config scripts for reference to run my code:
> docker run --gpus all -it -v /home/your_name:/your_name --network host nvcr.io/nvidia/pytorch:22.08-py3 bash
> 
> git clone [https://github.com/wujinzhong/pytorch-CycleGAN-and-pix2pix-TensorRT](https://github.com/wujinzhong/pytorch-CycleGAN-and-pix2pix-TensorRT.git)
> 
> cd /your/projects/folder/pytorch-CycleGAN-and-pix2pix-TensorRT/
> 
> pip install onnx_graphsurgeon dominate
> 
> pip install nvidia-pyindex
> 
> pip install polygraphy
> 
> pip install nvtx
> 
> pip install moviepy
> 
> pip install onnxruntime onnxruntime-gpu
> 
> #config cvcuda
> 
> tar -xvf nvcv-lib-0.4.0_beta-cuda11-x86_64-linux.tar.xz
> 
> tar -xvf nvcv-dev-0.4.0_beta-cuda11-x86_64-linux.tar.xz
> 
> export LD_LIBRARY_PATH=/your/cvcuda/install/folder/opt/nvidia/cvcuda0/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
> 
> export PYTHONPATH=/your/cvcuda/install/folder/opt/nvidia/cvcuda0/lib/x86_64-linux-gnu/python/:$PYTHONPATH
> 
> pip install nvcv_python-0.4.0_beta-cp38-cp38-linux_x86_64.whl
> 
> #run test.py
> 
> clear && python test.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained --export_onnx "./onnx_trt/pix2pix_bs1.onnx"
>
> #build TensorRT engine via trtexec, I use RTX6000 for developing.
>
> trtexec --onnx=./onnx_trt/pix2pix_bs1.onnx --saveEngine=./onnx_trt/pix2pix_bs1_fp32_6000.engine --workspace=4096  --exportOutput=output.json
> 
> clear && python test.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained --export_onnx "./onnx_trt/pix2pix_bs1.onnx" --export_trt "./onnx_trt/pix2pix_bs1_fp32_6000.engine"
>
> #use this scrip for NSight System profiling, set your nsys dir and python dir is necessary.
> 
> clear && /usr/local/bin/nsys profile /opt/conda/bin/python test.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained --export_onnx "./onnx_trt/pix2pix_bs1.onnx" --export_trt "./onnx_trt/pix2pix_bs1_fp32_6000.engine"
> 

# Optimizing steps
## Baseline
Check source code test_0_baseline.py.
Samilar to implementation in junyanz's repo, but I use my InferenceUtil.py for NVTX annoations in NSight System timeline, using explicitly CUDA streams in torch pipeline, monitoring GPU device memory usage between each stage in the E2E pipeline.

While profiling/optimizing, I mainly use NSight System timeline to check which optimizing scheme is feasible. Actually, you should use CUDA event for accurate timing, but using NSight System is simpler, no source code invading, and do no harm to your optimizing dev.

## CUDA multi-stream for inference stage and saving images to disk stage
Check source code test_1_2_streams_for_inference_and_saving.py.

## CUDA multi-stream for data loading
Check source code test_3_multi_threads_and_cuda_stream_for_data_loading.py.
I tried to use CPU multi-thread for parallel/higher data loading/image preprocessing, but sadly failed, check the description in my document at doc/ for detail.

## CV-CUDA for image preprocessing in data loading stage
Check source code test_6_cvcuda_for_data_preprocessing_and_another_cuda_stream_for_it_to_overlapping_with_inference.py.

## Optimizing document
I summary details in this document, did some performance analysis. finally the performance speed up to baseline is 2.6X.
