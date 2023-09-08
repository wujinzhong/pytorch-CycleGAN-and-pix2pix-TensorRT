"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from InferenceUtil import (NVTXUtil, 
                           check_onnx, 
                           TRT_Engine, 
                           SynchronizeUtil, 
                           TorchUtil, 
                           Memory_Manager,
                           FIFOTorchCUDATensors,
                           synchronize,
                           )
import torch
import warnings
import nvtx
import cvcuda

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

def torch_onnx_export_netG(onnx_model, fp16=False, onnx_model_path="model.onnx", maxBatch=1 ):
    if not os.path.exists(onnx_model_path):
        dynamic_axes = {
            "latent_model_input":   {0: "bs_x_2"},
            "prompt_embeds":        {0: "bs_x_2"},
            "noise_pred":           {0: "batch_size"}
        }

        device = torch.device("cuda:0")
        
        onnx_model2= onnx_model #onnx_model2= UNet_x(onnx_model)

        onnx_model2.eval()
        if isinstance(onnx_model2, torch.nn.DataParallel):
            onnx_model2 = onnx_model2.module

        onnx_model2 = onnx_model2.to(device=device)
        
        if fp16: dst_dtype = torch.float16
        else: dst_dtype = torch.float32

        '''
        input real_A: (torch.Size([1, 3, 256, 256]), torch.float32, device(type='cuda', index=0))
        output fake_B: (torch.Size([1, 3, 256, 256]), torch.float32, device(type='cuda', index=0))
        '''
        dummy_inputs = {
            "real_A": torch.randn((1, 3, 256, 256), dtype=dst_dtype).to(device),
        }
        # output_names = ["masks", "iou_predictions", "low_res_masks"]
        output_names = ["fake_B"]

        #import apex
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                if True:
                    torch.onnx.export(
                        onnx_model2,
                        tuple(dummy_inputs.values()),
                        onnx_model_path, #f,
                        export_params=True,
                        verbose=True,
                        opset_version=16,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                    )  
                else:
                    with open(onnx_model_path, "wb") as f:
                        torch.onnx.export(
                        onnx_model2,
                        tuple(dummy_inputs.values()),
                        f,
                        export_params=True,
                        verbose=True,
                        opset_version=18,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                        )  
    check_onnx(onnx_model_path)
    return

def main():
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.eval = True

    cvcuda_stream = cvcuda.Stream()
    cvcuda_tmp_stream = cvcuda.Stream()
    mm = Memory_Manager()
    mm.add_foot_print("prev-E2E")
    torchutil = TorchUtil(gpu=0, memory_manager=mm, cvcuda_stream=cvcuda_stream)
    save_stream = torch.cuda.Stream()

    
    load_stream = torch.cuda.ExternalStream(cvcuda_stream.handle)
    #load_stream = torch.cuda.Stream()
    
    inference_finish_event = torch.cuda.Event()
    save_finish_event = torch.cuda.Event()
    fifo_tensors = None

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    print(f"model: {model.device}")

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    if opt.export_onnx:
        assert opt.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
        print(f"export onnx: {opt.export_onnx}")
        torch_onnx_export_netG(model.netG, fp16=False, onnx_model_path=opt.export_onnx, maxBatch=1 )

    pix2pix_trt_engine = None
    if opt.export_trt:
        print(f"loading trt engine from: {opt.export_trt}")
        pix2pix_trt_engine = TRT_Engine(opt.export_trt, gpu_id=0, torch_stream=torchutil.torch_stream)
        if fifo_tensors is None:
            fifo_tensors = FIFOTorchCUDATensors(shape=pix2pix_trt_engine.output_tensors[0].shape,
                                                dtype = pix2pix_trt_engine.output_tensors[0].dtype,
                                                device=pix2pix_trt_engine.output_tensors[0].device,
                                                len=50) #50 is test dataset size, assuming here saving is too slow and inference is all run and first image is still not saved.#len=2 )
        #pix2pix_trt_engine = None

    with NVTXUtil("E2E", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        if True: #just for testing
            with NVTXUtil("warmup", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                nrg_load = nvtx.start_range(message="load", color="red")
                for i, data in enumerate(dataset):
                    nvtx.end_range(nrg_load)
                    with NVTXUtil(f"loop{i}", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                        if i >= opt.num_test:  # only apply our model to opt.num_test images.
                            break
                        
                        model.set_input(data)  # unpack data from data loader
                            
                        if pix2pix_trt_engine is not None:
                            with NVTXUtil(f"trt", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                                real_A = model.real_A
                                #real_B = model.real_B
                                trt_output = pix2pix_trt_engine.inference(inputs=[real_A.to(torch.float32),],
                                                                            outputs = pix2pix_trt_engine.output_tensors)
                                tmp = pix2pix_trt_engine.output_tensors[0]
                                fake_B = tmp.to(torch.float32)
                                model.fake_B = fake_B
                        else:
                            with NVTXUtil(f"torch", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                                model.test()           # run inference
                                fake_B = model.fake_B

                    nrg_load = nvtx.start_range(message="load", color="red")
                
                #        with NVTXUtil(f"save", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                #            visuals = model.get_current_visuals()  # get image results
                #            print("visuals:")
                #            for k,v in visuals.items():
                #                print(f"{k}")
                #            img_path = model.get_image_paths()     # get image paths
                #            
                #            if i % 5 == 0:  # save images to an HTML file
                #                print('processing (%04d)-th image... %s' % (i, img_path))
                #            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
                #webpage.save()  # save the HTML
                nvtx.end_range(nrg_load)

            with NVTXUtil("warmup", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                nrg_load = nvtx.start_range(message="load", color="red")
                for i, data in enumerate(dataset):
                    nvtx.end_range(nrg_load)
                    with NVTXUtil(f"loop{i}", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                        if i >= opt.num_test:  # only apply our model to opt.num_test images.
                            break
                        
                        model.set_input(data)  # unpack data from data loader
                            
                        #if pix2pix_trt_engine is not None:
                        if False:
                            with NVTXUtil(f"trt", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                                real_A = model.real_A
                                #real_B = model.real_B
                                trt_output = pix2pix_trt_engine.inference(inputs=[real_A.to(torch.float32),],
                                                                            outputs = pix2pix_trt_engine.output_tensors)
                                tmp = pix2pix_trt_engine.output_tensors[0]
                                fake_B = tmp.to(torch.float32)
                                model.fake_B = fake_B
                        else:
                            with NVTXUtil(f"torch", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                                model.test()           # run inference
                                fake_B = model.fake_B
                    nrg_load = nvtx.start_range(message="load", color="red")    
                #        with NVTXUtil(f"save", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                #            visuals = model.get_current_visuals()  # get image results
                #            print("visuals:")
                #            for k,v in visuals.items():
                #                print(f"{k}")
                #            img_path = model.get_image_paths()     # get image paths
                #            
                #            if i % 5 == 0:  # save images to an HTML file
                #                print('processing (%04d)-th image... %s' % (i, img_path))
                #            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
                #webpage.save()  # save the HTML
                nvtx.end_range(nrg_load)

            with NVTXUtil("inference_wo_saving", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                nrg_load = nvtx.start_range(message="load", color="red")
                for i, data in enumerate(dataset):
                    nvtx.end_range(nrg_load)
                    with NVTXUtil(f"loop{i}", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                        if i >= opt.num_test:  # only apply our model to opt.num_test images.
                            break
                        
                        model.set_input(data)  # unpack data from data loader
                            
                        if pix2pix_trt_engine is not None:
                            with NVTXUtil(f"trt", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                                real_A = model.real_A
                                #real_B = model.real_B
                                trt_output = pix2pix_trt_engine.inference(inputs=[real_A.to(torch.float32),],
                                                                            outputs = pix2pix_trt_engine.output_tensors)
                                tmp = pix2pix_trt_engine.output_tensors[0]
                                fake_B = tmp.to(torch.float32)
                                model.fake_B = fake_B
                        else:
                            with NVTXUtil(f"torch", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                                model.test()           # run inference
                                fake_B = model.fake_B
                    nrg_load = nvtx.start_range(message="load", color="red")
                nvtx.end_range(nrg_load)
            
            with NVTXUtil("inference_wo_saving", "red", mm):
                nrg_load = nvtx.start_range(message="load", color="red")
                for i, data in enumerate(dataset):
                    nvtx.end_range(nrg_load)
                    with NVTXUtil(f"loop{i}", "blue", mm):
                        if i >= opt.num_test:  # only apply our model to opt.num_test images.
                            break
                        
                        model.set_input(data)  # unpack data from data loader
                            
                        if pix2pix_trt_engine is not None:
                            with NVTXUtil(f"trt", "red", mm), torch.cuda.stream(torchutil.torch_stream):
                                real_A = model.real_A
                                #real_B = model.real_B
                                trt_output = pix2pix_trt_engine.inference(inputs=[real_A.to(torch.float32),],
                                                                            outputs = pix2pix_trt_engine.output_tensors)
                                tmp = pix2pix_trt_engine.output_tensors[0]
                                fake_B = tmp.to(torch.float32)
                                model.fake_B = fake_B
                        else:
                            with NVTXUtil(f"torch", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                                model.test()           # run inference
                                fake_B = model.fake_B
                    nrg_load = nvtx.start_range(message="load", color="red")
                nvtx.end_range(nrg_load)

        with NVTXUtil("inference_w_saving", "red", mm), SynchronizeUtil(torchutil.torch_stream):
            import queue # Queue is thread safe 
            q = queue.Queue()

            import threading

            def loading_one_data(dataset, q, idx, device, mean_tensor, stddev_tensor, load_stream):
                with NVTXUtil(f"load{idx}", "blue"), torch.cuda.stream(load_stream):
                    with NVTXUtil(f"CPU", "blue"):
                        data = dataset.__getitem__(idx, mean_tensor, stddev_tensor, cvcuda_stream)
                    dataA = data["A"]
                    dataB = data["B"]
                    #print(f"data in loading_one_data: {data.items()}")
                    #print(f"data[A]: {dataA.shape, dataA.device, dataA.dtype}")
                    #print(f"data[B]: {dataB.shape, dataB.device, dataB.dtype}")
                    if data["A"].ndim==3:
                        data["A"] = torch.unsqueeze(data["A"], dim=0).to(device, non_blocking=False) #this non_blocking=True is necessary
                        data["B"] = torch.unsqueeze(data["B"], dim=0).to(device, non_blocking=False)
                    q.put(data)

            def loading_one_data_multi_threads(thread_idx, thread_num, dataset_len, dataset, q, device, mean_tensor, stddev_tensor, load_stream):
                with NVTXUtil(f"load{thread_idx}", "blue"), torch.cuda.stream(load_stream):
                    for i in range(dataset_len//thread_num):
                        im_idx = thread_idx + i*thread_num
                        with NVTXUtil(f"CPU", "blue"):
                            data = dataset.__getitem__(im_idx, mean_tensor, stddev_tensor, cvcuda_stream)
                        dataA = data["A"]
                        dataB = data["B"]
                        #print(f"data in loading_one_data: {data.items()}")
                        #print(f"data[A]: {dataA.shape, dataA.device, dataA.dtype}")
                        #print(f"data[B]: {dataB.shape, dataB.device, dataB.dtype}")
                        if data["A"].ndim==3:
                            data["A"] = torch.unsqueeze(data["A"], dim=0).to(device, non_blocking=False) #this non_blocking=True is necessary
                            data["B"] = torch.unsqueeze(data["B"], dim=0).to(device, non_blocking=False)
                        q.put(data)
                
            print(f"dataset.__len__() = {dataset.__len__()}")
            
            mean_tensor = torch.Tensor([0.5, 0.5, 0.5])
            mean_tensor = mean_tensor.reshape(1, 1, 1, 3).cuda(0)
            mean_tensor = cvcuda.as_tensor(mean_tensor, "NHWC")
            stddev_tensor = torch.Tensor([0.5, 0.5, 0.5])
            stddev_tensor = stddev_tensor.reshape(1, 1, 1, 3).cuda(0)
            stddev_tensor = cvcuda.as_tensor(stddev_tensor, "NHWC")
            #if False: #multi thread for data loading
            #    threads = []
            #    with NVTXUtil(f"loading", "red"):
            #        thread_num = 2
            #        dataset_len = dataset.__len__()
            #        for idx in range(thread_num):
            #            t = threading.Thread( target=loading_one_data_multi_threads, args=(idx, thread_num, dataset_len, dataset, q, model.device, #mean_tensor, stddev_tensor, load_stream) )
            #            threads.append(t)
            #            t.start()
            #            
            #    #for i in range(1):
            #    #    threads[i].join()
            #else:
            #    with NVTXUtil(f"loading", "red"):
            #        
            #        for idx in range(dataset.__len__()):
            #            loading_one_data(dataset, q, idx, model.device, mean_tensor, stddev_tensor, load_stream)
            #        synchronize( load_stream )

            #for i in range(dataset.__len__()):
            #    #threads[i%thread_num].join()
            #    data = q.get()

            for i, data in enumerate(dataset):
                #with NVTXUtil(f"loop{i}", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                with NVTXUtil(f"loop{i}", "blue", mm):
                    if i >= opt.num_test:  # only apply our model to opt.num_test images.
                        break
                    
                    model.set_input(data)  # unpack data from data loader
                        
                    if pix2pix_trt_engine is not None:
                        with NVTXUtil(f"trt", "red", mm), torch.cuda.stream(torchutil.torch_stream):
                            real_A = model.real_A
                            print( f"model.real_A: {real_A.shape}" )
                            #real_B = model.real_B
                            trt_output = pix2pix_trt_engine.inference(inputs=[real_A.to(torch.float32),],
                                                                        outputs = pix2pix_trt_engine.output_tensors)
                            tmp = pix2pix_trt_engine.output_tensors[0]
                            fake_B = tmp.to(torch.float32)
                            model.fake_B = fake_B

                            fifo_tensors.push( fake_B, torchutil.torch_stream )
                            
                    else:
                        with NVTXUtil(f"torch", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                            model.test()           # run inference
                            fake_B = model.fake_B

                            fifo_tensors.push( fake_B, torchutil.torch_stream )

                    
                    if False:
                        #with NVTXUtil(f"save", "red", mm), SynchronizeUtil(save_stream):
                        with NVTXUtil(f"save", "red", mm), torch.cuda.stream(save_stream):
                            visuals = model.get_current_visuals()  # get image results
                            print("visuals:")
                            for k,v in visuals.items():
                                print(f"{k}")
                            img_path = model.get_image_paths()     # get image paths
                            
                            if i % 5 == 0:  # save images to an HTML file
                                print('processing (%04d)-th image... %s' % (i, img_path))
                            visuals["fake_B"] = fifo_tensors.pop()
                            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
                    else:
                        fifo_tensors.pop()
            webpage.save()  # save the HTML

        with NVTXUtil("inference_w_saving", "red", mm), SynchronizeUtil(torchutil.torch_stream):
            def loading_one_data(dataset, q, idx, device, mean_tensor, stddev_tensor, load_stream):
                with NVTXUtil(f"load{idx}", "blue"), torch.cuda.stream(load_stream):
                    with NVTXUtil(f"CPU", "blue"):
                        data = dataset.__getitem__(idx, mean_tensor, stddev_tensor, cvcuda_stream)
                    dataA = data["A"]
                    dataB = data["B"]
                    #print(f"data in loading_one_data: {data.items()}")
                    #print(f"data[A]: {dataA.shape, dataA.device, dataA.dtype}")
                    #print(f"data[B]: {dataB.shape, dataB.device, dataB.dtype}")
                    if data["A"].ndim==3:
                        data["A"] = torch.unsqueeze(data["A"], dim=0).to(device, non_blocking=False) #this non_blocking=True is necessary
                        data["B"] = torch.unsqueeze(data["B"], dim=0).to(device, non_blocking=False)
                    q.put(data)
                
            print(f"dataset.__len__() = {dataset.__len__()}")
            
            mean_tensor = torch.Tensor([0.5, 0.5, 0.5])
            mean_tensor = mean_tensor.reshape(1, 1, 1, 3).cuda(0)
            mean_tensor = cvcuda.as_tensor(mean_tensor, "NHWC")
            stddev_tensor = torch.Tensor([0.5, 0.5, 0.5])
            stddev_tensor = stddev_tensor.reshape(1, 1, 1, 3).cuda(0)
            stddev_tensor = cvcuda.as_tensor(stddev_tensor, "NHWC")

            for i in range( dataset.__len__() ):
                with torch.cuda.stream(torch.cuda.ExternalStream(cvcuda_tmp_stream.handle)), cvcuda_tmp_stream:
                    data = dataset.__getitem__(i, mean_tensor, stddev_tensor, cvcuda_tmp_stream)
            #for i, data in enumerate(dataset):
                #with NVTXUtil(f"loop{i}", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
                with NVTXUtil(f"loop{i}", "blue", mm):
                    if i >= opt.num_test:  # only apply our model to opt.num_test images.
                        break
                    
                    model.set_input(data)  # unpack data from data loader
                        
                    if pix2pix_trt_engine is not None:
                        with NVTXUtil(f"trt", "red", mm), torch.cuda.stream(torchutil.torch_stream):
                            real_A = model.real_A
                            print( f"model.real_A: {real_A.shape}" )
                            trt_output = pix2pix_trt_engine.inference(inputs=[real_A.to(torch.float32),],
                                                                        outputs = pix2pix_trt_engine.output_tensors)
                            tmp = pix2pix_trt_engine.output_tensors[0]
                            fake_B = tmp.to(torch.float32)
                            model.fake_B = fake_B

                            fifo_tensors.push( fake_B, torchutil.torch_stream )
                            
                    else:
                        with NVTXUtil(f"torch", "red", mm), SynchronizeUtil(torchutil.torch_stream):
                            model.test()           # run inference
                            fake_B = model.fake_B

                            fifo_tensors.push( fake_B, torchutil.torch_stream )

                    if True:
                        #with NVTXUtil(f"save", "red", mm), SynchronizeUtil(save_stream):
                        with NVTXUtil(f"save", "red", mm), torch.cuda.stream(save_stream):
                            visuals = model.get_current_visuals()  # get image results
                            print("visuals:")
                            for k,v in visuals.items():
                                print(f"{k}")
                            img_path = model.get_image_paths()     # get image paths
                            
                            if i % 5 == 0:  # save images to an HTML file
                                print('processing (%04d)-th image... %s' % (i, img_path))
                            visuals["fake_B"] = fifo_tensors.pop()
                            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
                    else:
                        fifo_tensors.pop()
            webpage.save()  # save the HTML

if __name__ == '__main__':
    main()



