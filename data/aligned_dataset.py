import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import cvcuda
import nvcv
import torch
import numpy as np
import numbers
import copy
import torchvision.transforms as transforms
from InferenceUtil import (NVTXUtil)
        

def generate_data(shape, dtype, max_random=None, rng=None):
    """Generate data as numpy array

    Args:
        shape (tuple or list): Data shape
        dtype (numpy dtype): Data type (e.g. np.uint8)
        max_random (number or tuple or list): Maximum random value
        rng (numpy random Generator): To fill data with random values

    Returns:
        numpy.array: The generated data
    """
    if rng is None:
        data = np.zeros(shape, dtype=dtype)
    else:
        if max_random is not None and type(max_random) in {tuple, list}:
            assert len(max_random) == shape[-1]
        if issubclass(dtype, numbers.Integral):
            if max_random is None:
                max_random = [np.iinfo(dtype).max for _ in range(shape[-1])]
            data = rng.integers(max_random, size=shape, dtype=dtype)
        elif issubclass(dtype, numbers.Real):
            if max_random is None:
                max_random = [1.0 for _ in range(shape[-1])]
            data = rng.random(size=shape, dtype=dtype) * np.array(max_random)
            data = data.astype(dtype)
    return data

def to_cuda_buffer(host):
    orig_dtype = copy.copy(host.dtype)

    # torch doesn't accept uint16. Let's make it believe
    # it is handling int16 instead.
    if host.dtype == np.uint16:
        host.dtype = np.int16

    dev = torch.as_tensor(host, device="cuda").cuda()
    host.dtype = orig_dtype  # restore it

    class CudaBuffer:
        __cuda_array_interface = None
        obj = None

    # The cuda buffer only needs the cuda array interface.
    # We can then set its dtype to whatever we want.
    buf = CudaBuffer()
    buf.__cuda_array_interface__ = dev.__cuda_array_interface__
    buf.__cuda_array_interface__["typestr"] = orig_dtype.str
    buf.obj = dev  # make sure it holds a reference to the torch buffer

    return buf

def to_nvcv_tensor(host_data, layout):
    """Convert a tensor in host data with layout to nvcv.Tensor

    Args:
        host_data (numpy array): Tensor in host data
        layout (string): Tensor layout (e.g. NC, HWC, NHWC)

    Returns:
        nvcv.Tensor: The converted tensor
    """
    return nvcv.as_tensor(to_cuda_buffer(host_data), layout=layout)

def create_tensor(shape, dtype, layout, max_random=None, rng=None, transform_dist=None):
    """Create a tensor

    Args:
        shape (tuple or list): Tensor shape
        dtype (numpy dtype): Tensor data type (e.g. np.uint8)
        layout (string): Tensor layout (e.g. NC, HWC, NHWC)
        max_random (number or tuple or list): Maximum random value
        rng (numpy random Generator): To fill tensor with random values
        transform_dist (function): To transform random values (e.g. MAKE_ODD)

    Returns:
        nvcv.Tensor: The created tensor
    """
    h_data = generate_data(shape, dtype, max_random, rng)
    if transform_dist is not None:
        vec_transform_dist = np.vectorize(transform_dist)
        h_data = vec_transform_dist(h_data)
        h_data = h_data.astype(dtype)
    return to_nvcv_tensor(h_data, layout)

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        print(f"in AlignedDataset.__getitem__()")

        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        print(f"transform_params: {transform_params}")
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        print(f"A: {A}")
        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
    
    def __getitem_cvcuda__(self, index, mean_tensor, stddev_tensor, cvcuda_stream):
        
        print(f"in AlignedDataset.__getitem_cvcuda__()")

        transform = transforms.Compose([
            #transforms.PILToTensor() # don't use this, or it will fail at cvcuda.as_tensor(t, "NHWC")
            transforms.ToTensor()
        ])

        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        with NVTXUtil(f"open", "red"):
            AB = Image.open(AB_path).convert('RGB')

        if True:
            if False:
                w, h = AB.size
                w2 = int(w / 2)
                AB = transform(AB).unsqueeze(dim=0).to(device="cuda:0") #don't know why this not work.
                #AB = torch.zeros((1,256,512,3), device="cuda:0") #this work
                #if index==0:
                #    print(f"AB: {AB, AB.shape, AB.device, AB.dtype}")
                ccAB = cvcuda.as_tensor(AB, "NCHW")
                ccAB = cvcuda.reformat(ccAB, "NHWC")
                #print(f"ccAB: {ccAB.shape, ccAB.dtype}")

                rectA = cvcuda.RectI(x=0 , y=0, width=w2,    height=h)
                rectB = cvcuda.RectI(x=w2, y=0, width=w-w2 , height=h)
                #print(f"rectA: {rectA}")
                #print(f"rectA: {rectB}")
                cropA = cvcuda.customcrop( ccAB, rectA )
                cropB = cvcuda.customcrop( ccAB, rectB )

                layout = "NHWC"
                out_tensorA = cvcuda.Tensor(cropA.shape, cropA.dtype, layout)
                out_tensorB = cvcuda.Tensor(cropB.shape, cropB.dtype, layout)

                num_images = 1
                RNG = np.random.default_rng(0)
                basep = (((1, 1, 1, 4), np.float32, "NHWC"))
                scalep = (((1, 1, 1, 4), np.float32, "NHWC"))
                base = cvcuda.Tensor(*basep)
                scale = cvcuda.Tensor(*scalep)
                flipCode = create_tensor(
                    (num_images, 1), np.int32, "NC", max_random=1, rng=RNG
                )
                gscale = 1
                gshift = 0
                eps = 1
                flags = cvcuda.NormalizeFlags.SCALE_IS_STDDEV
                border = cvcuda.Border.REPLICATE
                bvalue = 0

                tmpA = cvcuda.crop_flip_normalize_reformat_into(
                    dst=out_tensorA,
                    src=ccAB,
                    rect=cropA,
                    flip_code=flipCode,
                    base=base,
                    scale=scale,
                    globalscale=gscale,
                    globalshift=gshift,
                    epsilon=eps,
                    flags=flags,
                    border=border,
                    bvalue=bvalue,
                )

                assert tmpA is out_tensorA

                tmpB = cvcuda.crop_flip_normalize_reformat_into(
                    dst=out_tensorB,
                    src=ccAB,
                    rect=cropB,
                    flip_code=flipCode,
                    base=base,
                    scale=scale,
                    globalscale=gscale,
                    globalshift=gshift,
                    epsilon=eps,
                    flags=flags,
                    border=border,
                    bvalue=bvalue,
                    
                )

                assert tmpB is out_tensorB

                return {'A': out_tensorA, 'B': out_tensorB, 'A_paths': AB_path, 'B_paths': AB_path}
            else:
                w, h = AB.size
                w2 = int(w / 2)
                AB = transform(AB).unsqueeze(dim=0).to(device="cuda:0") #don't know why this not work.
                #AB = torch.zeros((1,256,512,3), device="cuda:0") #this work
                #if index==0:
                #    print(f"AB: {AB, AB.shape, AB.device, AB.dtype}")
                ccAB = cvcuda.as_tensor(AB, "NCHW")
                ccAB = cvcuda.reformat(ccAB, "NHWC", stream=cvcuda_stream)
                #print(f"ccAB: {ccAB.shape, ccAB.dtype}")

                rectA = cvcuda.RectI(x=0 , y=0, width=w2,    height=h)
                rectB = cvcuda.RectI(x=w2, y=0, width=w-w2 , height=h)
                #print(f"rectA: {rectA}")
                #print(f"rectA: {rectB}")
                cropA = cvcuda.customcrop( ccAB, rectA, stream=cvcuda_stream )
                cropB = cvcuda.customcrop( ccAB, rectB, stream=cvcuda_stream )
                #print(f"cropA: {cropA.shape, cropA.dtype}")
                #print(f"cropB: {cropB.shape, cropB.dtype}")

                #mean_tensor = torch.Tensor([0.5, 0.5, 0.5])
                #mean_tensor = mean_tensor.reshape(1, 1, 1, 3).cuda(0)
                #mean_tensor = cvcuda.as_tensor(mean_tensor, "NHWC")
                #stddev_tensor = torch.Tensor([0.5, 0.5, 0.5])
                #stddev_tensor = stddev_tensor.reshape(1, 1, 1, 3).cuda(0)
                #stddev_tensor = cvcuda.as_tensor(stddev_tensor, "NHWC")

                #normalizedA = cvcuda.convertto(cropA, np.float32, scale=1 / 255)
                #normalizedB = cvcuda.convertto(cropB, np.float32, scale=1 / 255)
                normalizedA = cropA
                normalizedB = cropB

                #device = "cuda:0"
                #if index==0:
                #    print(f"before normalize: {torch.as_tensor( normalizedA.cuda(), device=device).cpu()}")

                # Normalize with mean and std-dev.
                normalizedA = cvcuda.normalize(
                    normalizedA,
                    base= mean_tensor,
                    scale= stddev_tensor,
                    flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
                    stream=cvcuda_stream
                )
                #if index==0:
                #    print(f"after normalize: {torch.as_tensor( normalizedA.cuda(), device=device).cpu()}")

                normalizedA = cvcuda.reformat(normalizedA, "NCHW", stream=cvcuda_stream)

                normalizedB = cvcuda.normalize(
                    normalizedB,
                    base= mean_tensor,
                    scale= stddev_tensor,
                    flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
                    stream=cvcuda_stream
                )
                normalizedB = cvcuda.reformat(normalizedB, "NCHW", stream=cvcuda_stream)

                if isinstance(normalizedA, torch.Tensor):
                    if not normalizedA.is_cuda:
                        normalizedA = normalizedA.to("cuda:0")
                else:
                    # Convert CVCUDA tensor to Torch tensor.
                    normalizedA = torch.as_tensor( normalizedA.cuda(), device="cuda:0" )

                if isinstance(normalizedB, torch.Tensor):
                    if not normalizedB.is_cuda:
                        normalizedB = normalizedB.to("cuda:0")
                else:
                    # Convert CVCUDA tensor to Torch tensor.
                    normalizedB = torch.as_tensor( normalizedB.cuda(), device="cuda:0" )
                
                #if index==0:
                #    print(f"A_transform: {normalizedA.cpu()}")

                return {'A': normalizedA, 'B': normalizedB, 'A_paths': AB_path, 'B_paths': AB_path}

                if False:
                    layout = "NHWC"
                    out_tensor = cvcuda.Tensor(cropA.shape, cropA.dtype, layout)

                    num_images = 1
                    RNG = np.random.default_rng(0)
                    basep = (((1, 1, 1, 4), np.float32, "NHWC"))
                    scalep = (((1, 1, 1, 4), np.float32, "NHWC"))
                    base = cvcuda.Tensor(*basep)
                    scale = cvcuda.Tensor(*scalep)
                    flipCode = create_tensor(
                        (num_images, 1), np.int32, "NC", max_random=1, rng=RNG
                    )
                    gscale = 1
                    gshift = 0
                    eps = 1
                    flags = cvcuda.NormalizeFlags.SCALE_IS_STDDEV
                    border = cvcuda.Border.REPLICATE
                    bvalue = 0
                    '''
                    @t.mark.parametrize(
                        "format,num_images,min_size,max_size,border,bvalue,basep,scalep,gscale,gshift,eps,flags,ch,dtype,layout",
                        [
                            (
                                cvcuda.Format.RGBA8,
                                1,
                                (10, 10),
                                (20, 20),
                                cvcuda.Border.REPLICATE,
                                0,
                                (((1, 1, 1, 4), np.float32, "NHWC")),
                                (((1, 1, 1, 4), np.float32, "NHWC")),
                                1,
                                2,
                                3,
                                cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
                                4,
                                np.uint8,
                                "NHWC",
                            ),
                        ],
                    )
                    '''
                    tmp = cvcuda.crop_flip_normalize_reformat_into(
                        dst=out_tensor,
                        src=ccAB,
                        rect=cropA,
                        flip_code=flipCode,
                        base=base,
                        scale=scale,
                        globalscale=gscale,
                        globalshift=gshift,
                        epsilon=eps,
                        flags=flags,
                        border=border,
                        bvalue=bvalue,
                    )

                    assert tmp is out_tensor
        else:
            # split AB image into A and B
            w, h = AB.size
            w2 = int(w / 2)
            A = AB.crop((0, 0, w2, h))
            B = AB.crop((w2, 0, w, h))

            # apply the same transform to both A and B
            transform_params = get_params(self.opt, A.size)
            print(f"transform_params: {transform_params}")
            A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
            B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
            '''
            transform_params: {'crop_pos': (0, 0), 'flip': True}
            resize: Resize(size=[256, 256], interpolation=bicubic, max_size=None, antialias=None)
            __crop: ((0, 0), 256)
            Normalize
            '''
            A = A_transform(A)
            if index==0:
                print(f"A_transform: {A}")
            B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
