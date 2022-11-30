import numpy as np
import torch 
from collections import defaultdict
from typing import Union, Tuple, Dict
import torch
import numpy as np
import scipy.ndimage as ndi

from torchio.typing import TypeData, TypeTripletFloat, TypeSextetFloat
from torchio.data.subject import Subject
from torchio.transforms.intensity_transform import IntensityTransform
from torchio.transforms.augmentation.random_transform import RandomTransform


class RandomSharpen(RandomTransform, IntensityTransform):
    """Sharpening augmentation for BigAug

    """
    def __init__(
            self,
            alpha,
            std1: Union[float, Tuple[float, float]] = (0, 2),
            std2: Union[float, Tuple[float, float]] = (0, 2),
            **kwargs
    ):
        super().__init__(**kwargs)
        self.std_ranges1 = self.parse_params(std1, None, 'std1', min_constraint=0)
        self.std_ranges2 = self.parse_params(std2, None, 'std2', min_constraint=0)
        self.alpha_range = alpha

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict) 
        for name in self.get_images_dict(subject):
            std1 = self.get_params(self.std_ranges1)
            std2 = self.get_params(self.std_ranges2)
            alpha = torch.randint(low=self.alpha_range[0], high=self.alpha_range[1], size=(1,1))[0,0]
            arguments['std1'][name] = std1
            arguments['std2'][name] = std2
            arguments['alpha'][name] = alpha 
        transform = Sharpen(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

    def get_params(self, std_ranges: TypeSextetFloat) -> TypeTripletFloat:
        std = self.sample_uniform_sextet(std_ranges)
        return std

class Sharpen(IntensityTransform):
    """Sharpen an image using a Gaussian filter.
    """
    def __init__(
            self,
            std1: Union[TypeTripletFloat, Dict[str, TypeTripletFloat]],
            std2: Union[TypeTripletFloat, Dict[str, TypeTripletFloat]],
            alpha,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.std1 = std1
        self.std2 = std2
        self.alpha = alpha
        self.args_names = ('std1','std2', 'alpha')

    def apply_transform(self, subject: Subject) -> Subject:
        stds1 = self.std1
        stds2 = self.std2
        alpha = self.alpha
        for name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                stds1 = self.std1[name]
                stds2 = self.std2[name]
                alpha = self.alpha[name]
            stds_channels1 = np.tile(stds1, (image.num_channels, 1))
            stds_channels2 = np.tile(stds2, (image.num_channels, 1))
            transformed_tensors = []
            for std1, std2, channel in zip(stds_channels1, stds_channels2, image.data):
                blurred_tensor = self.blur(
                    channel,
                    image.spacing,
                    std1,
                )
                filter_blurred_tensor = self.blur(
                    blurred_tensor,
                    image.spacing,
                    std2,
                )
                img = blurred_tensor + alpha*(blurred_tensor- filter_blurred_tensor)

                transformed_tensors.append(img)
            image.set_data(torch.stack(transformed_tensors))
        return subject

    @staticmethod
    def blur(
            data: TypeData,
            spacing: TypeTripletFloat,
            std_physical: TypeTripletFloat,
    ) -> torch.Tensor:
        assert data.ndim == 3
        std_voxel = np.array(std_physical) / np.array(spacing)
        blurred = ndi.gaussian_filter(data, std_voxel)
        tensor = torch.as_tensor(blurred)
        return tensor 

class RandomBrightnessShift(RandomTransform, IntensityTransform):
    """Random Brightness shift for BigAug, randomly shift the intensity level with magnitude ranging between [a,b]
    """
    def __init__(
            self,
            shift_range,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.shift_range= shift_range

    def apply_transform(self, subject: Subject) -> Subject:
        shift_val= self.sample_uniform(self.shift_range[0], self.shift_range[1])
        transform = BrightnessShift(shift_val)
        transformed = transform(subject)
        return transformed

   
class BrightnessShift(IntensityTransform):
    """ Brightness shift for BigAug, randomly shift the intensity level with magnitude ranging between [a,b]
    """
    def __init__(
            self,
            shift_val,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.shift_val = shift_val

    def apply_transform(self, subject: Subject) -> Subject:
        for name, image in self.get_images_dict(subject).items():           
            transformed_tensors = []
            for i,channel in enumerate(image.data):
                img = self.shift_val+channel
                transformed_tensors.append(img)
            image.set_data(torch.stack(transformed_tensors))
        return subject

class RandomIntensityPerturbation(RandomTransform, IntensityTransform):
    """Random Intensity perturbation for BigAug, randomly shift the intensity level with a magnitude ranging between [a,b]

    """
    def __init__(
            self,
            shift_range,
            scale_range,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.shift_range= shift_range
        self.scale_range = scale_range

    def apply_transform(self, subject: Subject) -> Subject:
        shift_val= self.sample_uniform(self.shift_range[0], self.shift_range[1])
        scale_val= self.sample_uniform(self.scale_range[0], self.scale_range[1])
        transform = IntensityPerturbation(shift_val, scale_val)
        transformed = transform(subject)
        return transformed

   
class IntensityPerturbation(IntensityTransform):
    """ Brightness shift for BigAug, randomly shift the intensity level with magnitude ranging between [a,b]
    """
    def __init__(
            self,
            shift_val,
            scale_val,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.shift_val = shift_val
        self.scale_val = scale_val 

    def apply_transform(self, subject: Subject) -> Subject:
        for name, image in self.get_images_dict(subject).items():           
            transformed_tensors = []
            for i,channel in enumerate(image.data):
                img = self.shift_val+self.scale_val*channel
                transformed_tensors.append(img)
            image.set_data(torch.stack(transformed_tensors))
        return subject