#%%
import torchio as tio
from bigAug_transforms import RandomSharpen, RandomBrightnessShift, RandomIntensityPerturbation
import math
import matplotlib.pyplot as plt

path_mri = 'mri.nii.gz'
path_labels = 'labels.nii.gz'
path_name = 'patient_1'

current_subj = tio.Subject(
                data1 = tio.ScalarImage(path_mri),
                mask_labels=tio.LabelMap(path_labels),
                patient = path_name,
            )

bigaug_transforms = tio.Compose([
        tio.ToCanonical(),
        tio.ZNormalization(), 
        tio.RandomBlur(std=(0.25,1.5), p = 0.5),
        RandomSharpen(std1=(0.25,1.5),std2=(0.25,1.5), alpha=(10,30), p = 0.5), #sharpen
        tio.RandomNoise(std=(0.1,1), p = 0.5),
        RandomBrightnessShift(shift_range=(-0.1, 0.1), p = 0.5),
        tio.RandomGamma(log_gamma=(math.log(0.5), math.log(4.5)), p =0.5), #natural logarithm because torchio function takes e^value
        RandomIntensityPerturbation(shift_range=(-0.1,0.1),scale_range=(-0.1,0.1), p = 0.5),  # intensity perturbation -> multiply by scale factor and add shift factor to image 
        tio.RandomAffine(degrees=(-20,20), p = 0.5), # rotation
        tio.RandomAffine(scales=(0.4,1.6), p = 0.5), # scaling
        tio.RandomElasticDeformation(num_control_points=(7, 7, 7),locked_borders=2,p= 0.5) # deformation
    ])  



patient_dataset = tio.SubjectsDataset([current_subj], transform= bigaug_transforms)
# show a random slice to show transformation
plt.imshow(patient_dataset[0]['data1'].numpy()[0,100,:,:])
plt.show()