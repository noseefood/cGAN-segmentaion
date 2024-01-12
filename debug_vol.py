import SimpleITK as sitk

dataPath = './data/test_dataset/1/unsegmented_Volume.mhd'
itkimage_unSeg = sitk.ReadImage(dataPath)

print(itkimage_unSeg)