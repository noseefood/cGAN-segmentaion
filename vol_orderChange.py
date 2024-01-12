
import SimpleITK as sitk





def convert_coordinate_order(itk_image):
    '''
    change the order of the coordinate system from (z,y,x) to (x,y,z)  !!! for dataset 5
    '''
    # Create a permutation vector
    permutation_vector = [2, 1, 0]

    # Permute the axes of the image
    itk_image_permuted = sitk.PermuteAxes(itk_image, permutation_vector)

    return itk_image_permuted


itk_image = sitk.ReadImage("/home/xuesong/CAMP/segment/cGAN-segmentaion/data/test_dataset/5/unsegmented_Volume.mhd")
new_itk_image = convert_coordinate_order(itk_image)

#save the new image 
sitk.WriteImage(new_itk_image, "/home/xuesong/CAMP/segment/cGAN-segmentaion/data/test_dataset/5/unsegmented_Volume_new.mhd")