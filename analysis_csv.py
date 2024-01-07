import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Print the data
# continuity_gan = df_1.loc['Continuity', 'GAN']
# print(continuity_gan)

# Read in the data
df_1 = pd.read_csv('./results/df_1.csv', index_col=0)
df_2 = pd.read_csv('./results/df_2.csv', index_col=0)
df_3 = pd.read_csv('./results/df_3.csv', index_col=0)
df_4 = pd.read_csv('./results/df_4.csv', index_col=0)
df_5 = pd.read_csv('./results/df_5.csv', index_col=0)




def continuity_vis(df):

    # GAN,Unet,AttUnet,DeeplabV3Plus,UnetPlusPlus

    GAN_data = df.loc['Continuity', 'GAN'].replace("nan", "np.nan")
    Unet_data = df.loc['Continuity', 'Unet'].replace("nan", "np.nan")
    AttUnet_data = df.loc['Continuity', 'AttUnet'].replace("nan", "np.nan")
    DeeplabV3Plus_data = df.loc['Continuity', 'DeeplabV3Plus'].replace("nan", "np.nan")
    UnetPlusPlus_data = df.loc['Continuity', 'UnetPlusPlus'].replace("nan", "np.nan")

    # preprocess the data
    GAN_array = np.array(eval(GAN_data))
    Unet_array = np.array(eval(Unet_data))
    AttUnet_array = np.array(eval(AttUnet_data))
    DeeplabV3Plus_array = np.array(eval(DeeplabV3Plus_data))
    UnetPlusPlus_array = np.array(eval(UnetPlusPlus_data))

    # Create a range for the x-axis
    Gan_x = range(len(GAN_array))
    Unet_x = range(len(Unet_array))
    AttUnet_x = range(len(AttUnet_array))
    DeeplabV3Plus_x = range(len(DeeplabV3Plus_array))
    UnetPlusPlus_x = range(len(UnetPlusPlus_array))

    # calculate the mean and std
    GAN_mean, GAN_std = np.nanmean(GAN_array), np.nanstd(GAN_array)
    Unet_mean, Unet_std = np.nanmean(Unet_array), np.nanstd(Unet_array)
    AttUnet_mean, AttUnet_std = np.nanmean(AttUnet_array), np.nanstd(AttUnet_array)
    DeeplabV3Plus_mean, DeeplabV3Plus_std = np.nanmean(DeeplabV3Plus_array), np.nanstd(DeeplabV3Plus_array)
    UnetPlusPlus_mean, UnetPlusPlus_std = np.nanmean(UnetPlusPlus_array), np.nanstd(UnetPlusPlus_array)

    # Plot the data in different colors
    plt.plot(Gan_x, GAN_array, label='GAN', linewidth = 5.0)
    plt.plot(Unet_x, Unet_array, label='Unet')
    plt.plot(AttUnet_x, AttUnet_array, label='AttUnet')
    plt.plot(DeeplabV3Plus_x, DeeplabV3Plus_array, label='DeeplabV3Plus')
    plt.plot(UnetPlusPlus_x, UnetPlusPlus_array, label='UnetPlusPlus')

    # add the mean and std number
    # Create a string with the text
    text_str = '\n'.join([
        'GAN: mean = {:.4f}, std = {:.4f}'.format(GAN_mean, GAN_std),
        'Unet: mean = {:.4f}, std = {:.4f}'.format(Unet_mean, Unet_std),
        'AttUnet: mean = {:.4f}, std = {:.4f}'.format(AttUnet_mean, AttUnet_std),
        'DeeplabV3Plus: mean = {:.4f}, std = {:.4f}'.format(DeeplabV3Plus_mean, DeeplabV3Plus_std),
        'UnetPlusPlus: mean = {:.4f}, std = {:.4f}'.format(UnetPlusPlus_mean, UnetPlusPlus_std)
    ])

    # Add the text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.95, 0.05, text_str, transform=plt.gca().transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == '__main__':




    continuity_vis(df_3)

