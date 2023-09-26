import monai

metric_ssim = monai.losses.ssim_loss.SSIMLoss(spatial_dims=2)