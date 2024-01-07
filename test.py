models = ['GAN', 'Unet', 'AttUnet', 'DeeplabV3', 'DeeplabV3Plus']
metrics = ['Recall', 'Precision', 'F2']

data = {model: {metric: [] for metric in metrics} for model in models}

print(data)