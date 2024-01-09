'''Used in windows environment'''
from glob import glob
import cv2
import time
import numpy as np
from Metric_cal import Calculate_Error, calculate_metrics, calculate_iou, dice, Post_processing
import pandas as pd

from models import NetworkInference_GANVer2, NetworkInference_Unet, NetworkInference_DeepLabV3, NetworkInference_DeepLabV3PLUS, NetworkInference_UnetPLUSPLUS, NetworkInference_GAN_FirstStage

colinear_filter = False
Video_recording = False
visualizaion = True

print_results = True
csv_recording = True
Excel_recording = True

class Evaluation():
    
    def __init__(self, mode = "Insertion", dataset = 3):

        # initialize models
        self.net_GAN = NetworkInference_GANVer2("pork") 
        self.net_Unet = NetworkInference_Unet("pork", method = "Unet")  
        self.net_AttUnet = NetworkInference_Unet("pork", method = "AttentionUnet")  
        self.net_DeeplabPlus = NetworkInference_DeepLabV3PLUS()
        self.net_UnetPlusPlus = NetworkInference_UnetPLUSPLUS()
        self.net_GAN_FirstStage = NetworkInference_GAN_FirstStage("pork") # for first stage only using focal loss

        # initialize Calculate_Error object
        self.Calculate_Error_object = Calculate_Error(colinear_processing = colinear_filter)
        self.colinear_used = "Colinear" if colinear_filter else "NoColinear"

        # initialize data
        self.mode = mode
        self.dataPath = './data/test_dataset/' + str(dataset)
        print("dataPath:", self.dataPath)
        self.imgs = glob(self.dataPath + '/imgs' +"/*.png")
        self.masks = glob(self.dataPath + '/masks' +"/*.png") 
        self.imgs.sort()
        self.masks.sort()

        self.video_dirs = {
            'mask': './results/mask_' + mode + str(dataset) + self.colinear_used +'.avi',
            'GAN': './results/GAN_' + mode + str(dataset) + self.colinear_used + '.avi',
            'Unet': './results/Unet_' + mode + str(dataset) + self.colinear_used + '.avi',
            'AttUnet': './results/AttUnet_' + mode + str(dataset) + self.colinear_used + '.avi',
            'DeeplabV3Plus': './results/DeeplabPlus_' + mode + str(dataset) + self.colinear_used + '.avi',
            'UnetPlusPlus': './results/UnetPlusPlus_' + mode + str(dataset) + self.colinear_used + '.avi',
            'GAN_FirstStage': './results/GAN_FirstStage_' + mode + str(dataset) + self.colinear_used + '.avi',
            'img': './results/img_' + mode + str(dataset) + self.colinear_used + '.avi'
        }

    def start(self):

        assert len(self.imgs) == len(self.masks), \
            f'len(self.imgs) should be equal to len(self.masks), but got {len(self.imgs)} vs {len(self.masks)}'
        print("All frames number:", len(self.imgs))


        models = {
            'GAN': self.net_GAN,
            'Unet': self.net_Unet,
            'AttUnet': self.net_AttUnet,
            'DeeplabV3Plus': self.net_DeeplabPlus,
            'UnetPlusPlus': self.net_UnetPlusPlus,
            'GAN_FirstStage': self.net_GAN_FirstStage
        }


        if Video_recording:
            # initialize videoWriter
            self.videoWriter_mask = None
            self.videoWriter_GAN = None
            self.videoWriter_Unet = None
            self.videoWriter_AttUnet = None
            self.videoWriter_DeeplabV3Plus = None
            self.videoWriter_UnetPlusPlus = None
            self.videoWriter_GAN_FirstStage = None
            self.videoWriter_img = None

            fps = 25
            img_size = (671,657) # 50 mm depth
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            # Create VideoWriter
            for video_name, video_dir in self.video_dirs.items():
                isColor = True if video_name == 'img' else False
                setattr(self, f"self.videoWriter_{video_name}", cv2.VideoWriter(video_dir, fourcc, fps, img_size, isColor=isColor))

            video_writers = {
                'mask': self.videoWriter_mask,
                'GAN': self.videoWriter_GAN,
                'Unet': self.videoWriter_Unet,
                'AttUnet': self.videoWriter_AttUnet,
                'DeeplabV3Plus': self.videoWriter_DeeplabV3Plus,
                'UnetPlusPlus': self.videoWriter_UnetPlusPlus,
                'GAN_FirstStage': self.videoWriter_GAN_FirstStage,
                'img': self.videoWriter_img
            }


        metrics = ['Recall', 'Precision', 'F2', 'dice', 'iou', 'Continuity', 'TipError', 'Angle']
        models_metrics = {model: {**{metric: [] for metric in metrics}, 'output': None} for model in models.keys()}

        ######################## iterate all frames ##########################
        for i, (input, target) in enumerate(zip(self.imgs, self.masks)):

            # load image and mask
            img = cv2.imread(input)
            true_mask = cv2.cvtColor(cv2.imread(target), cv2.COLOR_BGR2GRAY) * 255  # 0-1 -> 0-255
            

            for model_name, model in models.items():
                output = model.inference(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))  # 0-1
                output = cv2.normalize(output, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)  # 0-1 -> 0-255
                models_metrics[model_name]['output'] = output

            if visualizaion:
                for model_name in models.keys():
                    cv2.imshow(f"output_{model_name}", models_metrics[model_name]['output'])
                cv2.waitKey(2) 
                    
            if Video_recording:
                video_writers['mask'].write(true_mask)
                video_writers['img'].write(img)
                for model_name in models.keys():
                    video_writers[model_name].write(models_metrics[model_name]['output'])

            ############################# Metrics in current frame #############################
                    
            # dice metric/iou metric/F2/Precision/Recall Pure segmentation metrics(No post-processing)
            for model_name, model in models.items():
                # F2/Precision/Recall
                recall, precision, F2 = calculate_metrics(models_metrics[model_name]['output'], true_mask)
                models_metrics[model_name]['Recall'].append(recall)
                models_metrics[model_name]['Precision'].append(precision)
                models_metrics[model_name]['F2'].append(F2)

                # dice/iou/
                models_metrics[model_name]['dice'].append(dice(models_metrics[model_name]['output'], true_mask, k = 255))
                models_metrics[model_name]['iou'].append(calculate_iou(true_mask, models_metrics[model_name]['output']))


            # continuity metric
            continuity_method = "LineProj"
            for model_name, model in models.items():
                models_metrics[model_name]['Continuity'].append(self.Calculate_Error_object.Calculate_Continuity(method = continuity_method, pred = models_metrics[model_name]['output'], mask = true_mask))

            # TipError/Angle error metric
            for model_name, model in models.items():
                TipError, AngleError = self.Calculate_Error_object.Calculate_TipError(pred=models_metrics[model_name]['output'], mask=true_mask, method="canny")
                models_metrics[model_name]['TipError'].append(TipError)
                models_metrics[model_name]['Angle'].append(AngleError)
                    
        ######################## iterate all frames ##########################

        if print_results:
            # iterate every metrics
            for metric in metrics:
                for model in models:
                    # Mean and std
                    print(f'mean {metric} based {model}:', np.nanmean(models_metrics[model][metric]), f'std {metric} based {model}:', np.nanstd(models_metrics[model][metric]))


        if csv_recording:

            # save all metrics in csv(Without output)

            # Remove 'output' from each model's metrics
            for model in models_metrics.keys():
                del models_metrics[model]['output']

            # Convert the nested dictionary to a pandas DataFrame
            df = pd.DataFrame(models_metrics)

            # Save the DataFrame to a CSV file(mode and dataset)
            df.to_csv(f'./results/df_{self.mode}_{str(dataset)}'+ self.colinear_used + '.csv')

            if Excel_recording:

                # Save the DataFrame to a Excel file(mode and dataset)
                # df.to_excel(f'./results/df_{self.mode}_{str(dataset)}.xlsx')
    
                writer = pd.ExcelWriter('./results/metrics_' + self.mode + str(dataset) + self.colinear_used + '.xlsx', engine='xlsxwriter')
                df_temp = pd.DataFrame()

                # excel: every page is a Matrics, every colume is a metric
                for metric in metrics:
                    data = {}  # Initialize an empty dictionary
                    for model in models:
                        temp = df.loc[metric, model]
                        temp_array = np.array(temp)
                        data[model] = pd.Series(temp_array)  # Add the new data to the dictionary

                    df_temp = pd.DataFrame(data)  # Convert the dictionary to a DataFrame
                    df_temp.to_excel(writer, sheet_name=metric)        
                
                # Save the Excel file
                writer.save()


        if Video_recording:
            # Release everything if job is finisheds
            for video_name, video_writer in video_writers.items():
                video_writer.release()
        

if __name__ == "__main__":
    test_mode = "Insertion"
    dataset = 3
    eval = Evaluation(mode = test_mode, dataset = dataset)
    eval.start()