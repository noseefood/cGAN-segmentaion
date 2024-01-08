# Read in the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# convert the csv(eval_baeslineWin output) to excel


df_1 = pd.read_csv('./results/df_1.csv', index_col=0)
df_2 = pd.read_csv('./results/df_2.csv', index_col=0)
df_3 = pd.read_csv('./results/df_3.csv', index_col=0)
df_4 = pd.read_csv('./results/df_4.csv', index_col=0)
df_5 = pd.read_csv('./results/df_5.csv', index_col=0)


def resize():
    

    for i, df in enumerate([df_1, df_2, df_3, df_4, df_5]):

        # new csv: every page is a model, every colume is a metric
    
        writer = pd.ExcelWriter('./results/metrics_' + str(i+1) + 'reshape.xlsx', engine='xlsxwriter')

        matrics = ['dice', 'iou', 'Recall', 'Precision', 'F2', 'TipError', 'Continuity', 'Angle']
        models = ['GAN', 'Unet', 'AttUnet', 'DeeplabV3Plus', 'UnetPlusPlus', 'GAN_FirstStage']
        # GAN,Unet,AttUnet,DeeplabV3Plus,UnetPlusPlus


        df_temp = pd.DataFrame()
        
        # excel: every page is a metric, every colume is a model
        # for model in models:
        #     data = {}  # Initialize an empty dictionary

        #     for metric in matrics:
        #         temp = df.loc[metric, model].replace("nan", "np.nan")
        #         temp_array = np.array(eval(temp))
        #         data[metric] = pd.Series(temp_array)  # Add the new data to the dictionary

        #     df_temp = pd.DataFrame(data)  # Convert the dictionary to a DataFrame
        #     df_temp.to_excel(writer, sheet_name=model)

        # excel: every page is a model, every colume is a metric
        for metric in matrics:
            data = {}  # Initialize an empty dictionary

            for model in models:
                temp = df.loc[metric, model].replace("nan", "np.nan")
                temp_array = np.array(eval(temp))
                data[model] = pd.Series(temp_array)  # Add the new data to the dictionary

            df_temp = pd.DataFrame(data)  # Convert the dictionary to a DataFrame
            df_temp.to_excel(writer, sheet_name=metric)        
        
        # Save the Excel file
        writer.save()



if __name__ == '__main__':

    resize()



