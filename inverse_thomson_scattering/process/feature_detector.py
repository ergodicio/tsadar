# this will be the feature detector for the first run of the code 
# importing necesary modelues
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import matplotlib.colors as colors
import tempfile, os

#defining the main function 
def first_guess(elecData, ionData, all_axes, config):
     
    with tempfile.TemporaryDirectory() as tdir: #temporary directory to save figures
        if config["other"]["extraoptions"]["load_ion_spec"]: # ploting the iaw spec w/out lineouts
            X, Y = np.meshgrid(all_axes["iaw_x"], all_axes["iaw_y"])
            fig1, ax1 = plt.subplots()
            ax1.axis('off')
            ax1.pcolormesh(
                X,
                Y,
                ionData,
                cmap="gray",
                vmin=0,
                vmax=0.05*np.amax(ionData),
            )
            fig1_path = os.path.join(tdir,'temp_iaw.png') #path for iaw plot 
            fig1.savefig(fig1_path , bbox_inches='tight') #save in temp dir
            #openinn iaw and geting roi
            iaw1 = cv.imread(fig1_path)
            img_iaw = iaw1[70:390,10:497]

            # convert the img to grayscale (migth delete later if the image is ploter in gray scale)
            gray_iaw =cv.cvtColor(img_iaw,cv.COLOR_BGR2GRAY)
            ret,bw_iaw = cv.threshold(gray_iaw, 127, 255, cv.THRESH_BINARY)  # unpack the result from Adaptative threshold into two variables
            kernel = np.ones((2,2),np.uint8 ) #kernel for eroding

            eroded = cv.erode(bw_iaw, kernel, iterations = 2) #bw_iaw holds the eroded image
            corners_iaw = cv.goodFeaturesToTrack(eroded,20,0.01,20) # eroded is now single channel
            corners_iaw = np.int0(corners_iaw)

            #list for corners coordinates
            corner_list_iaw = []
            for i in corners_iaw:
                x,y = i.ravel()
                cv.circle(img_iaw,(x,y),3,255,-1)
                corner_list_iaw.append((x,y))

            #Finding the min and max x and y points of the feature, and mapping the to the original image
            corner_list_iaw.sort()  # Sort in-place by x-coordinate (default behavior)
            x_min = corner_list_iaw[0][0]  # Extract only the x-value
            OGX_min = x_min + 0
            x_max = corner_list_iaw[-1][0]  # Extract only the x-value
            OGx_max = x_max + 10
            corner_list_iaw.sort(key=lambda point: point[1])  # Sort in-place by y-coordinate
            y_min = corner_list_iaw[0][1]  # Extract only the y-value
            OGy_min = y_min + 60
            y_max = corner_list_iaw[-1][1] # Extract only the y-value
            OGy_max = y_max + 70

            #maping findings to their corresponding parameters
            lineout_start = OGX_min
            lineout_end = OGx_max
            iaw_max = OGy_max
            iaw_max = OGy_min
            iaw_cf = (OGy_min + OGy_max)/2

        if config["other"]["extraoptions"]["load_ele_spec"]: #plotting the epw spec wi/out lineouts
            X, Y = np.meshgrid(all_axes["epw_x"], all_axes["epw_y"])
            fig2, ax2 = plt.subplots()
            ax2.axis('off')
            ax2.pcolormesh(
                X,
                Y,
                elecData,
                cmap="hot",
                vmin=0,
                vmax=0.15*np.amax(elecData)
            )
            """fig2_path = os.path.join(tdir, 'temp_epw.png') #path for epw spectra 
            fig2.savefig(fig2_path, bbox_inches='tight') #saving fig in temp dir
            #openinn iaw and geting roi
            epw1 = cv.imread(fig2_path)"""
            img_epw = elecData[70:390,10:497]

            # convert the img to grayscale (migth delete later if the image is ploter in gray scale)
            gray_epw =cv.cvtColor(img_epw,cv.COLOR_BGR2GRAY)
            ret,bw_epw = cv.threshold(gray_epw, 127, 255, cv.THRESH_BINARY)  # unpack the result from Adaptative threshold into two variables
            kernel = np.ones((2,2),np.uint8 ) #kernel for eroding

            eroded = cv.erode(bw_epw, kernel, iterations = 2) #bw_iaw holds the eroded image
            corners_epw = cv.goodFeaturesToTrack(eroded,20,0.01,20) # eroded is now single channel
            corners_epw = np.int0(corners_epw)

            #list for corners coordinates
            corner_list_epw = []
            for i in corners_epw:
                x,y = i.ravel()
                cv.circle(img_epw,(x,y),3,255,-1)
                corner_list_epw.append((x,y))

            #Finding the min and max x and y points of the feature, and mapping the to the original image
            corner_list_epw.sort()  # Sort in-place by x-coordinate (default behavior)
            x_min = corner_list_epw[0][0]  # Extract only the x-value
            OGX_min = x_min + 0
            x_max = corner_list_epw[-1][0]  # Extract only the x-value
            OGx_max = x_max + 10
            corner_list_iaw.sort(key=lambda point: point[1])  # Sort in-place by y-coordinate
            y_min = corner_list_epw[0][1]  # Extract only the y-value
            OGy_min = y_min + 60
            y_max = corner_list_epw[-1][1] # Extract only the y-value
            OGy_max = y_max + 70

            #maping findings to their corresponding parameters
            lineout_start = OGX_min
            lineout_end = OGx_max
            iaw_max = OGy_max
            iaw_max = OGy_min
            iaw_cf = (OGy_min + OGy_max)/2









"""  FIRST ATTEMPT AT MAKING AND TEMP SAVING THE PLOTS
    with tempfile.TemporaryDirectory() as tdir:
        #polting figure like data_visualizer.py but without the lineouts 
        if config["other"]["extraoptions"]["load_ion_spec"]:
            X, Y = np.meshgrid(all_axes["iaw_x"], all_axes["iaw_y"])
            fig, ax1 = plt.subplots()
            ax1.axis('off')
            cb = ax.pcolormesh(
                X,
                Y,
                ionData,
                cmap="gray",
                vmin=0,
                vmax=0.05*np.amax(ionData),
            )
            fig.savefig(tdir + '/temp_iaw.png',bbox_inches='tight')
            plt.close(fig)
        if config["other"]["extraoptions"]["load_ele_spec"]:
                X, Y = np.meshgrid(all_axes["epw_x"], all_axes["epw_y"])

                fig, ax = plt.subplots()
                jc= ax.pcolormesh(
                    X,
                    Y,
                    elecData,
                    cmap="hot",
                    vmin=0,
                    vmax=0.15*np.amax(elecData)
                )
                fig.savefig(tdir + '/temp_epw.png', bbox_inches='tight')

   

        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "plots"), exist_ok=True)
            # until this can be made interactive this plots all the data regions
            if config["other"]["extraoptions"]["load_ion_spec"]:
                X, Y = np.meshgrid(all_axes["iaw_x"], all_axes["iaw_y"])

                fig, ax = plt.subplots()
                cb = ax.pcolormesh(
                    X,
                    Y,
                    ionData,
                    cmap="gray",
                    #vmin=np.amin(ionData),
                    vmin=0,
                    vmax=0.05*np.amax(ionData),
                )
                fig.savefig(os.path.join(td, "ion1.png"), bbox_inches="tight")

            if config["other"]["extraoptions"]["load_ele_spec"]:
                X, Y = np.meshgrid(all_axes["epw_x"], all_axes["epw_y"])

                fig, ax = plt.subplots()
                jc= ax.pcolormesh(
                    X,
                    Y,
                    elecData,
                    #norm=colors.SymLogNorm( linthresh = 0.03, linscale = 0.03,vmin=0, vmax= np.amax(elecData)),
                    cmap="hot",
                    shading= "auto",
                    #vmin=np.amin(elecData),
                    vmin=0,
                    vmax=0.15*np.amax(elecData)
                )
                fig.savefig(os.path.join(td, "plots", "electron_fit_ranges.png"), bbox_inches="tight")

            mlflow.log_artifacts(td)
    #plotting the raw data without lineouts using functions from data visualizer 
#def plot_raw_data(): """