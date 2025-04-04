# this will be the feature detector for the first run of the code 
# importing necesary modelues
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import matplotlib.colors as colors
import tempfile, os

#defining the main function 
def first_guess(elecData, ionData, config):

    # normalizing data and accounting for feducials if necesary 
    def data_processing(data, config):

        if config["data"]["background"]["type"] == "pixel":
                #ROI without feducials for IAW
            if config["data"]["estimate_lineouts_iaw"]:
                a,b = 150,850
                # ROI without feducials for EPW 
            elif config["data"]["estimate_lineouts_epw"]:
                a, b = 200, 900
        
            feducials = data[a:b,0:1023]

            # Normalize the values to the range [0, 255]
            min_val = feducials.min()
            max_val = feducials.max()

            # Perform normalization
            feducials_normalized = ((feducials - min_val) / (max_val - min_val)) * 255.0

            # Convert the image to uint8
            img = feducials_normalized.astype(np.uint8)
            #plt.imshow(img, cmap = "gray")
            #plt.show()


        else:

            min_val = data.min()
            max_val = data.max()
            data_normalized = ((data - min_val)/(max_val - min_val))* 255.0
            img = data_normalized.astype(np.uint8)

        return img 
    
    # assuming the notch filter is located at 528 +-12 pixels divide epw image into blue and red regions    
    def notch_filter(img):

        blur_epw = cv.GaussianBlur(img,(21,21),0)

        notch_fliter_start = 516
        notch_filter_end = 540 

        if config["data"]["background"]["type"] == "pixel":
            notch_fliter_start = notch_fliter_start - 200
            notch_filter_end = notch_filter_end - 200
        else: 
            notch_fliter_start = notch_fliter_start
            notch_filter_end = notch_filter_end
        
        epw_red_box = blur_epw[:notch_fliter_start,:]
        epw_blue_box = blur_epw[notch_filter_end:,:]

        return epw_red_box, epw_blue_box
    
    #corner processing for red and blue shifted EPW
    def epw_corner_processing (epw_box):

        #gaussian blur 
        blur_box = cv.GaussianBlur(epw_box,(21,21),0)

        #find corners in eroded image
        corners = cv.goodFeaturesToTrack(blur_box, 50, 0.45, 10)
        corners = np.int0(corners).reshape(-1, 2)

        corner_list = []

        for corner in corners:
            x,y = corner.ravel()
            cv.circle(blur_box, (x, y), 5 , 255, 2)            
            corner_list.append((x,y))

        # Display the result
        #cv.imshow('Filtered Corners', blur_box)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

        #find the max and min x and y coordinates
        #print(corner_list)
        corner_list.sort()
        min_x = corner_list[0][0]  # Extract only the x-value
        max_x = corner_list[-1][0]  # Extract only the x-value

        corner_list.sort(key=lambda point: point[1])  # Sort in-place by y-coordinate
        max_y = corner_list[0][1]  # Extract only the y-value
        min_y = corner_list[-1][1] # Extract only the y-value

        #print(f"Minx: {min_x}, Maxx: {max_x}")
        #print(f"Min)y: {min_y}, Maxy: {max_y}")

        # add margins of error
        min_x -= 40
        max_x += 40
        min_y += 40
        max_y -= 40

        return min_x, max_x, min_y, max_y


    #corner detection and filtration for IAW
    def data_analysis(img):
        #dialate 

        contast= cv.convertScaleAbs(img,alpha = 1.5 )
        #plt.imshow(contast, cmap = "gist_ncar")
        #plt.show()

        #gaussian blur
        blur = cv.GaussianBlur(contast,(17,17),0)
        #plt.imshow(blur, cmap = "gist_ncar")
        #plt.show()
        #invert blurred image 
        inverted = invert(blur)
        #plt.imshow(inverted, cmap = "gist_ncar")
        #plt.show()

        #find corners in the inverted image

        corners = cv.goodFeaturesToTrack(inverted, 75, 0.01, 10)
        corners = np.int0(corners).reshape(-1, 2)

        #if corners == None:
        # if corners is empty run backup(img) 

        #filter found corners, only keep corners that have at least one neighboor within the max distance 
        filtered_corners = []
        max_distance = 50
        for i, corner in enumerate(corners):
            has_neighboor = False
            for j, other_corner in enumerate(corners):
                if i == j : 
                    continue # skip the same corner 
                # Euclaidian distance between current corner and other corners
                distance = np.linalg.norm(corner-other_corner)
                if distance <= max_distance:
                    has_neighboor = True
                    break
            if has_neighboor:
                filtered_corners.append(corner)
        #convert filtered corners into array
        filtered_corners = np.array(filtered_corners)
        
        for corner in filtered_corners:
            x,y = corner.ravel()
            
            cv.circle(contast, (x,y), 3 , (0,0,255), -1)

        # Display the result
        #cv.imshow('Filtered Corners', contast)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

        #find the max and min x and y coordinates
        min_x = filtered_corners[:, 0].min()
        #min_x = np.min(filtered_corners[:, 0])
        max_x = np.max(filtered_corners[:, 0])
        min_y = np.min(filtered_corners[:, 1])
        max_y = np.max(filtered_corners[:, 1])

        print(f"Minx: {min_x}, Maxx: {max_x}")
        print(f"Min)y: {min_y}, Maxy: {max_y}")

        # add margins of erro
        min_x -= 30
        max_x += 30
        min_y -= 30
        max_y += 30

        return min_x, max_x, min_y, max_y
    
    def backup(img):

        norm=colors.SymLogNorm( linthresh = 0.03, linscale = 0.03,vmin=np.amin(img), vmax= np.amax(img))
        normalized_data =norm(img)

        cmap=plt.get_cmap("turbo_r")
        new_map = cmap(normalized_data)
        #plt.imshow(new_map)
        #plt.show()
        gray_image = np.dot(new_map[..., :3], [0.299, 0.587, 0.114])

        # 3. Convert to 8-bit format for OpenCV processing
        gray_8bit = (gray_image * 255).astype(np.uint8)

        # 4. Adaptive Gaussian thresholding
        binary_image = cv.adaptiveThreshold(
            gray_8bit,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            11,  # Block size (must be odd)
            2    # Constant subtracted from mean
        )
        #fig1_path = os.path.join(tdir,'temp_iaw.png') #path for iaw plot 
        #fig1.savefig(fig1_path , bbox_inches='tight') #save in temp dir
        #openinn iaw and geting roi
        #iaw1 = cv.imread(fig1_path)

        # using fig without a temp directory 
        
        img_iaw = ionData[70:390,10:497]

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
        lineout_start = float(OGX_min)
        lineout_end = float(OGx_max)
        red_max = float(OGy_max)
        blue_min = float(OGy_min)









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