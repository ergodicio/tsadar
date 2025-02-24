# this will be the feature detector for the first run of the code 
# importing necesary modelues
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image
import matplotlib.colors as colors
import tempfile, os
from scipy import ndimage
from skimage import feature
from skimage.util import invert

#defining the main function 
def first_guess(elecData, ionData, config):

    # normalizing data nd accounting for feducials if necesary 
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

        #erode image 
        kernel = np.ones((2, 2), np.uint8)
        eroded = cv.erode(img, kernel, iterations=2)
        #plt.imshow(eroded)
        #plt.show()

        #find corners in eroded image
        corners = cv.goodFeaturesToTrack(eroded, 50, 0.01, 10)
        corners = np.int0(corners).reshape(-1, 2)

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
            x, y = corner.ravel()
            cv.circle(eroded, (x, y), 5 , 255, 2)

        # Display the result
        #cv.imshow('Filtered Corners', eroded)
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
    

    
    if config ["data"]["estimate_lineouts_iaw"]:
        # if the image is time resolved, select ROI without feducials
        #process data
        ion_img = data_processing(ionData, config)
        #find and filter corners
        ion_min_x, ion_max_x, ion_min_y, ion_max_y = data_analysis(ion_img)

        lineout_start = ion_min_x
        lineout_end = ion_max_x

        # Calculate iaw_cf, iaw_max, and iaw_min

        if config["data"]["background"]["type"] == "pixel":
            a = 150
            b= 173
            iaw_max = ion_max_y + a
            iaw_min = ion_min_y + a
            print(f"IAW max: {iaw_max}, IAW min: {iaw_min}")
            # multiplying values by 5 since magE = 5ps / px for temporal data 
        else:
            iaw_max = ion_max_y
            iaw_min = ion_min_y
        iaw_cf = (iaw_max + iaw_min)/2
        print(f"IAW cf: {iaw_cf}")

        return lineout_end,lineout_start,iaw_cf,iaw_max,iaw_min
    
    if config["data"]["estimate_lineouts_epw"]:

        elec_img = data_processing(elecData, config)

        epw_red, epw_blue= notch_filter(elec_img)


        red_min_x, red_max_x, red_min_y, red_max_y = epw_corner_processing(epw_red)

        blue_min_x, blue_max_x, blue_min_y, blue_max_y = epw_corner_processing(epw_blue)

        if config["data"]["background"]["type"] == "pixel":
            #ROI without feducials
            red_max_y = 824 - red_max_y
            red_min_y = 824 - red_min_y 
            blue_max_y = 484 - blue_max_y 
            blue_min_y = 484 - blue_min_y
        else:
            red_max_y = 1024 - red_max_y
            red_min_y = 1024 - red_min_y
            blue_max_y = 684 - blue_max_y
            blue_min_y =  684 - blue_min_y
        
        blue_max = blue_max_y 
        blue_min = blue_min_y 
        lineout_start = red_min_x
        lineout_end = red_max_x
        red_min =  red_min_y
        red_max = red_max_y 
        print("blue max", blue_max)
        print("blue min", blue_min)
        print("red max", red_max)
        print("red min", red_min)

        return lineout_end, lineout_start, blue_min, blue_max, red_min, red_max
    
    if config["data"]["estimate_lineouts_epw"] and config["data"]["estimate_lineouts_iaw"]:

        lineout_start = red_min_x
        lineout_end = red_max_x

        return lineout_start, lineout_end