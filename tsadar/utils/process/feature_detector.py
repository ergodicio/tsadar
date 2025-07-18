# this will be the feature detector for the first run of the code 
# importing necesary modelues
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import matplotlib.colors as colors
from scipy import ndimage
from skimage import feature
from skimage.util import invert

#defining the main function 
def first_guess(elecData, ionData, config):

    # normalizing data nd accounting for feducials if necesary 
    def data_processing(data, config,wave_type):
        # If data is timed resolved, remove feducials
        if config["other"]["extraoptions"]["spectype"] == "temporal":
                #ROI without feducials for IAW & EPW
            if wave_type == "IAW":
                a,b = 150,850
            elif wave_type == "EPW":
                a, b = 200, 900
        else:
            #ROI cuting omitting known noise for EPW imaging data
            if wave_type =="EPW":
                a= 150
            elif wave_type == "IAW":
                a= 0
            b = 1023
        data = data[a:b,0:1023]
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

        if config["other"]["extraoptions"]["spectype"] == "temporal":

            notch_fliter_start = notch_fliter_start - 200
            notch_filter_end = notch_filter_end - 200
        else: 
            notch_fliter_start = notch_fliter_start - 150
            notch_filter_end = notch_filter_end - 150
        
        epw_blue_box = blur_epw[:notch_fliter_start,:]
        epw_red_box = blur_epw[notch_filter_end:,:]

        return epw_red_box, epw_blue_box
    
    def morphological_opening(img):
        #morphological opening to reduce noice and enhance features
        kernel = np.ones((3, 3), np.uint8)
        erode = cv.erode(img,kernel,iterations = 1) 
        dilate = cv.dilate(erode,kernel, iterations = 1)

        kernel = np.ones((5, 5), np.uint8)
        erode = cv.erode(dilate,kernel,iterations = 1)
        dilate = cv.dilate(erode,kernel, iterations = 1)

        kernel = np.ones((7, 7), np.uint8)
        erode = cv.erode(dilate,kernel,iterations = 1) 
        img = cv.dilate(erode,kernel, iterations = 1)

        return img
    
    #corner detection and filtration
    def data_analysis(img):

        #find corners in eroded image
        corners = cv.goodFeaturesToTrack(img, 100, 0.2, 10)
        corners = np.intp(corners).reshape(-1, 2)

        #filter found corners, only keep corners that have at least one neighboor within the max distance 
        filtered_corners = []
        max_distance = 100
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
        filtered_corners = np.array(filtered_corners)
        #  uncomment the next few lines to display located corners
        # for corner in filtered_corners:
        #     x, y = corner.ravel()
        #     cv.circle(img, (x, y), 5 , 255, 2)
        # cv.imshow('Filtered Corners', img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        #find the max and min x and y coordinates
        min_x = filtered_corners[:, 0].min()
        max_x =  filtered_corners[:, 0].max()
        min_y = filtered_corners[:, 1].min()
        max_y = filtered_corners[:, 1].max()

        # add buffer to the min and max values
        x_start = config["feature_detector"]["buffer"]["lineout_start"]
        x_end = config["feature_detector"]["buffer"]["lineout_end"]
        y_start = config["feature_detector"]["buffer"]["spectral_start"]
        y_end = config["feature_detector"]["buffer"]["spectral_end"]
        min_x -= x_start
        max_x += x_end
        min_y -= y_start
        max_y += y_end

        return min_x, max_x, min_y, max_y

    def iaw_feature_detector():
        ion_img = data_processing(ionData, config,wave_type="IAW")

        ion_img = morphological_opening(ion_img)
        #find and filter corners
        ion_min_x, ion_max_x, ion_min_y, ion_max_y = data_analysis(ion_img)

        lineout_start = ion_min_x
        lineout_end = ion_max_x

        # Calculate iaw_cf, iaw_max, and iaw_min

        if config["other"]["extraoptions"]["spectype"] == "temporal":
            a = 150
            b= 173
            iaw_max = ion_max_y + a
            iaw_min = ion_min_y + a
        else:
            iaw_max = ion_max_y
            iaw_min = ion_min_y
        iaw_cf = (iaw_max - iaw_min) * 0.2
        midpoint = (iaw_max + iaw_min)/2
        iaw_cf_min = midpoint - iaw_cf
        iaw_cf_max = midpoint + iaw_cf

        print("this are the min and maxs for IAW")
        print(f"IAW min x: {lineout_start}, IAW max x: {lineout_end}")
        print(f"IAW max y : {iaw_max}, IAW min y: {iaw_min}")
        print(f"IAW cf min: {iaw_cf_min}, IAW cf max: {iaw_cf_max}")

        return lineout_end,lineout_start,iaw_cf_min,iaw_cf_max,iaw_max,iaw_min
    

    def epw_feature_detector():
        elec_img = data_processing(elecData, config, wave_type="EPW")

        epw_red, epw_blue= notch_filter(elec_img)

        preprocessed_ewp_red = morphological_opening(epw_red)
        preprocessed_ewp_blue = morphological_opening(epw_blue)
        red_min_x, red_max_x, red_min_y, red_max_y = data_analysis(preprocessed_ewp_red)
        blue_min_x, blue_max_x, blue_min_y, blue_max_y = data_analysis(preprocessed_ewp_blue)

        if config["other"]["extraoptions"]["spectype"] == "temporal":
            a = 200
            b = 540
        else: 
            a = 150
            b = 540
        red_max_y = red_max_y + a
        red_min_y = red_min_y + a
        blue_max_y = blue_max_y + b
        blue_min_y = blue_min_y + b

        print("this are the min and maxs for red and blue shifted EPW")
        print(f"Blue min x: {blue_min_x}, Blue max x: {blue_max_x}")
        print(f"Blue min y: {blue_min_y}, Blue max y: {blue_max_y}")
        print(f"Red min x: {red_min_x}, Red max x: {red_max_x}")
        print(f"Red min y: {red_min_y}, Red max y: {red_max_y}")        

        blue_max = blue_max_y
        blue_min = blue_min_y
        lineout_start = red_min_x
        lineout_end = red_max_x
        red_min =  red_min_y
        red_max = red_max_y 

        return lineout_end, lineout_start, blue_min, blue_max, red_min, red_max
    

    if config["feature_detector"]["estimate_lineouts_iaw"] and not config["feature_detector"]["estimate_lineouts_epw"]:
        lineout_end,lineout_start,iaw_cf_min, iaw_cf_max,iaw_max,iaw_min = iaw_feature_detector()
        #Sanity checks
        if lineout_end < lineout_start or iaw_min > iaw_max:
            raise ValueError("Lineout end is less than lineout start or IAW min is greater than max. Detector failed")
        else:
            pass
        return lineout_end, lineout_start, iaw_cf_min, iaw_cf_max, iaw_max, iaw_min

    if config["feature_detector"]["estimate_lineouts_epw"] and not config["feature_detector"]["estimate_lineouts_iaw"]:    

        lineout_end, lineout_start, blue_min, blue_max, red_min, red_max = epw_feature_detector()
        #Sanity checks
        if lineout_end < lineout_start or blue_min > blue_max or red_min > red_max:
            raise ValueError("Lineout end is less than lineout start or  blue/red min is greater than max. Detector failed")
        else:
            pass
        return lineout_end, lineout_start, blue_min, blue_max, red_min, red_max

    if config["feature_detector"]["estimate_lineouts_epw"] and config["feature_detector"]["estimate_lineouts_iaw"]:

        iaw_lineout_end, iaw_lineout_start, iaw_cf_min, iaw_cf_max, iaw_max, iaw_min = iaw_feature_detector()

        epw_lineout_end, epw_lineout_start, blue_min, blue_max, red_min, red_max = epw_feature_detector()
        
        #Sanity checks
        if epw_lineout_end < epw_lineout_start or blue_min > blue_max or red_min > red_max:
            raise ValueError("Lineout end is less than lineout start or  blue/red min is greater than max. Detector failed")
        else:
            pass

        if iaw_lineout_start > iaw_lineout_end or iaw_min > iaw_max:
            raise ValueError("Lineout end is less than lineout start or IAW min is greater than max. Detector failed")
        else:
            pass
        # Calculate ion t0 shift
        if iaw_lineout_start == epw_lineout_start:
            ion_t0_shift = 0
        else:
            ion_t0_shift =  epw_lineout_start - iaw_lineout_start

        return epw_lineout_end, epw_lineout_start, iaw_cf_min, iaw_cf_max, iaw_max, iaw_min, ion_t0_shift, blue_min, blue_max, red_min, red_max
