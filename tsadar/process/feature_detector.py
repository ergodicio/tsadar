# this will be the feature detector for the first run of the code 
# importing necesary modelues
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import matplotlib.colors as colors
import tempfile, os
from skimage import feature

#defining the main function 
def first_guess(elecData, ionData, all_axes, config):

    #if config["other"]["extraoptions"]["load_ion_spec"]: # ploting the iaw spec w/out lineouts
    
    if config ["data"]["estimate_lineouts_iaw"]:
        # if the image is time resolved, select ROI without feducials
        if config["data"]["background"]["type"] == "pixel":
            #ROI without feducials
            a = 150
            b= 173
            feducials_iaw = ionData[150:850,0:1023]

            # Normalize the values to the range [0, 255]
            min_val = feducials_iaw.min()
            max_val = feducials_iaw.max()

            # Perform normalization
            feducials_iaw_normalized = ((feducials_iaw - min_val) / (max_val - min_val)) * 255.0

            # Convert the image to uint8
            img = feducials_iaw_normalized.astype(np.uint8)

        else:

            min_val = ionData.min()
            max_val = ionData.max()
            ionData_normalized = ((ionData - min_val)/(max_val - min_val))* 255.0
            img = ionData_normalized.astype(np.uint8)
        

            # Ensure the selected region is in uint8 format
            #feducials_iaw = (feducials_iaw*255).astype(np.uint8)
            #img= feducials_iaw.astype(np.float32)

                            # Plot both ionData and feducials_iaw side by side
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Plot ionData
            axes[0].imshow(ionData, cmap='inferno')
            axes[0].set_title("ionData")
            axes[0].axis('off')  # Hide axes for cleaner view

            # Plot feducials_iaw
            axes[1].imshow(feducials_iaw_normalized, cmap='inferno')
            axes[1].set_title("feducials_iaw")
            axes[1].axis('off')  # Hide axes for cleaner view

            # Plot img which is the uint8 version of feducials_iaw_normalized
            axes[1].imshow(img, cmap='inferno')
            axes[1].set_title("feducials_iaw")
            axes[1].axis('off')  # Hide axes for cleaner view

            plt.tight_layout()  # Adjust layout to avoid overlap
            plt.show()
             # Find corners in the eroded image
            # Perform erosion on the binary image

            kernel = np.ones((2, 2), np.uint8)
            eroded = cv.erode(img, kernel, iterations=2)

            # Show the eroded image
            plt.imshow(eroded)
            plt.title("Eroded Image")
            plt.axis('off')
            plt.show()

            corners = cv.goodFeaturesToTrack(eroded, 50, 0.01, 10)
            corners = np.int0(corners).reshape(-1,2)

            #filter conrners: only keep corners that have at least one other corner within the max distance
            filtered_corners = []
            max_distance = 50
            for i, corner in enumerate(corners):
                has_neighboor = False
                for j, other_corner in enumerate(corners):
                    if i == j:
                        continue # skip same corner
                    #Euclidean distance between current corner and other corners
                    distance = np.linalg.norm(corner-other_corner)
                    if distance <= max_distance:
                        has_neighboor = True
                        break
                if has_neighboor:
                    filtered_corners.append(corner)
            # Convert filtered corners back to the required format
            filtered_corners = np.array(filtered_corners)
            # Find the max and min x and y coordinates
            if filtered_corners.size > 0:
                min_x = np.min(filtered_corners[:, 0])  # Minimum x-coordinate
                max_x = np.max(filtered_corners[:, 0])  # Maximum x-coordinate
                min_y = np.min(filtered_corners[:, 1])  # Minimum y-coordinate
                max_y = np.max(filtered_corners[:, 1])  # Maximum y-coordinate

                print(f"Min x: {min_x}, Max x: {max_x}")
                print(f"Min y: {min_y}, Max y: {max_y}")
            else:
                print("No filtered corners found.")
            
            
            # Add a margin of erron

            min_x -= 20
            max_x += 20
            min_y -= 20
            max_y += 20

            lineout_start = min_x
            lineout_end = max_x

            if config["data"]["background"]["type"] == "pixel":

                # mapping found points to the original image
                iaw_max = max_y + a
                iaw_min =  min_y - b

            else:
                iaw_max = max_y
                iaw_min =  min_y
            

            iaw_cf = (iaw_max+iaw_min)/2

        return lineout_end,lineout_start,iaw_cf,iaw_max,iaw_min

    if config ["data"]["estimate_lineouts_epw"]:
        # if the image is time resolved, select ROI without feducials
        if config["data"]["background"]["type"] == "pixel":
            #ROI without feducials
            a = 150
            b= 173
            feducials_epw = elecData[150:850,0:1023]

            # Normalize the values to the range [0, 255]
            min_val = feducials_epw.min()
            max_val = feducials_epw.max()

            # Perform normalization
            feducials_epw_normalized = ((feducials_epw - min_val) / (max_val - min_val)) * 255.0

            # Convert the image to uint8
            img = feducials_epw_normalized.astype(np.uint8)

        else:

            min_val = elecData.min()
            max_val = elecData.max()
            elecData_normalized = ((elecData - min_val)/(max_val - min_val))* 255.0
            img = elecData_normalized.astype(np.uint8)
        

            # Ensure the selected region is in uint8 format
            #feducials_iaw = (feducials_iaw*255).astype(np.uint8)
            #img= feducials_iaw.astype(np.float32)

                            # Plot both ionData and feducials_iaw side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot ionData
        axes[0].imshow(elecData, cmap='inferno')
        axes[0].set_title("elecData")
        axes[0].axis('off')  # Hide axes for cleaner view

        # Plot feducials_iaw
        axes[1].imshow(feducials_epw_normalized, cmap='inferno')
        axes[1].set_title("feducials_epw")
        axes[1].axis('off')  # Hide axes for cleaner view

        # Plot img which is the uint8 version of feducials_iaw_normalized
        axes[1].imshow(img, cmap='inferno')
        axes[1].set_title("feducials_epw_img")
        axes[1].axis('off')  # Hide axes for cleaner view

        plt.tight_layout()  # Adjust layout to avoid overlap
        plt.show()

        # Draw filtered corners on the image
        for corner in filtered_corners:
            x, y = corner.ravel()
            cv.circle(eroded, (x, y), 5 , 255, 2)

        # Display the result
        cv.imshow('Filtered Corners', img)
        cv.waitKey(0)
        cv.destroyAllWindows()


        # List to store the corner coordinates
        corner_list = []
        for i in corners:
            x, y = i.ravel()
            cv.circle(eroded, (x, y), 5,255, 2)  # Mark the corners on the image
            corner_list.append((x, y))

        # Show the image with marked corners
        plt.imshow(eroded, cmap='gray')
        plt.title("Image with Corners")
        plt.axis('off')
        plt.show()

        # Find the smallest and largest x and y values from the corner coordinates
        corner_list.sort()  # Sort by x-coordinate
        min_x = corner_list[0][0]
        max_x = corner_list[-1][0]

        corner_list.sort(key=lambda point: point[1])  # Sort by y-coordinate
        min_y = corner_list[0][1]
        max_y = corner_list[-1][1]

        # Calculate the average y-coordinate (iaw_cf)
        iaw_cf = (min_y + max_y) / 2

        # Output the results
        print("min x coordinate:", min_x)
        print("max x coordinate:", max_x)
        print("min y coordinate:", min_y)
        print("max y coordinate:", max_y)
        print("Average y-coordinate (iaw_cf):", iaw_cf)



        if iaw_cf == 0:
            # Ensure the selected region is in uint8 format
            img = feducials_iaw.astype(np.uint8)
            X, Y = np.meshgrid(np.arange(feducials_iaw.shape[1]), np.arange(feducials_iaw.shape[0]))

            # Create a figure to plot
            fig, ax = plt.subplots()

            # Plotting using pcolormesh
            mesh = ax.pcolormesh(X, Y, feducials_iaw, cmap="inferno", 
                                norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                        vmin=np.amin(feducials_iaw), vmax=np.amax(feducials_iaw)),
                                shading="auto")  # Use "flat" or "auto" for shading
            #fig.colorbar(mesh, label="Intensity")
            #plt.show()


            # Apply adaptive thresholding to get a binary image
            bw_img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
            # Show the binary (black & white) image
            plt.imshow(bw_img, cmap='gray')
            plt.title("Binary Image")
            plt.axis('off')
            plt.show()

            # Perform erosion on the binary image
            kernel = np.ones((2, 2), np.uint8)
            eroded = cv.erode(bw_img, kernel, iterations=2)

            # Show the eroded image
            plt.imshow(eroded)
            plt.title("Eroded Image")
            plt.axis('off')
            plt.show()
            # Find corners in the eroded image
            corners = cv.goodFeaturesToTrack(eroded, 10, 0.01, 10)
            corners = np.int0(corners)

            # List to store the corner coordinates
            corner_list = []
            for i in corners:
                x, y = i.ravel()
                cv.circle(img, (x, y), 3, 255, -1)  # Mark the corners on the image
                corner_list.append((x, y))

            # Show the image with marked corners
            plt.imshow(img, cmap='gray')
            plt.title("Image with Corners")
            plt.axis('off')
            plt.show()

            # Find the smallest and largest x and y values from the corner coordinates
            corner_list.sort()  # Sort by x-coordinate
            min_x = corner_list[0][0]
            max_x = corner_list[-1][0]

            corner_list.sort(key=lambda point: point[1])  # Sort by y-coordinate
            min_y = corner_list[0][1]
            max_y = corner_list[-1][1]

            # Calculate the average y-coordinate (iaw_cf)
            iaw_cf = (min_y + max_y) / 2

            # Output the results
            print("min x coordinate:", min_x)
            print("max x coordinate:", max_x)
            print("min y coordinate:", min_y)
            print("max y coordinate:", max_y)
            print("Average y-coordinate (iaw_cf):", iaw_cf)

            # Create a figure to plot
            
        #if the data is not time resolved, image size remains 1024x1024
        #else:
            img = ionData.astype(np.uint8)
            X, Y = np.meshgrid(all_axes["iaw_x"], all_axes["iaw_y"])
            fig, ax = plt.subplots()
            ax.axis('on')
            cb = ax.pcolormesh(
            X,
            Y,
            ionData,
            cmap="turbo_r",
            #norm=colors.SymLogNorm( linthresh = 0.03, linscale = 0.03, vmin =0, vmax= np.amax(ionData)),
            norm=colors.SymLogNorm( linthresh = 0.03, linscale = 0.03,vmin=np.amin(ionData), vmax= np.amax(ionData)),
            shading= "auto",

            )               
            X, Y = np.meshgrid(all_axes["iaw_x"], all_axes["iaw_y"])
            ionData,
            fig1, ax1 = plt.subplots()
            ax1.axis('on')
            ax1.pcolormesh(
                X,
                Y,
                ionData,
                cmap="turbo_r",
                norm=colors.SymLogNorm( linthresh = 0.03, linscale = 0.03,vmin=np.amin(ionData), vmax= np.amax(ionData)),
                #vmin=0,
                #vmax=0.05*np.amax(ionData),
                #shading = "auto",
            )
        
            """plt.figure(figsize=(10, 8))

            # Displaying the extracted portion using imshow
            plt.imshow(feducials_iaw, cmap="turbo_r", origin="lower", aspect="auto", 
        norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                vmin=np.amin(feducials_iaw), vmax=np.amax(feducials_iaw)))
            #fig1.show()"""
            # Add color bar to the plot
            #fig1.colorbar(mesh, ax=ax1)
            #plt.imshow(img_iaw)
            plt.show()
            fig1_path = os.path.join(tdir,'temp_iaw.png') #path for iaw plot 
            fig1.savefig(fig1_path , bbox_inches='tight') #save in temp dir
            #openinn iaw and geting roi
            iaw1 = cv.imread(fig1_path)
            #fig1.show()
            #plt.plot(ionData)
            #plt.show()




            # assuming these are the right coordinates for feducials
            a = 70
            b = 390
            c = 10
            d = 487
            
            img_iaw = ionData[a:b,c:d]# to account for time feducials
    else:

        img_iaw = ionData.shape[:2] # if there are no feducials
    img_iaw = ionData.shape[:2]
    # convert the img to grayscale (migth delete later if the image is ploter in gray scale)
    #gray_iaw =cv.cvtColor(img_iaw,cv.COLOR_BGR2GRAY)
    max_gray_value = np.max(img_iaw) #find the highest value from the gray scale image 
    threshold_value = 0.5 * max_gray_value #set the threshold to 50% of the maximum value 
    ret,bw_iaw = cv.threshold(img_iaw,threshold_value, 255, cv.THRESH_BINARY)  # apply thresholdin0g
    kernel = np.ones((2,2),np.uint8 ) #kernel for eroding

    eroded = cv.erode(bw_iaw, kernel, iterations = 2) #bw_iaw holds the eroded image
    eroded = eroded.astype(np.uint8) 
    corners_iaw = cv.goodFeaturesToTrack(eroded,50,0.01,10) # eroded is now single channel
    corners_iaw = np.int0(corners_iaw)

    #list for corners coordinates
    corner_list_iaw = []
    for i in corners_iaw:
        x,y = i.ravel()
        cv.circle(img_iaw,(x,y),2,255,-1)
        corner_list_iaw.append((x,y))

    #Finding the min and max x and y points of the feature
    corner_list_iaw.sort()  # Sort in-place by x-coordinate (default behavior)
    x_min = corner_list_iaw[0][0]  # Extract only the x-value
    x_max = corner_list_iaw[-1][0]
    corner_list_iaw.sort(key=lambda point: point[1])  # Sort in-place by y-coordinate
    y_min = corner_list_iaw[0][1]  # Extract only the y-value
    y_max = corner_list_iaw[-1][1] # Extract only the y-value

    #mapoping found points to the original image
    OGX_min = x_min + c
    OGx_max = x_max + d
    OGy_min = y_min + b
    OGy_max = y_max + a

    #maping findings to their corresponding parameters
    lineout_start = OGX_min
    lineout_end = OGx_max
    iaw_max = OGy_max
    iaw_min = OGy_min


    #argin of error
    x_min -= 20
    x_max += 20
    y_min -= 20
    y_max += 20

    iaw_cf = (y_min + y_max)/2 

    return lineout_end,lineout_start,iaw_cf,x_min,x_max,y_max,y_min


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
    return red_max,blue_min,lineout_end,lineout_start,iaw_max,iaw_min, iaw_cf,x_min,x_max,y_max,y_min

#make the function return the outputs, theese will be used later by another function 


def get_lambda(config,img_iaw,img_epw,OGy_min,OGy_max,x_min,x_max,y_max,y_min):
 

#if the get boolean for calculating the lambda fro iaw is true do the following 

    if config["data"]["estimate_lambda_iaw"]:
        iaw_box = img_iaw[y_min:y_max,x_min:x_max] # use min and max x.y values found with detector to selelct the ROI
        edges = feature.canny(iaw_box,sigma=3) #canny edge detector
        indices = np.where(edges != [0]) 
        coordinates = list(zip(indices[1],indices[0])) #make a list of coordinates with the found points
        coordinates.sort(key=lambda point: point[1]) #sort by y value 
        edge_y_min = coordinates[0][1]
        edge_y_max = coordinates[-1][1] 
        iaw_cf = (edge_y_max+ edge_y_min)/2
        #dictyionaries for storing upper and lower half coordinates 
        upper_half = {} 
        lower_half ={}

        for x, y in coordinates:
            if y < iaw_cf:
                if x not in upper_half:
                    upper_half[x] = []
                upper_half[x].append(y)
            else:
                if x not in lower_half:
                    lower_half[x]= []
                lower_half[x].append(y)

        results_upper = {}
        upper_midpoint = {}

        for x, y_values in upper_half.items():
            min_y = min(y_values)
            max_y = max(y_values)
            mid_point = (min_y + max_y) / 2
            upper_midpoint[x] = mid_point
            results_upper[x] = {"min_y": min_y, "max_y": max_y, "mid_point": mid_point}
        
        results_lower = {}
        lower_midpoint ={}
        # Find min and max y-values for each x-group in the lower half and calculate midpoints
        for x, y_values in lower_half.items():
            min_y = min(y_values)
            max_y = max(y_values)
            mid_point = (min_y + max_y) / 2
            lower_midpoint[x] = mid_point
            results_lower[x] = {"min_y": min_y, "max_y": max_y, "mid_point": mid_point}

        






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
        #fig1_path = os.path.join(tdir,'temp_iaw.png') #path for iaw plot 
        #fig1.savefig(fig1_path , bbox_inches='tight') #save in temp dir
        #openinn iaw and geting roi
        #iaw1 = cv.imread(fig1_path)

        # using fig without a temp directory 
        
        img_iaw = ionData[70:390,10:497]# to account for time feducials

        #img_iaw = ionData.shape[:2] # if there are no feducials

        # convert the img to grayscale (migth delete later if the image is ploter in gray scale)
        #gray_iaw =cv.cvtColor(img_iaw,cv.COLOR_BGR2GRAY)
        max_gray_value = np.max(img_iaw) #find the highest value from the gray scale image 
        threshold_value = 0.5 * max_gray_value #set the threshold to 50% of the maximum value 
        ret,bw_iaw = cv.threshold(img_iaw,threshold_value, 255, cv.THRESH_BINARY)  # apply thresholdin0g
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
        iaw_min = OGy_min
        iaw_cf = (OGy_min + OGy_max)/2






    x_start = config["data"]["lineouts"]["start"]
    x_end = config["data"]["lineouts"]["end"]
    skips = config["data"]["lineouts"]["skip"]
    delta_x = x2 - x1 
    #lineout_size = delta_x / config["data"]["lineouts"]["skip"]
    #for lineout_size in range(delta_x):
    
    chuncks = [i for i in range (x_start,x_end,skips)]
    first_chuck = chuncks[0],chuncks[1]
    first_chunk_epw = img_epw[chuncks[0]:chuncks[1],OGy_min:OGy_max]
    first_chunck_iaw = img_iaw[OGy_max:OGy_min,chuncks[0],chuncks[1]]

    if config["data"]["estimate_lambda"]:
        inside_box = ionData
        for i in range(chuncks):
            chunk_iaw = img_iaw[OGy_max:OGy_min,chuncks[i],chuncks[i+1]]










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