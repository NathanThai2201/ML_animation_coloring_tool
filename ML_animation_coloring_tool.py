import os
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')   
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

'''
    PREPROCESSING
'''
def get_nearest_palette_color(color_bgr, palette_bgr):
    # Map a BGR color to nearest palette color

    diffs = palette_bgr - np.array(color_bgr, dtype=np.float32)
    dists = np.linalg.norm(diffs, axis=1)
    return palette_bgr[np.argmin(dists)].astype(np.uint8)

def load_palette_colors(palette_path):

    # returns palette colors as BGR 

    palette_img = cv.imread(palette_path, cv.IMREAD_COLOR)
    if palette_img is None:
        raise ValueError("Failed to load palette image")

    pixels = palette_img.reshape(-1, 3)
    colors = np.unique(pixels, axis=0)

    return colors.astype(np.float32)

def contour_features(contours,img):
    area = cv.contourArea(contours)
    if area == 0:
        return None

    perimeter = cv.arcLength(contours, True)

    # Bounding box
    x, y, w, h = cv.boundingRect(contours)
    if h != 0:
        aspect_ratio = w/h
    else: 
        aspect_ratio = 0
    if w * h != 0:
        rectangularity = area /(w * h)
    else: 
        rectangularity = 0

    # Convex hull
    hull = cv.convexHull(contours)
    hull_area = cv.contourArea(hull)
    if hull_area != 0:
        solidity = area / hull_area 
    else: 
        solidity = 0

    convexity = cv.arcLength(hull, True)/perimeter

    # circularity
    if perimeter != 0:
        circularity = 4 * np.pi * area/ (perimeter ** 2) 
    else:
        circularity = 0


    # Hu moments /invariant moments
    moments = cv.moments(contours)
    hu = cv.HuMoments(moments).flatten()
    # log scale hu moments
    hu2 = []

    for h in hu:
        if h != 0:
            value = -np.sign(h) * np.log10(abs(h))
        else:
            value = 0
        hu2.append(value)

    hu = hu2
    


    img_area = img.shape[0] * img.shape[1]

    norm_area = area / img_area
    norm_perimeter = perimeter /np.sqrt(img_area)

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    cx /= img.shape[1]
    cy /= img.shape[0]

    # midpoint distance from x and y
    dx = abs(cx - 0.5)
    dy = abs(cy - 0.5)



    # number of neighbors (TODO)

    return [
        aspect_ratio,
        rectangularity,
        solidity,
        convexity,
        circularity,
        *hu,
        norm_area,
        norm_perimeter,
        cx,
        cy,
        dx,
        dy
    ]

def batch_process(blank_dir, regular_dir, palette_path):

    palette_colors = load_palette_colors(palette_path).astype(np.uint8)

    palette_map = {
        tuple(color): i
        for i, color in enumerate(palette_colors)
    }

    all_data = []
    all_classes = []

    files = sorted(os.listdir(blank_dir))

    for fname in files:
        line_path = os.path.join(blank_dir, fname)
        regular_path = os.path.join(regular_dir, fname)

        if not os.path.exists(regular_path):
            continue

        data, labels, _ = process_image_pair(
            line_path,
            regular_path,
            palette_colors,
            show=False
        )

        if data is None:
            continue

        y_cls = []
        for c in labels:
            y_cls.append(palette_map[tuple(c)])

        all_data.append(data)
        all_classes.append(y_cls)
        print(f"Processed {fname}")

    return (
        np.vstack(all_data),
        np.concatenate(all_classes),
        palette_colors
    )

def process_image_pair(line_path,regular_path, palette_colors, show):

    '''
    THRESHOLDING STAGE
    '''
    img = cv.imread(line_path, cv.IMREAD_UNCHANGED)
    ref_img = cv.imread(regular_path, cv.IMREAD_COLOR)
    alpha_img = cv.imread(regular_path, cv.IMREAD_UNCHANGED)


    if img is None or ref_img is None:
        print(f"Failed to load {line_path}")
        return None, None, None

    if show:
        cv.imshow("original image", ref_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    #get greyscale
    # if img.shape[2] == 4:  # RGBA
    #     print("has alpha")
    #     alpha = img[:, :, 3]
    #     grey = alpha
    # else:
    #     print("does not have alpha")
    #     grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    #edged = cv.Canny(img, 30, 255)
    #cv.imshow("Test greyscale", image1)


    # Thresholding:
    # thresh = cv.adaptiveThreshold(
    #     grey, 255,
    #     cv.ADAPTIVE_THRESH_MEAN_C,
    #     cv.THRESH_BINARY_INV,
    #     7, 2
    # )
    #ret,thresh = cv.threshold(grey,10,255,cv.THRESH_BINARY)
    ret, thresh = cv.threshold(grey, 0, 255, cv.THRESH_BINARY)
    # #line cleanup:
    # kernel = np.ones((3,3), np.uint8)
    # thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)



    #print(alpha_img)
    #print(ref_img)

    titles = ['original image','greyscale','threshold']
    images = [img, grey, thresh]
    if show:

        cv.imshow("threshold", thresh)

        cv.waitKey(0)
        cv.destroyAllWindows()

    
    


    '''
    CONTOURING STAGE
    '''
    #isolating contours with black
    min_YCrCb = np.array([0,0,0],np.uint8) # Create a lower bound HSV
    max_YCrCb = np.array([200,200,250],np.uint8) # Create an upper bound HSV


    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    skinRegion = cv.inRange(hsv,min_YCrCb,max_YCrCb) # Create a mask with boundaries



    # add white border
    skinRegion[0, :] = 255
    skinRegion[-1, :] = 255
    skinRegion[:, 0] = 255
    skinRegion[:, -1] = 255

    # remove border where original image is transparent
    if alpha_img.ndim == 3 and alpha_img.shape[2] == 4:
        alpha = alpha_img[:, :, 3]
        alpha_mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
    else:
        alpha_mask = np.full(skinRegion.shape, 255, dtype=np.uint8)

    skinRegion = cv.bitwise_and(skinRegion, alpha_mask)




    if show:
        cv.imshow("skin region for black lines",skinRegion)
        cv.waitKey(0)
        cv.destroyAllWindows()
    contours, hierarchy = cv.findContours(skinRegion, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE) # Find the contour on the skin detection
   



    ''' 
        CONTOUR DRAW METHODS
    '''
   


    # separate by area
    # for i, c in enumerate(contours): # Draw the contour on the source frame
    #     area = cv.contourArea(c)
    #     if area > 100:
    #         color = tuple(np.random.randint(0, 256, size=3).tolist())
    #         cv.drawContours(img, contours, i, color, -1) 
    # cv.imshow('skin region contour approximation', img)
    # cv.waitKey(0)
    # cv.imwrite('contours_none_image1.jpg', img)
    # cv.destroyAllWindows()


    #contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
                   
    # remove first contour, ie silouhette
    #print("A",hierarchy)
    contours2 = []

    for c in range(len(contours)):
         if hierarchy[0][c][3] != -1:
            contours2.append(contours[c])

    image_blank = np.zeros_like(img)
    

    for c in range(len(contours2)):
            color = tuple(np.random.randint(0, 256, size=3).tolist())
            cv.drawContours(image_blank, [contours2[c]], -1, color, -1)

    #cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
                    
    # view results
    if show:
        print("number of contours found",len(contours))
        cv.imshow('None contour approximation', image_blank)
        cv.waitKey(0)
        cv.imwrite('contours_none_image1.jpg', image_blank)
        cv.destroyAllWindows()





    '''
        DATA AND LABELS
    '''

    labels = []

    for c in contours2:
        # Create a mask for each contour
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv.drawContours(mask, [c], -1, 255, -1)

        # get pixels from the original reference image
        pixels = ref_img[mask == 255]

        if len(pixels) == 0:
            labels.append((0, 0, 0)) # default to black
            continue


        # Find most common color
        pixels_reshaped = pixels.reshape(-1, 3)
        colors, counts = np.unique(pixels_reshaped, axis=0, return_counts=True)
        dominant_color = colors[np.argmax(counts)]
        snapped = get_nearest_palette_color(dominant_color, palette_colors)
        labels.append(tuple(int(c) for c in snapped))



    if show:
        for c, color in zip(contours2, labels):
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(ref_img, (x,y), (x+w,y+h), color, 2)

        cv.imshow("Labeled regions", ref_img)
        cv.waitKey(0)
        cv.destroyAllWindows()



    '''
    DATA AND LABELS
    '''

    features = []
    valid_labels = []
    valid_contours = []

    for cont, lb in zip(contours2, labels):
        f = contour_features(cont,img)
        if f is None:
            continue
        features.append(f)
        valid_labels.append(lb)
        valid_contours.append(cont)


    if len(features) == 0:
        return None, None, None

    data = np.array(features, dtype=np.float32)
    labels = np.array(valid_labels)

    return data, labels, valid_contours


'''
    LEARNING
'''

def learn_classifier(data, y_cls):
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        y_cls,
        test_size=0.2,
        random_state=42,
        stratify=y_cls
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    print("Color prediction accuracy:", acc)

    return x_test, y_test, y_pred, clf


'''
    RECONSTRUCTION
'''
def extract_contours_and_features(line_path, regular_path, show):
    '''
    THRESHOLDING STAGE
    '''
    img = cv.imread(line_path, cv.IMREAD_UNCHANGED)
    ref_img = cv.imread(regular_path, cv.IMREAD_COLOR)
    alpha_img = cv.imread(regular_path, cv.IMREAD_UNCHANGED)


    if img is None or ref_img is None:
        print(f"Failed to load {line_path}")
        return None, None

    if show:
        cv.imshow("original image", ref_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    ret, thresh = cv.threshold(grey, 0, 255, cv.THRESH_BINARY)

    titles = ['original image','greyscale','threshold']
    images = [img, grey, thresh]
    # if show:

    #     cv.imshow("threshold", thresh)

    #     cv.waitKey(0)
    #     cv.destroyAllWindows()

    
    
    '''
    CONTOURING STAGE
    '''
    #isolating contours with black
    min_YCrCb = np.array([0,0,0],np.uint8) # Create a lower bound HSV
    max_YCrCb = np.array([200,200,250],np.uint8) # Create an upper bound HSV


    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    skinRegion = cv.inRange(hsv,min_YCrCb,max_YCrCb) # Create a mask with boundaries



    # add white border
    skinRegion[0, :] = 255
    skinRegion[-1, :] = 255
    skinRegion[:, 0] = 255
    skinRegion[:, -1] = 255

    # remove border where original image is transparent
    if alpha_img.ndim == 3 and alpha_img.shape[2] == 4:
        alpha = alpha_img[:, :, 3]
        alpha_mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
    else:
        alpha_mask = np.full(skinRegion.shape, 255, dtype=np.uint8)

    skinRegion = cv.bitwise_and(skinRegion, alpha_mask)



    contours, hierarchy = cv.findContours(skinRegion, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

    '''
      FEATURES 
    '''
    contours2 = [
        contours[i]
        for i in range(len(contours))
        if hierarchy[0][i][3] != -1
    ]

    features = []
    valid_contours = []

    for c in contours2:
        f = contour_features(c, img)
        if f is not None:
            features.append(f)
            valid_contours.append(c)

    if len(features) == 0:
        return None, None, None

    return np.array(features, np.float32), valid_contours, img

def reconstruct_blank_image(line_path, regular_path, clf, scaler, palette_colors, show=True):

    data, contours, img = extract_contours_and_features(line_path,regular_path,show)

    if data is None:
        return None

    #data_scaled = scaler.transform(data)
    cls_pred = clf.predict(data)

    reconstructed = img.copy()

    for c, cls in zip(contours, cls_pred):
        color = palette_colors[cls]
        cv.drawContours( reconstructed, [c], -1, tuple(int(v) for v in color), -1)

    if show:
        cv.imshow("Reconstructed Image", reconstructed)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return reconstructed


if __name__ == "__main__":

    from sklearn.preprocessing import StandardScaler

    '''
        Training
    '''
    data, y_cls, palette_colors = batch_process(
        blank_dir="data/lines",
        regular_dir="data/regular",
        palette_path="palette.png"
    )


    print("Total samples:", len(data))

    # Scale features (currently not in use)
    scaler = StandardScaler()
    #data_scaled = scaler.fit_transform(data)

    # Train random forest classifier
    x_test, y_test, y_pred, clf = learn_classifier(data, y_cls)



    '''
        Reconstruction
    '''
    # Test cases within dataset
    for i in range(4):
        random_idx = np.random.randint(10,70)
        reconstructed = reconstruct_blank_image(
            line_path="data/lines/Timeline 1_00"+str(random_idx)+".png",
            regular_path = "data/regular/Timeline 1_00"+str(random_idx)+".png",
            clf=clf,
            scaler=scaler,
            palette_colors=palette_colors,
            show=True
        )





def opencv_startup():

    # # create a numpy array filled with zeros to use as a blank image
    # image = np.zeros ( (512, 512, 3), np.uint8)
    # # draw a green line on the image
    # cv. line(image, (0, 0), (511, 511), (0, 255, 0), 5)
    # # draw a red rectangle on the image
    # cv. rectangle(image, (384, 0), (510, 128), (0, 0, 255), 3)
    # # draw a blue circle on the image
    # cv. circle(image, (447, 63), 63, (255, 0, 0), -1)
    # # display the image
    # cv. imshow(' Image', image)
    # # wait for a key press and then close the window

    # cv.waitKey(0)
    # cv.destroyAllWindows()
    pass
