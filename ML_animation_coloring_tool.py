import os
import cv2 as cv
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')   
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from joblib import dump, load



'''
    PREPROCESSING AND HELPERS
'''

def dominant_palette_color(pixels, palette_colors):
    if pixels.size == 0:
        return np.array([0, 0, 0], dtype=np.uint8)

    # Strip alpha if present
    if pixels.shape[-1] == 4:
        pixels = pixels[:, :3]

    pixels = pixels.reshape(-1, 3)

    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    dominant = colors[np.argmax(counts)]

    return get_nearest_palette_color(dominant, palette_colors)

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

    # assume clean no antialiasing
    pixels = palette_img.reshape(-1, 3)
    colors = np.unique(pixels, axis=0)

    # include black as an exception case
    # np.append(colors, [0,0,0])
        

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

def filter_bad_contours(contours, hierarchy, img, ref_img, palette_colors):
    h, w = img.shape[:2]

    filtered_contours = []

    DARK_THRESH = np.array([130, 130, 130], dtype=np.float32)

    for i, contour in enumerate(contours):
        # remove silhouette
        if hierarchy[0][i][3] == -1:
            continue

        # mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv.drawContours(mask, [contour], -1, 255, -1)

        img_pixels = img[mask == 255]

        if img_pixels.size == 0:
            continue

        # check contour for mean color over line art image
        mean_color = img_pixels[:, :3].mean(axis=0) 

        if np.all(mean_color < DARK_THRESH):
            continue

        # snapped_img = dominant_palette_color(img_pixels, palette_colors)

        # # old skip contours, getting the domininant collor of the lineart image if is not blank
        # if np.all(snapped_img == 0):
        #     continue

        # must have valid features
        if contour_features(contour, img) is None:
            continue

     
        filtered_contours.append(contour)
    
    # debug_img = np.zeros_like(img)
    # for c in filtered_contours:
    #     color = tuple(np.random.randint(0, 256, 3).tolist())
    #     cv.drawContours(debug_img, [c], -1, color, -1)

    # cv.imshow("Final Filtered Contours", debug_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    return filtered_contours

def batch_process(blank_dir, regular_dir, palette_path):

    palette_colors = load_palette_colors(palette_path).astype(np.uint8)

    palette_map = {
        tuple(color): i
        for i, color in enumerate(palette_colors)
    }

    all_data = []
    all_classes = []

    files = sorted(os.listdir(blank_dir))

    # loop through files
    for f in files:
        line_path = os.path.join(blank_dir, f)
        regular_path = os.path.join(regular_dir, f)

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
        print(f"Processed {f}")

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
    
    contours2 = filter_bad_contours(
        contours,
        hierarchy,
        img,
        ref_img,
        palette_colors
    )
    
    labels = []
    h, w = img.shape[:2]

    for contour in contours2:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv.drawContours(mask, [contour], -1, 255, -1)

        ref_pixels = ref_img[mask == 255]
        snapped_ref = dominant_palette_color(ref_pixels, palette_colors)

        labels.append(tuple(map(int, snapped_ref)))



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

def learn_classifier(data, y_cls,n_estimators, max_features):
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        y_cls,
        test_size=0.2,
        random_state=42,
        stratify=y_cls
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        n_jobs=-1,
        random_state=42
    )

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(f"n_estimators: {n_estimators}, max_features: {max_features}")
    acc = accuracy_score(y_test, y_pred)
    print("Color prediction accuracy:", acc)

    return x_test, y_test, y_pred, clf, acc


'''
    RECONSTRUCTION
'''

def extract_contours_and_features(line_path, regular_path, palette_colors, show):

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

    contours2 = filter_bad_contours(
        contours,
        hierarchy,
        img,
        ref_img,
        palette_colors
    )


    features = []
    valid_contours = []

    for c in contours2:
        f = contour_features(c, img)
        if f is not None:
            features.append(f)
            valid_contours.append(c)

    if len(features) == 0:
        return None, None, None

    return np.array(features, np.float32), valid_contours, img, ref_img

def reconstruct_blank_image(line_path, regular_path, clf, scaler, palette_colors, show=True):

    data, contours, img, ref_img = extract_contours_and_features(
        line_path,
        regular_path,
        palette_colors,
        show
    )


    if data is None:
        return None

    #data_scaled = scaler.transform(data)
    cls_pred = clf.predict(data)

    #reconstructed = np.full_like(img,255)
    reconstructed = np.zeros_like(img)

    for c, cls in zip(contours, cls_pred):
        color = palette_colors[cls]


        color_ints = []
        for v in color:
            color_ints.append(int(v))

        draw_color = tuple(color_ints)

        cv.drawContours( reconstructed, [c], contourIdx = -1, lineType=cv.LINE_8, color = draw_color, thickness=-1)
        # cv.drawContours( reconstructed, [c], contourIdx = -1, lineType=cv.LINE_AA, color = draw_color, thickness=2)

    checkpoint1 = reconstructed.copy()
    # both img and reconstructed have 4 channels
    # print(reconstructed.shape, reconstructed)
    # print(img.shape, img)
    # alpha manipulation for line art addition

    # ensure float type for image values
    img = img.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)


    #simplified floodfill
    # w,h,_ = img.shape
    # for i in range(0,w-1,1):
    #     for j in range(0,h-1,1):
    #         pixel = reconstructed[i,j]
    #         if (pixel[:3] == [0,0,0]).all() or (pixel[:3] == [255,255,255]).all(): # if pixel is black or white
    #             #print("detected black",pixel[:3])
    #             for s_pixel in [reconstructed[i,j+1], # udlr
    #                             reconstructed[i+1,j],
    #                             reconstructed[i-1,j],
    #                             reconstructed[i,j-1],
    #                             reconstructed[i-1,j-1], #diags
    #                             reconstructed[i+1,j+1], 
    #                              reconstructed[i+1,j-1], 
    #                             reconstructed[i-1,j+1]]:
    #                 #print("-",s_pixel)
    #                 if s_pixel[:3] in palette_colors:        
    #                     pixel[:3] = s_pixel[:3]
    #                     break
    
    ff_its = 4 # proportional to blur strength
    for it in range(ff_its):
        img_rgb = reconstructed[:, :,:3]

        #mask where black and white
        mask = (
            np.all(img_rgb == [0,0,0], axis = 2) | np.all(img_rgb == [255,255,255],  axis = 2)
        )

        # pad image to safely get neighbours
        img_padded = np.pad(img_rgb, ((1,1),(1,1),(0,0)), mode = 'edge')


        neighbours = np.stack([
            img_padded[1:-1, 2:,   :],  # right
            img_padded[2:,   1:-1, :],  # down
            img_padded[:-2,  1:-1, :],  # up
            img_padded[1:-1, :-2,  :],  # left
            img_padded[2:,   2:,   :],  # down right
            img_padded[:-2, :-2,   :],  # up left
            img_padded[2:,   :-2,  :],  # down left
            img_padded[:-2,  2:,   :],  # up right
        ], axis=0)

        palette = palette_colors.astype(np.float32)

        #print("shapes:",neighbours.shape, palette.shape)
        # reshape for broadcasting
        neighbours_exp = neighbours[..., None, :]              # (8, h, w, 3) -> (8, h, w, 1, 3 )
        palette_exp   = palette[None, None, None, :, :]     # (p,3) -> (1, 1, 1, p, 3)

        matches = np.all(neighbours_exp == palette_exp, axis=-1)  # (8,h,w,p)

        valid_neighbor = np.any(matches, axis=-1)  # (8,h,w)

        first_valid = np.argmax(valid_neighbor, axis=0)

        rows, cols = np.where(mask)

        reconstructed[rows, cols, :3] = neighbours[first_valid[rows, cols], rows, cols]
            
        #print("Palette colors:",palette_colors)


    #checkpoint2 = reconstructed.copy()


    # img alpha over reconstructed

    # w,h,_ = img.shape
    # for i in range(0,w-1,1):
    #     for j in range(0,h-1,1):
    #         pixel = reconstructed[i,j]
    #         calculated_scalar = (255-img[i,j,3])/255
    #         pixel[:3] =  pixel[:3]*calculated_scalar + img[i,j][:3] * (1-calculated_scalar)

    # # vectorized version
    alpha = img[:, :,3] / 255.0             
    scalar = 1.0 - alpha                      

    scalar = scalar[..., None]               

    reconstructed[:, :,:3] = (
        reconstructed[:, :,:3] * scalar + img[:, :,:3] * (1 - scalar)
    )


    # clip values and return as readable image type.
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    # plt.title("contour drawing")
    # plt.imshow(cv.cvtColor(checkpoint1, cv.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.show()

    # plt.title("Floodfill")
    # plt.imshow(cv.cvtColor(checkpoint2, cv.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.show()

    plt.title("Reconstructed image")
    plt.imshow(cv.cvtColor(reconstructed, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    return reconstructed


if __name__ == "__main__":

    from sklearn.preprocessing import StandardScaler

    '''
        Training
    '''

    training_input = input("Train model on data? (Y/N)")
    if training_input in "yY":
        not_trained = True
    else:
        not_trained = False

    if not_trained:
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
        params = {
            "n_estimators": [200,400,600,800,1000],
            "max_depth": [None, 10, 20, 30 , 50],
            "max_features": [None,"sqrt", "log2"]
        }
        best_acc = 0
        best_clf = None
        for i in params["n_estimators"]:
            for j in params["max_features"]:
                x_test, y_test, y_pred, clf, acc = learn_classifier(data, y_cls,i,j)
                if acc > best_acc:
                    best_acc = acc
                    best_clf = clf


        dump(best_clf, "cache/model.joblib")
        dump(best_acc, "cache/scores.joblib")
    else:
        scaler = StandardScaler()
        palette_colors = load_palette_colors("palette.png")
        clf = load("cache/model.joblib")
        acc = load("cache/scores.joblib")

    
    
    '''
        Reconstruction
    '''
    print("Color prediction accuracy:", acc)

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

    pass
