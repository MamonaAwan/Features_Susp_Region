# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 12:38:15 2020

@author: Mona
"""
import math
import cv2
import statistics
import numpy as np
from skimage.transform import resize
from skimage.feature.texture import greycomatrix, greycoprops
from scipy.ndimage.morphology import distance_transform_edt, binary_opening
from skimage.morphology import thin
from skimage.measure import label, regionprops
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


def get_ROI_MMG(orig_img, gen_bin, O_shape, type='Malignant', tag=''):
    gen_bin= resize(gen_bin, (O_shape[0], O_shape[1]), mode='constant', preserve_range=True)
    from skimage import img_as_ubyte
    gen_bin_cv = img_as_ubyte(gen_bin)
    gen_bin_1 = cv2.Canny(gen_bin_cv, 50, 100)
    contours, _ = cv2.findContours(gen_bin_1, 1, 2)
    if type == 'Malignant':
        color = (255, 0, 0)
    elif type == 'Call Back':
        color = (255, 128, 0)
    else:
        color = (0, 255, 0)
    ROI_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)

    try:
        cv2.drawContours(ROI_img, contours, -1, color, 5)
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.putText(ROI_img, tag, (int(x+(0.5*w)), int(y+(0.5*h))), cv2.FONT_HERSHEY_TRIPLEX, 5, color, 5)
    except:
        pass
    return ROI_img

def get_ROI_MMG_heatmap(orig_img, gen_bin, O_shape):
    gen_bin = resize(gen_bin, (O_shape[0], O_shape[1]), mode='constant', preserve_range=True)
    from PIL import Image, ImageDraw, ImageFilter
    from skimage import img_as_ubyte
    ROI_img_heatmap = Image.fromarray(img_as_ubyte(cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)))
    all_labels = label(gen_bin)

    im_heat = Image.open('static_mmg/images/heatmap.png').convert('RGBA')
    max_area = max(i.area for i in regionprops(all_labels))
    for region in regionprops(label_image=all_labels):
        if region.area > (0.3 * max_area):
            try:
                y, x = region.centroid
                r = int(region.major_axis_length)
                im_heat = im_heat.resize((r, r))
                (left, upper, right, lower) = (
                int(x - (r / 2)), int(y - (r / 2)), int(x - (r / 2) + r), int(y - (r / 2) + r))
                mask = Image.new("L", im_heat.size, 0)
                draw = ImageDraw.Draw(mask)
                per_r = int(r * 0.1)
                draw.ellipse((0 + per_r, 0 + per_r, r - per_r, r - per_r), fill=255)
                mask = mask.filter(ImageFilter.GaussianBlur(10))

                im_crop = ROI_img_heatmap.crop((left, upper, right, lower)).convert('RGBA')
                im_pe = Image.blend(im_crop, im_heat, .5)
                im_pe.putalpha(mask)
                ROI_img_heatmap.paste(im_pe, (left, upper), im_pe)
            except:
                pass
        else:
            pass
    return ROI_img_heatmap

def get_ROI_USG(orig_img, gen_bin, N_shape, O_shape, type = 'Malignant'):
    if len(orig_img.shape) == 2:
        orig_w, orig_h = orig_img.shape
    else:
        orig_w, orig_h, _ = orig_img.shape

    if (gen_bin.shape[0] == gen_bin.shape[1] == 256):
        gen_bin = resize(gen_bin, (orig_w, orig_h), mode='constant', preserve_range=True)

    gen_bin_2 = cv2.Canny(gen_bin.astype('uint8'), 50, 200)

    contours, hierarchy = cv2.findContours(gen_bin_2, 1, 2)
    if type == 'Malignant':
        color = (255,0,0)
    elif type == 'Call Back':
        color = (255, 128, 0)
    else:
        color = (0,255,0)
    try:
        c = contours[0]
        cv2.drawContours(orig_img, contours, -1, color, 2)
    except:
        c = 0

    ROI_img = orig_img[0:N_shape[1], 0:N_shape[0]]
    ROI_img = cv2.resize(ROI_img, (O_shape[1], O_shape[0]), interpolation=cv2.INTER_AREA)
    return ROI_img

def shape_mass(orig_img, gen_bin):
    if len(orig_img.shape) == 2:
        orig_w, orig_h = orig_img.shape
    else:
        orig_w, orig_h, _ = orig_img.shape

    if (gen_bin.shape[0] == gen_bin.shape[1] == 256):
        gen_bin = resize(gen_bin, (orig_h, orig_w), mode='constant', preserve_range=True)

    edged = cv2.Canny(gen_bin.astype('uint8'), 10, 200)

    try:
        contours, hierarchy = cv2.findContours(edged, 1, 2)
        c = contours[0]
        mask = np.zeros(edged.shape, np.uint8)
        cv2.drawContours(mask, [c], 0, 255, -1)

    except IndexError as error:
        c = 0

    try:
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        app = len(approx)

    except cv2.error as error:
        app = 0
    Shape = ""
    if app == 0:
        Shape = 'Undefined'
    elif 6 < app < 15:
        Shape = 'Oval'
    else:
        Shape = 'Irregular'
    return Shape


def margin_mass(orig_img, gen_bin):
    if (orig_img.shape[0] != 256) or (orig_img.shape[1] != 256):
        orig_img = resize(orig_img, (256, 256), mode='constant', preserve_range=True)
        orig_img = np.uint16(orig_img)
    if (gen_bin.shape[0] != 256) or (gen_bin.shape[1] != 256):
        gen_bin = resize(gen_bin, (256, 256), mode='constant', preserve_range=True)

    gen_bin = np.uint8(gen_bin)
    edged = cv2.Canny(gen_bin, 10, 200)
    try:
        contours, hierarchy = cv2.findContours(edged, 1, 2)
        c = contours[0]
        mask = np.zeros(edged.shape, np.uint8)
        cv2.drawContours(mask, [c], 0, 255, -1)

    except IndexError as error:
        c = 0
    try:
        convexity = cv2.isContourConvex(c)

    except cv2.error as error:
        convexity = 0.0

    Margin = ""
    Lobules = 0
    Ang = 0
    if len(gen_bin.shape) == 3:
        gen_bin_1 = cv2.cvtColor(gen_bin, cv2.COLOR_BGR2GRAY)
    else:
        gen_bin_1 = gen_bin.copy()
    if convexity == True:
        Margin = 'Circumscribed'
    else:
        #Indices of non zero elements
        [y, x] = np.nonzero(gen_bin_1)
        try:
            xmn, xmx = min(x), max(x)
        except:
            xmn, xmx = 0, 0
        try:
            ymn, ymx = min(y), max(y)
        except:
            ymn, ymx = 0, 0

        BW2 = gen_bin_1[ymn:ymx, xmn:xmx]
        BW2 = np.pad(BW2, (1, 1), 'constant')
        # Compute distance map
        D = distance_transform_edt(np.logical_not(1-BW2))
        # Inscribed circle distance map
        [M, N] = np.shape(BW2)
        [y, x] = np.nonzero(D == max(np.concatenate(D)))
        try:
            xc = statistics.mean(x)
        except:
            xc = 0
        try:
            yc = statistics.mean(y)
        except:
            yc = 0

        [X, Y] = np.meshgrid(N, M, sparse=True)
        C = math.sqrt((X -xc)^2 + (Y-yc)^2)

        # Get lobules with the maximum inscribed circle
        r = max(np.concatenate(D))+1 # Radius
        BW3 = np.logical_not(BW2) ^ (C>=r)
        BW3 = binary_opening(BW3, structure=np.ones((2,2))).astype(int)
        temp = BW3.astype('uint8')
        L1= cv2.connectedComponents(temp, 8)
        L = L1[0]
        U = 0
        i = 1
        for i in range(L):
            idx = L==i
            D3 = distance_transform_edt(np.logical_not(idx))    #Interior distances
            if max(D3) > 0:
                U = U + 1 # Its a lobule
        Lobules = U

        Se_Ang = thin(BW2, max_iter = None)
        Clr = np.ones(Se_Ang.shape)
        cv2.circle(Clr,(int(X/2),int(Y/2)), int(r*2), 0, -1)
        BW4 = np.logical_and(Se_Ang, Clr)
        temp2 = BW4.astype('uint8')
        L2 = cv2.connectedComponents(temp2, 8)
        Ang = L2[0]

        orig_img_1 = orig_img.copy()
        outer_gb = cv2.dilate(gen_bin_1, kernel = np.ones((3,3)))
        inner_gb = cv2.erode(gen_bin_1, kernel = np.ones((3,3)))
        outer_gb = np.logical_and(outer_gb, np.logical_not(inner_gb))

        roi_r = np.multiply(gen_bin_1, orig_img_1)
        avg_I_in = (roi_r.astype('uint8')).sum()/max(np.count_nonzero(roi_r), 1)

        roi_outer = np.multiply(outer_gb, orig_img_1)
        n_pixels_outer = max(np.count_nonzero(roi_outer),1)
        avg_I_out = (roi_outer.sum())/max(n_pixels_outer, 1)

        Diff_avg = ((avg_I_out - avg_I_in)/n_pixels_outer).__round__(ndigits=3)

        if Diff_avg >- 0.05 and Diff_avg < 0.05:
            Margin = 'Indistinct '
        else:
            Margin = 'Distinct '

        if U <= 3:
            Margin = Margin + 'and Not Circumscribed with smooth lobulations'
        else:
            Margin = Margin + 'and Not Circumscribed with micro lobulations'
    return Margin, Lobules, Ang

def orientation_axis_mass(orig_img, gen_bin):
    if (gen_bin.shape[0] != 256) or (gen_bin.shape[1] != 256):
        gen_bin = resize(gen_bin, (256, 256), mode='constant', preserve_range=True)
        gen_bin = np.uint8(gen_bin)

    # Orientation of Tumor
    edged = cv2.Canny(gen_bin, 50, 100)

    # Tumor Properties
    label_img = label(edged)
    regions = regionprops(label_img)
    for props in regions:
        orientation = props.orientation

    Orientation = ""

    try:
        if (-1.57 < orientation <= -1.45) or (1.45 <= orientation < 1.57):
            Orientation = 'Parallel to Skin'
        elif -1.45 < orientation < 1.45:
            Orientation = 'Perpendicular to Skin'

    except UnboundLocalError as error:
        Orientation = 'Undefined'

    majoraxis = []
    minoraxis = []

    try:
        majoraxis = ((props.major_axis_length) / 37.79).__round__(3)
        minoraxis = ((props.minor_axis_length) / 37.79).__round__(3)

    except UnboundLocalError as error:
        majoraxis = 0
        minoraxis = 0

    # Depth to Width Ratio
    DWR = ""
    gen_bin_1 = cv2.cvtColor(gen_bin, cv2.COLOR_BGR2GRAY)
    [yBW,xBW] = np.nonzero(gen_bin_1)
    try:
        xBWmax = max(xBW)
        xBWmin = min(xBW)
    except:
        xBWmax = 0
        xBWmin = 0
    try:
        yBWmax = max(yBW)
        yBWmin = min(yBW)
    except:
        yBWmax = 0
        yBWmin = 0
    try:
        DWR = ((yBWmax-yBWmin)/(xBWmax-xBWmin)).__round__(3)
    except:
        0

    return Orientation, majoraxis, minoraxis, DWR

def echo_pattern_mass(orig_img, gen_bin):
    if (gen_bin.shape[0] != 256) or (gen_bin.shape[1] != 256):
        gen_bin = resize(gen_bin, (256, 256), mode='constant', preserve_range=True)
        gen_bin = np.uint8(gen_bin)

    edged = cv2.Canny(gen_bin, 50, 100)
    try:
        contours, hierarchy = cv2.findContours(edged, 1, 2)
        c = contours[0]
    except:
        c = 0

    try:
        mask = np.zeros(edged.shape, np.uint8)
        cv2.drawContours(mask, [c], 0, 255, -1)
        mean_val = cv2.mean(gen_bin, mask=mask)
        mean_val = mean_val[0]
    except:
        mean_val = 0

    Echo_pattern = ""
    if 0 <= mean_val <= 40:
        Echo_pattern = 'Anechoic'
    elif 41 <= mean_val <= 100:
        Echo_pattern = 'Hypoechoic'
    elif 101 <= mean_val <= 170:
        Echo_pattern = 'Isoechoic'
    elif 171 <= mean_val <= 220:
        Echo_pattern = 'Hyperechoic'
    else:
        Echo_pattern = 'Echogenic'

    # print("Echo pattern: ", Echo_pattern)
    return Echo_pattern

def texture_mass(orig_img, gen_bin):
    # Texture of Tumor
    gen_bin_1 = (rgb2gray(gen_bin)).astype('uint8')
    g = greycomatrix(gen_bin_1, [1, 2], [0, np.pi / 2])
    homogeneity = greycoprops(g, 'homogeneity')[0, 0]
    Texture = ""
    if homogeneity == 0.5:
        Texture = 'Homogeneous'
    else:
        Texture = 'Heterogenous'

    # print("Texture of Tumor: ", Texture)
    return Texture

def shadowing_mass(orig_img, gen_bin):
    if len(orig_img.shape) == 2:
        orig_w, orig_h = orig_img.shape
    else:
        orig_w, orig_h, _ = orig_img.shape

    if (gen_bin.shape[0] == gen_bin.shape[1] == 256):
        gen_bin = resize(gen_bin, (orig_w, orig_h), mode='constant', preserve_range=True)
    gen_bin_1 = rgb2gray(gen_bin)

    img1 = rgb2gray(orig_img)
    ROI_img = cv2.multiply(gen_bin_1, img1)
    ROI_img = ROI_img.astype(np.uint8)

    Shadowing = ''
    try:
        # ROI Display with Outline
        edged = cv2.Canny(ROI_img, 30, 200)

        # Tumor Properties
        label_img = label(edged)
        regions = regionprops(label_img)
        for props in regions:
            box = props.bbox

        minr, minc, maxr, maxc = box
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)

        # Acoustic Shadowing of Tumor
        # Tumor Intensity
        tumor_img = gen_bin_1[minr:maxr, minc:maxc]
        tumor_img = tumor_img.astype(np.uint8)
        tumor_intensity = np.mean(tumor_img)

        # Height of Shadow
        H1 = maxr - minr
        H2 = 256 - maxr

        if H1 > H2:
            Maxy = maxr + H2
        else:
            Maxy = maxr + H1

        # Width of Shadow
        W = maxc - minc
        W1 = (2 * (maxc - minc)) / 3
        W0 = round(W - W1)
        minc1 = minc + round(W0 / 2)
        maxc1 = maxc - round(W0 / 2)

        # Points of Shadow
        Miny, Minx, Maxx = maxr, minc1, maxc1

        # Shadow Intentsity
        shadow_img = gen_bin_1[Miny:Maxy, Minx:Maxx]
        shadow_img = shadow_img.astype(np.uint8)
        post_intensity = np.mean(shadow_img)

        Shadow = post_intensity - tumor_intensity
        try:
            if Shadow > 0:
                Shadowing = 'Posterior Acoustic Enhancement'
            else:
                Shadowing = 'Posterior Shadowing'

        except UnboundLocalError as error:
            Shadow = 0
            Shadowing = 'Undefined'

    except UnboundLocalError as error:
        Shadow = 0
        Shadowing = 'Undefined'

    return Shadowing

def check_valid(gen_bin):
    if np.all(gen_bin == 0):
        return False
    else:
        return True

def select_roi(gen_bin):
    labels = label(gen_bin)
    masked = np.zeros(gen_bin.shape)
    max_area = 0
    for region in regionprops(labels):
        if region.area >= max_area:
            minr, minc, maxr, maxc = region.bbox
            max_area = region.area
        else:
            pass
    masked[minr:maxr, minc:maxc] = gen_bin[minr:maxr, minc:maxc]
    return masked

def MMG_Roi_Features(orig_img, gen_bin, filename):
    Shape_Roi = shape_mass(orig_img, gen_bin)
    Margin_Type, Lobules, Angularities = margin_mass(orig_img, gen_bin)
    return Shape_Roi, Margin_Type, Lobules, Angularities

def USG_Roi_Features(orig_img, gen_bin):
    Shape_Roi = shape_mass(orig_img, gen_bin)
    Margin_Type, Lobules, Angularities = margin_mass(orig_img, gen_bin)
    Orientation, Major_Axis, Minor_Axis, DepthtoWidthRatio = orientation_axis_mass(orig_img, gen_bin)
    Echo_Pattern = echo_pattern_mass(orig_img, gen_bin)
    Texture = texture_mass(orig_img, gen_bin)
    Shadowing = shadowing_mass(orig_img, gen_bin)
    return Shape_Roi, Margin_Type, Lobules, Angularities, Orientation, Major_Axis, Minor_Axis, DepthtoWidthRatio, Echo_Pattern, Texture, Shadowing


def ff_multi_rios(orig_img, gen_bin, filename):
    O_shape = orig_img.shape
    #getting heatmap
    Result_heatmap = get_ROI_MMG_heatmap(orig_img, gen_bin, O_shape)

    labels = label(gen_bin)
    sep_rois = []
    max_area = max(i.area for i in regionprops(labels))
    for region in regionprops(labels):
        if region.area > (0.3*max_area):
            masked = np.zeros(gen_bin.shape)
            minr, minc, maxr, maxc = region.bbox
            masked[minr:maxr, minc:maxc] = gen_bin[minr:maxr, minc:maxc]
            sep_rois.append(masked)

    Result = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
    data = []
    for i in range(len(sep_rois)):
        tag = str('roi'+str(i))
        Shape_Roi, Margin_Type, Lobules, Angularities = MMG_Roi_Features(orig_img, sep_rois[i], filename)
        from classifyMMG import classify_tumor_MMG
        tumour_type = classify_tumor_MMG(Shape_Roi, Margin_Type)
        #collecting data per roi
        data.append([tag, Shape_Roi, Margin_Type, tumour_type])
        Res_roi = get_ROI_MMG(orig_img, sep_rois[i], O_shape, tumour_type, tag)
        #collecting rois with tags
        Result = cv2.bitwise_or(Res_roi, Result)

    from Pec_Rem_MMG import Calculate_Breast_Density
    Breast_Density_Category, Breast_Density = Calculate_Breast_Density(filename)
    return Result, Result_heatmap, data, Breast_Density_Category, Breast_Density




