!pip install colormath

import cv2
import os
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
import pandas as pd
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color



def enhance_image(image):

    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB channels
    l, a, b = cv2.split(lab)

    # Enhance the L channel
    clahe = cv2.createCLAHE(clipLimit=0.07, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)

    # Merge the enhanced L channel with the original A and B channels
    enhanced_lab = cv2.merge([enhanced_l, a, b])

    # Convert the enhanced LAB image back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_image


def change_resolution(image):
  # Define the desired width and height
  new_width = 600
  new_height = 700

  # Resize the image
  resized_image = cv2.resize(image, (new_width, new_height))
  return resized_image



def segment_garment(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask for the foreground (garment) and initialize it as background
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[gray > 0] = cv2.GC_PR_BGD

    # Define the rectangle region that contains the garment
    h, w = image.shape[:2]
    rect = (10, 10, w-20, h-20)

    # Apply GrabCut algorithm to refine the mask
    mask, bgdModel, fgdModel = cv2.grabCut(image, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)
    

    # Create a binary mask where the foreground is labeled as likely or definite foreground
    foreground_mask = np.where((mask == cv2.GC_PR_FGD) | (mask == cv2.GC_FGD), 255, 0).astype(np.uint8)

    # Apply the binary mask to extract the garment object
    segmented_image = cv2.bitwise_and(image, image, mask=foreground_mask)

    return segmented_image


def get_dominant_bgr_using_meanshift_brightness(image):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Flatten the image into a 1D array of pixels
    pixels = hsv_image.reshape(-1, 3)


    # Extract the brightness values (V channel)
    brightness_values = pixels[:, 2]
    # print(brightness_values)
    # Define the threshold for background-like brightness values (adjust as needed)
    background_threshold = 1
    # Create a mask to exclude background pixels based on brightness
    mask = brightness_values > background_threshold
   # Apply the mask to filter out background pixels
    non_background_pixels = pixels[mask]


  
 # Use MeanShift clustering to find the dominant color
    bandwidth = estimate_bandwidth(non_background_pixels, quantile=0.2, n_samples=500)
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    meanshift.fit(non_background_pixels)

    # Get the HSV values of the dominant color
    dominant_color_hsv = meanshift.cluster_centers_[0]

    # Convert dominant color back to BGR color space for visualization
    dominant_color_bgr = cv2.cvtColor(np.uint8([[dominant_color_hsv]]), cv2.COLOR_HSV2BGR)[0][0]

    # Create a copy of the image
    marked_image = image.copy()

    # Draw a rectangle with the dominant color on the copied image
    cv2.rectangle(marked_image, (0, 0), (100, 100), dominant_color_bgr.tolist(), -1)


    return dominant_color_bgr.astype(int), marked_image



# rgb values for desired colors for finding closest color through distance
# https://cloford.com/resources/colours/500col.htm
custom_colors = {
    'indian red' :  (176,23,31) , 'crimson' :  (220,20,60) , 'lightpink' :  (255,182,193) , 'lightpink 1' :  (255,174,185) , 'lightpink 2' :  (238,162,173) , 'lightpink 3' :  (205,140,149) , 'lightpink 4' :  (139,95,101) , 'pink' :  (255,192,203) , 'pink 1' :  (255,181,197) , 'pink 2' :  (238,169,184) , 'pink 3' :  (205,145,158) , 'pink 4' :  (139,99,108) , 'palevioletred' :  (219,112,147) , 'palevioletred 1' :  (255,130,171) , 'palevioletred 2' :  (238,121,159) , 'palevioletred 3' :  (205,104,137) , 'palevioletred 4' :  (139,71,93) , 'lavenderblush 1 (lavenderblush)' :  (255,240,245) , 'lavenderblush 2' :  (238,224,229) , 'lavenderblush 3' :  (205,193,197) , 'lavenderblush 4' :  (139,131,134) , 'violetred 1' :  (255,62,150) , 'violetred 2' :  (238,58,140) , 'violetred 3' :  (205,50,120) , 'violetred 4' :  (139,34,82) , 'hotpink' :  (255,105,180) , 'hotpink 1' :  (255,110,180) , 'hotpink 2' :  (238,106,167) , 'hotpink 3' :  (205,96,144) , 'hotpink 4' :  (139,58,98) , 'raspberry' :  (135,38,87) , 'deeppink 1 (deeppink)' :  (255,20,147) , 'deeppink 2' :  (238,18,137) , 'deeppink 3' :  (205,16,118) , 'deeppink 4' :  (139,10,80) , 'maroon 1' :  (255,52,179) , 'maroon 2' :  (238,48,167) , 'maroon 3' :  (205,41,144) , 'maroon 4' :  (139,28,98) , 'mediumvioletred' :  (199,21,133) , 'violetred' :  (208,32,144) , 'orchid' :  (218,112,214) , 'orchid 1' :  (255,131,250) , 'orchid 2' :  (238,122,233) , 'orchid 3' :  (205,105,201) , 'orchid 4' :  (139,71,137) , 'thistle' :  (216,191,216) , 'thistle 1' :  (255,225,255) , 'thistle 2' :  (238,210,238) , 'thistle 3' :  (205,181,205) , 'thistle 4' :  (139,123,139) , 'plum 1' :  (255,187,255) , 'plum 2' :  (238,174,238) , 'plum 3' :  (205,150,205) , 'plum 4' :  (139,102,139) , 'plum' :  (221,160,221) , 'violet' :  (238,130,238) , 'magenta (fuchsia*)' :  (255,0,255) , 'magenta 2' :  (238,0,238) , 'magenta 3' :  (205,0,205) , 'magenta 4 (darkmagenta)' :  (139,0,139) , 'purple*' :  (128,0,128) , 'mediumorchid' :  (186,85,211) , 'mediumorchid 1' :  (224,102,255) , 'mediumorchid 2' :  (209,95,238) , 'mediumorchid 3' :  (180,82,205) , 'mediumorchid 4' :  (122,55,139) , 'darkviolet' :  (148,0,211) , 'darkorchid' :  (153,50,204) , 'darkorchid 1' :  (191,62,255) , 'darkorchid 2' :  (178,58,238) , 'darkorchid 3' :  (154,50,205) , 'darkorchid 4' :  (104,34,139) , 'indigo' :  (75,0,130) , 'blueviolet' :  (138,43,226) , 'purple 1' :  (155,48,255) , 'purple 2' :  (145,44,238) , 'purple 3' :  (125,38,205) , 'purple 4' :  (85,26,139) , 'mediumpurple' :  (147,112,219) , 'mediumpurple 1' :  (171,130,255) , 'mediumpurple 2' :  (159,121,238) , 'mediumpurple 3' :  (137,104,205) , 'mediumpurple 4' :  (93,71,139) , 'darkslateblue' :  (72,61,139) , 'lightslateblue' :  (132,112,255) , 'mediumslateblue' :  (123,104,238) , 'slateblue' :  (106,90,205) , 'slateblue 1' :  (131,111,255) , 'slateblue 2' :  (122,103,238) , 'slateblue 3' :  (105,89,205) , 'slateblue 4' :  (71,60,139) , 'ghostwhite' :  (248,248,255) , 'lavender' :  (230,230,250) , 'blue*' :  (0,0,255) , 'blue 2' :  (0,0,238) , 'blue 3 (mediumblue)' :  (0,0,205) , 'blue 4 (darkblue)' :  (0,0,139) , 'navy*' :  (0,0,128) , 'midnightblue' :  (25,25,112) , 'cobalt' :  (61,89,171) , 'royalblue' :  (65,105,225) , 'royalblue 1' :  (72,118,255) , 'royalblue 2' :  (67,110,238) , 'royalblue 3' :  (58,95,205) , 'royalblue 4' :  (39,64,139) , 'cornflowerblue' :  (100,149,237) , 'lightsteelblue' :  (176,196,222) , 'lightsteelblue 1' :  (202,225,255) , 'lightsteelblue 2' :  (188,210,238) , 'lightsteelblue 3' :  (162,181,205) , 'lightsteelblue 4' :  (110,123,139) , 'lightslategray' :  (119,136,153) , 'slategray' :  (112,128,144) , 'slategray 1' :  (198,226,255) , 'slategray 2' :  (185,211,238) , 'slategray 3' :  (159,182,205) , 'slategray 4' :  (108,123,139) , 'dodgerblue 1 (dodgerblue)' :  (30,144,255) , 'dodgerblue 2' :  (28,134,238) , 'dodgerblue 3' :  (24,116,205) , 'dodgerblue 4' :  (16,78,139) , 'aliceblue' :  (240,248,255) , 'steelblue' :  (70,130,180) , 'steelblue 1' :  (99,184,255) , 'steelblue 2' :  (92,172,238) , 'steelblue 3' :  (79,148,205) , 'steelblue 4' :  (54,100,139) , 'lightskyblue' :  (135,206,250) , 'lightskyblue 1' :  (176,226,255) , 'lightskyblue 2' :  (164,211,238) , 'lightskyblue 3' :  (141,182,205) , 'lightskyblue 4' :  (96,123,139) , 'skyblue 1' :  (135,206,255) , 'skyblue 2' :  (126,192,238) , 'skyblue 3' :  (108,166,205) , 'skyblue 4' :  (74,112,139) , 'skyblue' :  (135,206,235) , 'deepskyblue 1 (deepskyblue)' :  (0,191,255) , 'deepskyblue 2' :  (0,178,238) , 'deepskyblue 3' :  (0,154,205) , 'deepskyblue 4' :  (0,104,139) , 'peacock' :  (51,161,201) , 'lightblue' :  (173,216,230) , 'lightblue 1' :  (191,239,255) , 'lightblue 2' :  (178,223,238) , 'lightblue 3' :  (154,192,205) , 'lightblue 4' :  (104,131,139) , 'powderblue' :  (176,224,230) , 'cadetblue 1' :  (152,245,255) , 'cadetblue 2' :  (142,229,238) , 'cadetblue 3' :  (122,197,205) , 'cadetblue 4' :  (83,134,139) , 'turquoise 1' :  (0,245,255) , 'turquoise 2' :  (0,229,238) , 'turquoise 3' :  (0,197,205) , 'turquoise 4' :  (0,134,139) , 'cadetblue' :  (95,158,160) , 'darkturquoise' :  (0,206,209) , 'azure 1 (azure)' :  (240,255,255) , 'azure 2' :  (224,238,238) , 'azure 3' :  (193,205,205) , 'azure 4' :  (131,139,139) , 'lightcyan 1 (lightcyan)' :  (224,255,255) , 'lightcyan 2' :  (209,238,238) , 'lightcyan 3' :  (180,205,205) , 'lightcyan 4' :  (122,139,139) , 'paleturquoise 1' :  (187,255,255) , 'paleturquoise 2 (paleturquoise)' :  (174,238,238) , 'paleturquoise 3' :  (150,205,205) , 'paleturquoise 4' :  (102,139,139) , 'darkslategray' :  (47,79,79) , 'darkslategray 1' :  (151,255,255) , 'darkslategray 2' :  (141,238,238) , 'darkslategray 3' :  (121,205,205) , 'darkslategray 4' :  (82,139,139) , 'cyan / aqua*' :  (0,255,255) , 'cyan 2' :  (0,238,238) , 'cyan 3' :  (0,205,205) , 'cyan 4 (darkcyan)' :  (0,139,139) , 'teal*' :  (0,128,128) , 'mediumturquoise' :  (72,209,204) , 'lightseagreen' :  (32,178,170) , 'manganeseblue' :  (3,168,158) , 'turquoise' :  (64,224,208) , 'coldgrey' :  (128,138,135) , 'turquoiseblue' :  (0,199,140) , 'aquamarine 1 (aquamarine)' :  (127,255,212) , 'aquamarine 2' :  (118,238,198) , 'aquamarine 3 (mediumaquamarine)' :  (102,205,170) , 'aquamarine 4' :  (69,139,116) , 'mediumspringgreen' :  (0,250,154) , 'mintcream' :  (245,255,250) , 'springgreen' :  (0,255,127) , 'springgreen 1' :  (0,238,118) , 'springgreen 2' :  (0,205,102) , 'springgreen 3' :  (0,139,69) , 'mediumseagreen' :  (60,179,113) , 'seagreen 1' :  (84,255,159) , 'seagreen 2' :  (78,238,148) , 'seagreen 3' :  (67,205,128) , 'seagreen 4 (seagreen)' :  (46,139,87) , 'emeraldgreen' :  (0,201,87) , 'mint' :  (189,252,201) , 'cobaltgreen' :  (61,145,64) , 'honeydew 1 (honeydew)' :  (240,255,240) , 'honeydew 2' :  (224,238,224) , 'honeydew 3' :  (193,205,193) , 'honeydew 4' :  (131,139,131) , 'darkseagreen' :  (143,188,143) , 'darkseagreen 1' :  (193,255,193) , 'darkseagreen 2' :  (180,238,180) , 'darkseagreen 3' :  (155,205,155) , 'darkseagreen 4' :  (105,139,105) , 'palegreen' :  (152,251,152) , 'palegreen 1' :  (154,255,154) , 'palegreen 2 (lightgreen)' :  (144,238,144) , 'palegreen 3' :  (124,205,124) , 'palegreen 4' :  (84,139,84) , 'limegreen' :  (50,205,50) , 'forestgreen' :  (34,139,34) , 'green 1 (lime*)' :  (0,255,0) , 'green 2' :  (0,238,0) , 'green 3' :  (0,205,0) , 'green 4' :  (0,139,0) , 'green*' :  (0,128,0) , 'darkgreen' :  (0,100,0) , 'sapgreen' :  (48,128,20) , 'lawngreen' :  (124,252,0) , 'chartreuse 1 (chartreuse)' :  (127,255,0) , 'chartreuse 2' :  (118,238,0) , 'chartreuse 3' :  (102,205,0) , 'chartreuse 4' :  (69,139,0) , 'greenyellow' :  (173,255,47) , 'darkolivegreen 1' :  (202,255,112) , 'darkolivegreen 2' :  (188,238,104) , 'darkolivegreen 3' :  (162,205,90) , 'darkolivegreen 4' :  (110,139,61) , 'darkolivegreen' :  (85,107,47) , 'olivedrab' :  (107,142,35) , 'olivedrab 1' :  (192,255,62) , 'olivedrab 2' :  (179,238,58) , 'olivedrab 3 (yellowgreen)' :  (154,205,50) , 'olivedrab 4' :  (105,139,34) , 'ivory 1 (ivory)' :  (255,255,240) , 'ivory 2' :  (238,238,224) , 'ivory 3' :  (205,205,193) , 'ivory 4' :  (139,139,131) , 'beige' :  (245,245,220) , 'lightyellow 1 (lightyellow)' :  (255,255,224) , 'lightyellow 2' :  (238,238,209) , 'lightyellow 3' :  (205,205,180) , 'lightyellow 4' :  (139,139,122) , 'lightgoldenrodyellow' :  (250,250,210) , 'yellow 1 (yellow*)' :  (255,255,0) , 'yellow 2' :  (238,238,0) , 'yellow 3' :  (205,205,0) , 'yellow 4' :  (139,139,0) , 'warmgrey' :  (128,128,105) , 'olive*' :  (128,128,0) , 'darkkhaki' :  (189,183,107) , 'khaki 1' :  (255,246,143) , 'khaki 2' :  (238,230,133) , 'khaki 3' :  (205,198,115) , 'khaki 4' :  (139,134,78) , 'khaki' :  (240,230,140) , 'palegoldenrod' :  (238,232,170) , 'lemonchiffon 1 (lemonchiffon)' :  (255,250,205) , 'lemonchiffon 2' :  (238,233,191) , 'lemonchiffon 3' :  (205,201,165) , 'lemonchiffon 4' :  (139,137,112) , 'lightgoldenrod 1' :  (255,236,139) , 'lightgoldenrod 2' :  (238,220,130) , 'lightgoldenrod 3' :  (205,190,112) , 'lightgoldenrod 4' :  (139,129,76) , 'banana' :  (227,207,87) , 'gold 1 (gold)' :  (255,215,0) , 'gold 2' :  (238,201,0) , 'gold 3' :  (205,173,0) , 'gold 4' :  (139,117,0) , 'cornsilk 1 (cornsilk)' :  (255,248,220) , 'cornsilk 2' :  (238,232,205) , 'cornsilk 3' :  (205,200,177) , 'cornsilk 4' :  (139,136,120) , 'goldenrod' :  (218,165,32) , 'goldenrod 1' :  (255,193,37) , 'goldenrod 2' :  (238,180,34) , 'goldenrod 3' :  (205,155,29) , 'goldenrod 4' :  (139,105,20) , 'darkgoldenrod' :  (184,134,11) , 'darkgoldenrod 1' :  (255,185,15) , 'darkgoldenrod 2' :  (238,173,14) , 'darkgoldenrod 3' :  (205,149,12) , 'darkgoldenrod 4' :  (139,101,8) , 'orange 1 (orange)' :  (255,165,0) , 'orange 2' :  (238,154,0) , 'orange 3' :  (205,133,0) , 'orange 4' :  (139,90,0) , 'floralwhite' :  (255,250,240) , 'oldlace' :  (253,245,230) , 'wheat' :  (245,222,179) , 'wheat 1' :  (255,231,186) , 'wheat 2' :  (238,216,174) , 'wheat 3' :  (205,186,150) , 'wheat 4' :  (139,126,102) , 'moccasin' :  (255,228,181) , 'papayawhip' :  (255,239,213) , 'blanchedalmond' :  (255,235,205) , 'navajowhite 1 (navajowhite)' :  (255,222,173) , 'navajowhite 2' :  (238,207,161) , 'navajowhite 3' :  (205,179,139) , 'navajowhite 4' :  (139,121,94) , 'eggshell' :  (252,230,201) , 'tan' :  (210,180,140) , 'brick' :  (156,102,31) , 'cadmiumyellow' :  (255,153,18) , 'antiquewhite' :  (250,235,215) , 'antiquewhite 1' :  (255,239,219) , 'antiquewhite 2' :  (238,223,204) , 'antiquewhite 3' :  (205,192,176) , 'antiquewhite 4' :  (139,131,120) , 'burlywood' :  (222,184,135) , 'burlywood 1' :  (255,211,155) , 'burlywood 2' :  (238,197,145) , 'burlywood 3' :  (205,170,125) , 'burlywood 4' :  (139,115,85) , 'bisque 1 (bisque)' :  (255,228,196) , 'bisque 2' :  (238,213,183) , 'bisque 3' :  (205,183,158) , 'bisque 4' :  (139,125,107) , 'melon' :  (227,168,105) , 'carrot' :  (237,145,33) , 'darkorange' :  (255,140,0) , 'darkorange 1' :  (255,127,0) , 'darkorange 2' :  (238,118,0) , 'darkorange 3' :  (205,102,0) , 'darkorange 4' :  (139,69,0) , 'orange' :  (255,128,0) , 'tan 1' :  (255,165,79) , 'tan 2' :  (238,154,73) , 'tan 3 (peru)' :  (205,133,63) , 'tan 4' :  (139,90,43) , 'linen' :  (250,240,230) , 'peachpuff 1 (peachpuff)' :  (255,218,185) , 'peachpuff 2' :  (238,203,173) , 'peachpuff 3' :  (205,175,149) , 'peachpuff 4' :  (139,119,101) , 'seashell 1 (seashell)' :  (255,245,238) , 'seashell 2' :  (238,229,222) , 'seashell 3' :  (205,197,191) , 'seashell 4' :  (139,134,130) , 'sandybrown' :  (244,164,96) , 'rawsienna' :  (199,97,20) , 'chocolate' :  (210,105,30) , 'chocolate 1' :  (255,127,36) , 'chocolate 2' :  (238,118,33) , 'chocolate 3' :  (205,102,29) , 'chocolate 4 (saddlebrown)' :  (139,69,19) , 'ivoryblack' :  (41,36,33) , 'flesh' :  (255,125,64) , 'cadmiumorange' :  (255,97,3) , 'burntsienna' :  (138,54,15) , 'sienna' :  (160,82,45) , 'sienna 1' :  (255,130,71) , 'sienna 2' :  (238,121,66) , 'sienna 3' :  (205,104,57) , 'sienna 4' :  (139,71,38) , 'lightsalmon 1 (lightsalmon)' :  (255,160,122) , 'lightsalmon 2' :  (238,149,114) , 'lightsalmon 3' :  (205,129,98) , 'lightsalmon 4' :  (139,87,66) , 'coral' :  (255,127,80) , 'orangered 1 (orangered)' :  (255,69,0) , 'orangered 2' :  (238,64,0) , 'orangered 3' :  (205,55,0) , 'orangered 4' :  (139,37,0) , 'sepia' :  (94,38,18) , 'darksalmon' :  (233,150,122) , 'salmon 1' :  (255,140,105) , 'salmon 2' :  (238,130,98) , 'salmon 3' :  (205,112,84) , 'salmon 4' :  (139,76,57) , 'coral 1' :  (255,114,86) , 'coral 2' :  (238,106,80) , 'coral 3' :  (205,91,69) , 'coral 4' :  (139,62,47) , 'burntumber' :  (138,51,36) , 'tomato 1 (tomato)' :  (255,99,71) , 'tomato 2' :  (238,92,66) , 'tomato 3' :  (205,79,57) , 'tomato 4' :  (139,54,38) , 'salmon' :  (250,128,114) , 'mistyrose 1 (mistyrose)' :  (255,228,225) , 'mistyrose 2' :  (238,213,210) , 'mistyrose 3' :  (205,183,181) , 'mistyrose 4' :  (139,125,123) , 'snow 1 (snow)' :  (255,250,250) , 'snow 2' :  (238,233,233) , 'snow 3' :  (205,201,201) , 'snow 4' :  (139,137,137) , 'rosybrown' :  (188,143,143) , 'rosybrown 1' :  (255,193,193) , 'rosybrown 2' :  (238,180,180) , 'rosybrown 3' :  (205,155,155) , 'rosybrown 4' :  (139,105,105) , 'lightcoral' :  (240,128,128) , 'indianred' :  (205,92,92) , 'indianred 1' :  (255,106,106) , 'indianred 2' :  (238,99,99) , 'indianred 4' :  (139,58,58) , 'indianred 3' :  (205,85,85) , 'brown' :  (165,42,42) , 'brown 1' :  (255,64,64) , 'brown 2' :  (238,59,59) , 'brown 3' :  (205,51,51) , 'brown 4' :  (139,35,35) , 'firebrick' :  (178,34,34) , 'firebrick 1' :  (255,48,48) , 'firebrick 2' :  (238,44,44) , 'firebrick 3' :  (205,38,38) , 'firebrick 4' :  (139,26,26) , 'red 1 (red*)' :  (255,0,0) , 'red 2' :  (238,0,0) , 'red 3' :  (205,0,0) , 'red 4 (darkred)' :  (139,0,0) , 'maroon*' :  (128,0,0) , 'sgi beet' :  (142,56,142) , 'sgi slateblue' :  (113,113,198) , 'sgi lightblue' :  (125,158,192) , 'sgi teal' :  (56,142,142) , 'sgi chartreuse' :  (113,198,113) , 'sgi olivedrab' :  (142,142,56) , 'sgi brightgray' :  (197,193,170) , 'sgi salmon' :  (198,113,113) , 'sgi darkgray' :  (85,85,85) , 'sgi gray 12' :  (30,30,30) , 'sgi gray 16' :  (40,40,40) , 'sgi gray 32' :  (81,81,81) , 'sgi gray 36' :  (91,91,91) , 'sgi gray 52' :  (132,132,132) , 'sgi gray 56' :  (142,142,142) , 'sgi lightgray' :  (170,170,170) , 'sgi gray 72' :  (183,183,183) , 'sgi gray 76' :  (193,193,193) , 'sgi gray 92' :  (234,234,234) , 'sgi gray 96' :  (244,244,244) , 'white*' :  (255,255,255) , 'white smoke (gray 96)' :  (245,245,245) , 'gainsboro' :  (220,220,220) , 'lightgrey' :  (211,211,211) , 'silver*' :  (192,192,192) , 'darkgray' :  (169,169,169) , 'gray*' :  (128,128,128) , 'dimgray (gray 42)' :  (105,105,105) , 'black*' :  (0,0,0) , 'gray 99' :  (252,252,252) , 'gray 98' :  (250,250,250) , 'gray 97' :  (247,247,247) , 'white smoke (gray 96)' :  (245,245,245) , 'gray 95' :  (242,242,242) , 'gray 94' :  (240,240,240) , 'gray 93' :  (237,237,237) , 'gray 92' :  (235,235,235) , 'gray 91' :  (232,232,232) , 'gray 90' :  (229,229,229) , 'gray 89' :  (227,227,227) , 'gray 88' :  (224,224,224) , 'gray 87' :  (222,222,222) , 'gray 86' :  (219,219,219) , 'gray 85' :  (217,217,217) , 'gray 84' :  (214,214,214) , 'gray 83' :  (212,212,212) , 'gray 82' :  (209,209,209) , 'gray 81' :  (207,207,207) , 'gray 80' :  (204,204,204) , 'gray 79' :  (201,201,201) , 'gray 78' :  (199,199,199) , 'gray 77' :  (196,196,196) , 'gray 76' :  (194,194,194) , 'gray 75' :  (191,191,191) , 'gray 74' :  (189,189,189) , 'gray 73' :  (186,186,186) , 'gray 72' :  (184,184,184) , 'gray 71' :  (181,181,181) , 'gray 70' :  (179,179,179) , 'gray 69' :  (176,176,176) , 'gray 68' :  (173,173,173) , 'gray 67' :  (171,171,171) , 'gray 66' :  (168,168,168) , 'gray 65' :  (166,166,166) , 'gray 64' :  (163,163,163) , 'gray 63' :  (161,161,161) , 'gray 62' :  (158,158,158) , 'gray 61' :  (156,156,156) , 'gray 60' :  (153,153,153) , 'gray 59' :  (150,150,150) , 'gray 58' :  (148,148,148) , 'gray 57' :  (145,145,145) , 'gray 56' :  (143,143,143) , 'gray 55' :  (140,140,140) , 'gray 54' :  (138,138,138) , 'gray 53' :  (135,135,135) , 'gray 52' :  (133,133,133) , 'gray 51' :  (130,130,130) , 'gray 50' :  (127,127,127) , 'gray 49' :  (125,125,125) , 'gray 48' :  (122,122,122) , 'gray 47' :  (120,120,120) , 'gray 46' :  (117,117,117) , 'gray 45' :  (115,115,115) , 'gray 44' :  (112,112,112) , 'gray 43' :  (110,110,110) , 'gray 42' :  (107,107,107) , 'dimgray (gray 42)' :  (105,105,105) , 'gray 40' :  (102,102,102) , 'gray 39' :  (99,99,99) , 'gray 38' :  (97,97,97) , 'gray 37' :  (94,94,94) , 'gray 36' :  (92,92,92) , 'gray 35' :  (89,89,89) , 'gray 34' :  (87,87,87) , 'gray 33' :  (84,84,84) , 'gray 32' :  (82,82,82) , 'gray 31' :  (79,79,79) , 'gray 30' :  (77,77,77) , 'gray 29' :  (74,74,74) , 'gray 28' :  (71,71,71) , 'gray 27' :  (69,69,69) , 'gray 26' :  (66,66,66) , 'gray 25' :  (64,64,64) , 'gray 24' :  (61,61,61) , 'gray 23' :  (59,59,59) , 'gray 22' :  (56,56,56) , 'gray 21' :  (54,54,54) , 'gray 20' :  (51,51,51) , 'gray 19' :  (48,48,48) , 'gray 18' :  (46,46,46) , 'gray 17' :  (43,43,43) , 'gray 16' :  (41,41,41) , 'gray 15' :  (38,38,38) , 'gray 14' :  (36,36,36) , 'gray 13' :  (33,33,33) , 'gray 12' :  (31,31,31) , 'gray 11' :  (28,28,28) , 'gray 10' :  (26,26,26) , 'gray 9' :  (23,23,23) , 'gray 8' :  (20,20,20) , 'gray 7' :  (18,18,18) , 'gray 6' :  (15,15,15) , 'gray 5' :  (13,13,13) , 'gray 4' :  (10,10,10) , 'gray 3' :  (8,8,8) , 'gray 2' :  (5,5,5) , 'gray 1' :  (3,3,3)
}

custom_color_family = {
    'indian red' : 'maroon' , 'crimson' : 'red' , 'lightpink' : 'light pink' , 'lightpink 1' : 'light pink' , 'lightpink 2' : 'light pink' , 'lightpink 3' : 'rose' , 'lightpink 4' : 'rose' , 'pink' : 'light pink' , 'pink 1' : 'light pink' , 'pink 2' : 'pink' , 'pink 3' : 'rose' , 'pink 4' : 'rose' , 'palevioletred' : 'pink' , 'palevioletred 1' : 'pink' , 'palevioletred 2' : 'pink' , 'palevioletred 3' : 'rose' , 'palevioletred 4' : 'rose' , 'lavenderblush 1 (lavenderblush)' : 'light pink' , 'lavenderblush 2' : 'white/off-white' , 'lavenderblush 3' : 'gray' , 'lavenderblush 4' : 'dark gray' , 'violetred 1' : 'hot pink' , 'violetred 2' : 'hot pink' , 'violetred 3' : 'deep pink' , 'violetred 4' : 'burgundy' , 'hotpink' : 'hot pink' , 'hotpink 1' : 'hot pink' , 'hotpink 2' : 'hot pink' , 'hotpink 3' : 'rose' , 'hotpink 4' : 'burgundy' , 'raspberry' : 'burgundy' , 'deeppink 1 (deeppink)' : 'hot pink' , 'deeppink 2' : 'hot pink' , 'deeppink 3' : 'deep pink' , 'deeppink 4' : 'burgundy' , 'maroon 1' : 'hot pink' , 'maroon 2' : 'hot pink' , 'maroon 3' : 'deep pink' , 'maroon 4' : 'burgundy' , 'mediumvioletred' : 'deep pink' , 'violetred' : 'deep pink' , 'orchid' : 'purple' , 'orchid 1' : 'lilac' , 'orchid 2' : 'lilac' , 'orchid 3' : 'purple' , 'orchid 4' : 'purple' , 'thistle' : 'lilac' , 'thistle 1' : 'lilac' , 'thistle 2' : 'lilac' , 'thistle 3' : 'lilac' , 'thistle 4' : 'dark gray' , 'plum 1' : 'lilac' , 'plum 2' : 'lilac' , 'plum 3' : 'lilac' , 'plum 4' : 'purple' , 'plum' : 'lilac' , 'violet' : 'lilac' , 'magenta (fuchsia*)' : 'magenta/fuchsia' , 'magenta 2' : 'magenta/fuchsia' , 'magenta 3' : 'magenta/fuchsia' , 'magenta 4 (darkmagenta)' : 'purple' , 'purple*' : 'purple' , 'mediumorchid' : 'purple' , 'mediumorchid 1' : 'purple' , 'mediumorchid 2' : 'purple' , 'mediumorchid 3' : 'purple' , 'mediumorchid 4' : 'purple' , 'darkviolet' : 'purple' , 'darkorchid' : 'purple' , 'darkorchid 1' : 'purple' , 'darkorchid 2' : 'purple' , 'darkorchid 3' : 'purple' , 'darkorchid 4' : 'purple' , 'indigo' : 'purple' , 'blueviolet' : 'purple' , 'purple 1' : 'purple' , 'purple 2' : 'purple' , 'purple 3' : 'purple' , 'purple 4' : 'purple' , 'mediumpurple' : 'voilet' , 'mediumpurple 1' : 'voilet' , 'mediumpurple 2' : 'voilet' , 'mediumpurple 3' : 'voilet' , 'mediumpurple 4' : 'voilet' , 'darkslateblue' : 'voilet' , 'lightslateblue' : 'voilet' , 'mediumslateblue' : 'voilet' , 'slateblue' : 'voilet' , 'slateblue 1' : 'voilet' , 'slateblue 2' : 'voilet' , 'slateblue 3' : 'voilet' , 'slateblue 4' : 'voilet' , 'ghostwhite' : 'white/off-white' , 'lavender' : 'white/off-white' , 'blue*' : 'blue' , 'blue 2' : 'blue' , 'blue 3 (mediumblue)' : 'navy' , 'blue 4 (darkblue)' : 'navy' , 'navy*' : 'navy' , 'midnightblue' : 'navy' , 'cobalt' : 'royal blue' , 'royalblue' : 'royal blue' , 'royalblue 1' : 'royal blue' , 'royalblue 2' : 'royal blue' , 'royalblue 3' : 'royal blue' , 'royalblue 4' : 'navy' , 'cornflowerblue' : 'royal blue' , 'lightsteelblue' : 'light blue' , 'lightsteelblue 1' : 'light blue' , 'lightsteelblue 2' : 'light blue' , 'lightsteelblue 3' : 'sky blue' , 'lightsteelblue 4' : 'dark gray' , 'lightslategray' : 'dark gray' , 'slategray' : 'dark gray' , 'slategray 1' : 'light blue' , 'slategray 2' : 'light blue' , 'slategray 3' : 'sky blue' , 'slategray 4' : 'dark gray' , 'dodgerblue 1 (dodgerblue)' : 'blue' , 'dodgerblue 2' : 'blue' , 'dodgerblue 3' : 'blue' , 'dodgerblue 4' : 'blue' , 'aliceblue' : 'white/off-white' , 'steelblue' : 'blue' , 'steelblue 1' : 'sky blue' , 'steelblue 2' : 'sky blue' , 'steelblue 3' : 'sky blue' , 'steelblue 4' : 'sky blue' , 'lightskyblue' : 'light blue' , 'lightskyblue 1' : 'light blue' , 'lightskyblue 2' : 'light blue' , 'lightskyblue 3' : 'sky blue' , 'lightskyblue 4' : 'dark gray' , 'skyblue 1' : 'sky blue' , 'skyblue 2' : 'sky blue' , 'skyblue 3' : 'sky blue' , 'skyblue 4' : 'sky blue' , 'skyblue' : 'sky blue' , 'deepskyblue 1 (deepskyblue)' : 'sky blue' , 'deepskyblue 2' : 'sky blue' , 'deepskyblue 3' : 'sky blue' , 'deepskyblue 4' : 'sky blue' , 'peacock' : 'sky blue' , 'lightblue' : 'light blue' , 'lightblue 1' : 'light blue' , 'lightblue 2' : 'light blue' , 'lightblue 3' : 'light blue' , 'lightblue 4' : 'dark gray' , 'powderblue' : 'aqua' , 'cadetblue 1' : 'aqua' , 'cadetblue 2' : 'aqua' , 'cadetblue 3' : 'turquoise' , 'cadetblue 4' : 'teal blue' , 'turquoise 1' : 'aqua' , 'turquoise 2' : 'aqua' , 'turquoise 3' : 'turquoise' , 'turquoise 4' : 'teal blue' , 'cadetblue' : 'teal blue' , 'darkturquoise' : 'turquoise' , 'azure 1 (azure)' : 'white/off-white' , 'azure 2' : 'aqua' , 'azure 3' : 'gray' , 'azure 4' : 'dark gray' , 'lightcyan 1 (lightcyan)' : 'aqua' , 'lightcyan 2' : 'aqua' , 'lightcyan 3' : 'aqua' , 'lightcyan 4' : 'dark gray' , 'paleturquoise 1' : 'aqua' , 'paleturquoise 2 (paleturquoise)' : 'aqua' , 'paleturquoise 3' : 'turquoise' , 'paleturquoise 4' : 'teal blue' , 'darkslategray' : 'teal blue' , 'darkslategray 1' : 'aqua' , 'darkslategray 2' : 'aqua' , 'darkslategray 3' : 'turquoise' , 'darkslategray 4' : 'teal blue' , 'cyan / aqua*' : 'aqua' , 'cyan 2' : 'aqua' , 'cyan 3' : 'turquoise' , 'cyan 4 (darkcyan)' : 'teal blue' , 'teal*' : 'teal blue' , 'mediumturquoise' : 'turquoise' , 'lightseagreen' : 'teal blue' , 'manganeseblue' : 'teal blue' , 'turquoise' : 'turquoise' , 'coldgrey' : 'dark gray' , 'turquoiseblue' : 'green' , 'aquamarine 1 (aquamarine)' : 'aquamarine/mint' , 'aquamarine 2' : 'aquamarine/mint' , 'aquamarine 3 (mediumaquamarine)' : 'aquamarine/mint' , 'aquamarine 4' : 'dark green' , 'mediumspringgreen' : 'neon green' , 'mintcream' : 'white/off-white' , 'springgreen' : 'neon green' , 'springgreen 1' : 'green' , 'springgreen 2' : 'green' , 'springgreen 3' : 'dark green' , 'mediumseagreen' : 'green' , 'seagreen 1' : 'neon green' , 'seagreen 2' : 'aquamarine/mint' , 'seagreen 3' : 'green' , 'seagreen 4 (seagreen)' : 'dark green' , 'emeraldgreen' : 'green' , 'mint' : 'aquamarine/mint' , 'cobaltgreen' : 'dark green' , 'honeydew 1 (honeydew)' : 'white/off-white' , 'honeydew 2' : 'aquamarine/mint' , 'honeydew 3' : 'aquamarine/mint' , 'honeydew 4' : 'dark gray' , 'darkseagreen' : 'light olive/light khaki' , 'darkseagreen 1' : 'aquamarine/mint' , 'darkseagreen 2' : 'aquamarine/mint' , 'darkseagreen 3' : 'light olive/light khaki' , 'darkseagreen 4' : 'olive/khaki' , 'palegreen' : 'aquamarine/mint' , 'palegreen 1' : 'neon green' , 'palegreen 2 (lightgreen)' : 'aquamarine/mint' , 'palegreen 3' : 'green' , 'palegreen 4' : 'olive/khaki' , 'limegreen' : 'green' , 'forestgreen' : 'dark green' , 'green 1 (lime*)' : 'neon green' , 'green 2' : 'neon green' , 'green 3' : 'green' , 'green 4' : 'dark green' , 'green*' : 'dark green' , 'darkgreen' : 'dark green' , 'sapgreen' : 'dark green' , 'lawngreen' : 'neon green' , 'chartreuse 1 (chartreuse)' : 'neon green' , 'chartreuse 2' : 'neon green' , 'chartreuse 3' : 'green' , 'chartreuse 4' : 'dark green' , 'greenyellow' : 'neon green' , 'darkolivegreen 1' : 'neon green' , 'darkolivegreen 2' : 'neon green' , 'darkolivegreen 3' : 'olive/khaki' , 'darkolivegreen 4' : 'olive/khaki' , 'darkolivegreen' : 'olive/khaki' , 'olivedrab' : 'olive/khaki' , 'olivedrab 1' : 'neon green' , 'olivedrab 2' : 'neon green' , 'olivedrab 3 (yellowgreen)' : 'light olive/light khaki' , 'olivedrab 4' : 'olive/khaki' , 'ivory 1 (ivory)' : 'white/off-white' , 'ivory 2' : 'yellow/lemon yellow' , 'ivory 3' : 'white/off-white' , 'ivory 4' : 'dark gray' , 'beige' : 'white/off-white' , 'lightyellow 1 (lightyellow)' : 'white/off-white' , 'lightyellow 2' : 'yellow/lemon yellow' , 'lightyellow 3' : 'gray' , 'lightyellow 4' : 'dark gray' , 'lightgoldenrodyellow' : 'yellow/lemon yellow' , 'yellow 1 (yellow*)' : 'yellow/lemon yellow' , 'yellow 2' : 'yellow/lemon yellow' , 'yellow 3' : 'light olive/light khaki' , 'yellow 4' : 'olive/khaki' , 'warmgrey' : 'olive/khaki' , 'olive*' : 'olive/khaki' , 'darkkhaki' : 'light olive/light khaki' , 'khaki 1' : 'yellow/lemon yellow' , 'khaki 2' : 'light olive/light khaki' , 'khaki 3' : 'light olive/light khaki' , 'khaki 4' : 'olive/khaki' , 'khaki' : 'light olive/light khaki' , 'palegoldenrod' : 'light olive/light khaki' , 'lemonchiffon 1 (lemonchiffon)' : 'yellow/lemon yellow' , 'lemonchiffon 2' : 'light olive/light khaki' , 'lemonchiffon 3' : 'gray' , 'lemonchiffon 4' : 'dark gray' , 'lightgoldenrod 1' : 'yellow/lemon yellow' , 'lightgoldenrod 2' : 'yellow/lemon yellow' , 'lightgoldenrod 3' : 'yellow/lemon yellow' , 'lightgoldenrod 4' : 'olive/khaki' , 'banana' : 'yellow/lemon yellow' , 'gold 1 (gold)' : 'mustard' , 'gold 2' : 'mustard' , 'gold 3' : 'ochre' , 'gold 4' : 'olive/khaki' , 'cornsilk 1 (cornsilk)' : 'white/off-white' , 'cornsilk 2' : 'yellow/lemon yellow' , 'cornsilk 3' : 'gray' , 'cornsilk 4' : 'dark gray' , 'goldenrod' : 'ochre' , 'goldenrod 1' : 'mustard' , 'goldenrod 2' : 'mustard' , 'goldenrod 3' : 'ochre' , 'goldenrod 4' : 'olive/khaki' , 'darkgoldenrod' : 'ochre' , 'darkgoldenrod 1' : 'mustard' , 'darkgoldenrod 2' : 'mustard' , 'darkgoldenrod 3' : 'ochre' , 'darkgoldenrod 4' : 'olive/khaki' , 'orange 1 (orange)' : 'orange' , 'orange 2' : 'ochre' , 'orange 3' : 'ochre' , 'orange 4' : 'dark browns' , 'floralwhite' : 'white/off-white' , 'oldlace' : 'white/off-white' , 'wheat' : 'tan/beige' , 'wheat 1' : 'tan/beige' , 'wheat 2' : 'tan/beige' , 'wheat 3' : 'tan/beige' , 'wheat 4' : 'dark browns' , 'moccasin' : 'tan/beige' , 'papayawhip' : 'white/off-white' , 'blanchedalmond' : 'white/off-white' , 'navajowhite 1 (navajowhite)' : 'tan/beige' , 'navajowhite 2' : 'tan/beige' , 'navajowhite 3' : 'tan/beige' , 'navajowhite 4' : 'dark browns' , 'eggshell' : 'white/off-white' , 'tan' : 'tan/beige' , 'brick' : 'dark browns' , 'cadmiumyellow' : 'orange' , 'antiquewhite' : 'white/off-white' , 'antiquewhite 1' : 'white/off-white' , 'antiquewhite 2' : 'peach' , 'antiquewhite 3' : 'gray' , 'antiquewhite 4' : 'dark gray' , 'burlywood' : 'tan/beige' , 'burlywood 1' : 'tan/beige' , 'burlywood 2' : 'tan/beige' , 'burlywood 3' : 'tan/beige' , 'burlywood 4' : 'dark browns' , 'bisque 1 (bisque)' : 'peach' , 'bisque 2' : 'peach' , 'bisque 3' : 'tan/beige' , 'bisque 4' : 'dark gray' , 'melon' : 'tan/beige' , 'carrot' : 'ochre' , 'darkorange' : 'orange' , 'darkorange 1' : 'orange' , 'darkorange 2' : 'orange' , 'darkorange 3' : 'dark browns' , 'darkorange 4' : 'dark browns' , 'orange' : 'orange' , 'tan 1' : 'tan/beige' , 'tan 2' : 'tan/beige' , 'tan 3 (peru)' : 'dark browns' , 'tan 4' : 'dark browns' , 'linen' : 'white/off-white' , 'peachpuff 1 (peachpuff)' : 'peach' , 'peachpuff 2' : 'peach' , 'peachpuff 3' : 'tan/beige' , 'peachpuff 4' : 'dark browns' , 'seashell 1 (seashell)' : 'peach' , 'seashell 2' : 'peach' , 'seashell 3' : 'gray' , 'seashell 4' : 'dark gray' , 'sandybrown' : 'tan/beige' , 'rawsienna' : 'dark browns' , 'chocolate' : 'dark browns' , 'chocolate 1' : 'orange' , 'chocolate 2' : 'orange' , 'chocolate 3' : 'dark browns' , 'chocolate 4 (saddlebrown)' : 'dark browns' , 'ivoryblack' : 'black' , 'flesh' : 'coral/salmon' , 'cadmiumorange' : 'orange' , 'burntsienna' : 'dark browns' , 'sienna' : 'dark browns' , 'sienna 1' : 'coral/salmon' , 'sienna 2' : 'coral/salmon' , 'sienna 3' : 'dark browns' , 'sienna 4' : 'dark browns' , 'lightsalmon 1 (lightsalmon)' : 'coral/salmon' , 'lightsalmon 2' : 'coral/salmon' , 'lightsalmon 3' : 'dark browns' , 'lightsalmon 4' : 'dark browns' , 'coral' : 'coral/salmon' , 'orangered 1 (orangered)' : 'red' , 'orangered 2' : 'red' , 'orangered 3' : 'coral/salmon' , 'orangered 4' : 'maroon' , 'sepia' : 'dark browns' , 'darksalmon' : 'coral/salmon' , 'salmon 1' : 'coral/salmon' , 'salmon 2' : 'coral/salmon' , 'salmon 3' : 'dark browns' , 'salmon 4' : 'dark browns' , 'coral 1' : 'coral/salmon' , 'coral 2' : 'coral/salmon' , 'coral 3' : 'coral/salmon' , 'coral 4' : 'maroon' , 'burntumber' : 'maroon' , 'tomato 1 (tomato)' : 'coral/salmon' , 'tomato 2' : 'coral/salmon' , 'tomato 3' : 'coral/salmon' , 'tomato 4' : 'maroon' , 'salmon' : 'coral/salmon' , 'mistyrose 1 (mistyrose)' : 'peach' , 'mistyrose 2' : 'peach' , 'mistyrose 3' : 'peach' , 'mistyrose 4' : 'dark gray' , 'snow 1 (snow)' : 'white/off-white' , 'snow 2' : 'white/off-white' , 'snow 3' : 'gray' , 'snow 4' : 'dark gray' , 'rosybrown' : 'rose' , 'rosybrown 1' : 'peach' , 'rosybrown 2' : 'peach' , 'rosybrown 3' : 'rose' , 'rosybrown 4' : 'dark browns' , 'lightcoral' : 'coral/salmon' , 'indianred' : 'coral/salmon' , 'indianred 1' : 'coral/salmon' , 'indianred 2' : 'coral/salmon' , 'indianred 4' : 'maroon' , 'indianred 3' : 'coral/salmon' , 'brown' : 'maroon' , 'brown 1' : 'orange' , 'brown 2' : 'red' , 'brown 3' : 'red' , 'brown 4' : 'maroon' , 'firebrick' : 'maroon' , 'firebrick 1' : 'red' , 'firebrick 2' : 'red' , 'firebrick 3' : 'red' , 'firebrick 4' : 'maroon' , 'red 1 (red*)' : 'red' , 'red 2' : 'red' , 'red 3' : 'red' , 'red 4 (darkred)' : 'maroon' , 'maroon*' : 'maroon' , 'sgi beet' : 'purple' , 'sgi slateblue' : 'voilet' , 'sgi lightblue' : 'sky blue' , 'sgi teal' : 'teal blue' , 'sgi chartreuse' : 'green' , 'sgi olivedrab' : 'olive/khaki' , 'sgi brightgray' : 'light olive/light khaki' , 'sgi salmon' : 'coral/salmon' , 'sgi darkgray' : 'dark gray' , 'sgi gray 12' : 'black' , 'sgi gray 16' : 'black' , 'sgi gray 32' : 'dark gray' , 'sgi gray 36' : 'dark gray' , 'sgi gray 52' : 'dark gray' , 'sgi gray 56' : 'gray' , 'sgi lightgray' : 'gray' , 'sgi gray 72' : 'gray' , 'sgi gray 76' : 'gray' , 'sgi gray 92' : 'white/off-white' , 'sgi gray 96' : 'white/off-white' , 'white*' : 'white/off-white' , 'white smoke (gray 96)' : 'white/off-white' , 'gainsboro' : 'light gray' , 'lightgrey' : 'gray' , 'silver*' : 'gray' , 'darkgray' : 'gray' , 'gray*' : 'dark gray' , 'dimgray (gray 42)' : 'dark gray' , 'black*' : 'black' , 'gray 99' : 'white/off-white' , 'gray 98' : 'white/off-white' , 'gray 97' : 'white/off-white' , 'white smoke (gray 96)' : 'white/off-white' , 'gray 95' : 'white/off-white' , 'gray 94' : 'white/off-white' , 'gray 93' : 'white/off-white' , 'gray 92' : 'white/off-white' , 'gray 91' : 'white/off-white' , 'gray 90' : 'white/off-white' , 'gray 89' : 'white/off-white' , 'gray 88' : 'white/off-white' , 'gray 87' : 'light gray' , 'gray 86' : 'light gray' , 'gray 85' : 'light gray' , 'gray 84' : 'light gray' , 'gray 83' : 'light gray' , 'gray 82' : 'light gray' , 'gray 81' : 'light gray' , 'gray 80' : 'gray' , 'gray 79' : 'gray' , 'gray 78' : 'gray' , 'gray 77' : 'gray' , 'gray 76' : 'gray' , 'gray 75' : 'gray' , 'gray 74' : 'gray' , 'gray 73' : 'gray' , 'gray 72' : 'gray' , 'gray 71' : 'gray' , 'gray 70' : 'gray' , 'gray 69' : 'gray' , 'gray 68' : 'gray' , 'gray 67' : 'gray' , 'gray 66' : 'gray' , 'gray 65' : 'gray' , 'gray 64' : 'gray' , 'gray 63' : 'gray' , 'gray 62' : 'gray' , 'gray 61' : 'gray' , 'gray 60' : 'gray' , 'gray 59' : 'dark gray' , 'gray 58' : 'dark gray' , 'gray 57' : 'dark gray' , 'gray 56' : 'dark gray' , 'gray 55' : 'dark gray' , 'gray 54' : 'dark gray' , 'gray 53' : 'dark gray' , 'gray 52' : 'dark gray' , 'gray 51' : 'dark gray' , 'gray 50' : 'dark gray' , 'gray 49' : 'dark gray' , 'gray 48' : 'dark gray' , 'gray 47' : 'dark gray' , 'gray 46' : 'dark gray' , 'gray 45' : 'dark gray' , 'gray 44' : 'dark gray' , 'gray 43' : 'dark gray' , 'gray 42' : 'dark gray' , 'dimgray (gray 42)' : 'dark gray' , 'gray 40' : 'dark gray' , 'gray 39' : 'charcoal' , 'gray 38' : 'charcoal' , 'gray 37' : 'charcoal' , 'gray 36' : 'charcoal' , 'gray 35' : 'charcoal' , 'gray 34' : 'charcoal' , 'gray 33' : 'charcoal' , 'gray 32' : 'charcoal' , 'gray 31' : 'charcoal' , 'gray 30' : 'charcoal' , 'gray 29' : 'charcoal' , 'gray 28' : 'charcoal' , 'gray 27' : 'charcoal' , 'gray 26' : 'charcoal' , 'gray 25' : 'charcoal' , 'gray 24' : 'charcoal' , 'gray 23' : 'charcoal' , 'gray 22' : 'charcoal' , 'gray 21' : 'charcoal' , 'gray 20' : 'charcoal' , 'gray 19' : 'charcoal' , 'gray 18' : 'charcoal' , 'gray 17' : 'black' , 'gray 16' : 'black' , 'gray 15' : 'black' , 'gray 14' : 'black' , 'gray 13' : 'black' , 'gray 12' : 'black' , 'gray 11' : 'black' , 'gray 10' : 'black' , 'gray 9' : 'black' , 'gray 8' : 'black' , 'gray 7' : 'black' , 'gray 6' : 'black' , 'gray 5' : 'black' , 'gray 4' : 'black' , 'gray 3' : 'black' , 'gray 2' : 'black' , 'gray 1' : 'black'
}

def ciede2000_distance(color1, color2):
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    lab1 = convert_rgb_to_lab(color1)
    lab2 = convert_rgb_to_lab(color2)
    distance = delta_e_cie2000(lab1, lab2)
    return distance


def convert_rgb_to_lab(color_rgb):
    r, g, b = color_rgb
    rgb_color = sRGBColor(r, g, b)
    lab_color = convert_color(rgb_color, LabColor)
    return lab_color

def get_closest_color_name_using_ciede2000_distance(color_bgr):
    color_rgb = convert_bgr_to_rgb(color_bgr)
    min_distance = float('inf')
    closest_color = None

    for color_name, color_value in custom_colors.items():
        distance = ciede2000_distance(color_rgb, color_value)

        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    return closest_color


def get_color_family_from_color_name(closest_color):
  return custom_color_family[closest_color]

def convert_bgr_to_rgb(bgr_tuple):
  return (bgr_tuple[2], bgr_tuple[1], bgr_tuple[0])


directory_path = '/drive/MyDrive/Fashion-Analytics-Project/t-shirt-articles_100-images'
# Get a list of all files in the directory
directory_to_read = '/drive/MyDrive/Fashion-Analytics-Project/t-shirt-articles_100-images/test_images'
image_names_list = os.listdir(directory_to_read) 
# print(image_names_list)
# Create a new directory for saving the output images
marked_directory = os.path.join(directory_path, 'marked_images_grabcut_meanshift_edistance')
segmented_directory = os.path.join(directory_path, 'segmented_images_grabcut_meanshift_edistance')
enhanced_directory = os.path.join(directory_path, 'enhanced_images_grabcut_meanshift_edistance')
# os.mkdir(marked_directory)
# os.mkdir(segmented_directory)
# os.mkdir(enhanced_directory)

# empty list for data frames
image_names = []
rgb_code_tuples = []
color_names = []
color_family_names = []

for image_name in image_names_list:
  # getting segmentation using grabcut
  filepath = os.path.join(directory_to_read, image_name)
  image = cv2.imread(filepath)

  # enhance images before segmentation
  enhanced_image = enhance_image(image)
  # enhanced_path = os.path.join(enhanced_directory, 'enhanced_' + image_name)
  # cv2.imwrite(enhanced_path, enhanced_image)

  resized_image = change_resolution(enhanced_image)
  segmented_image = segment_garment(resized_image)
  # segmented_path = os.path.join(segmented_directory, 'segmented_' + image_name)
  # cv2.imwrite(segmented_path, segmented_image)

  
  get_color_BGR, marked_image = get_dominant_bgr_using_meanshift_brightness(segmented_image)
  # print(image_name ,get_color_BGR)
  # marked_path = os.path.join(marked_directory, 'marked_' + image_name)
  # Save the marked image with the dominant color region
  # cv2.imwrite(marked_path, marked_image)

  # color_name = convert_gbr_to_names_using_KDtree(tuple(get_color_BGR))
  color_name = get_closest_color_name_using_ciede2000_distance(tuple(get_color_BGR))
  # color_name = get_closest_color_name_using_distance(tuple(get_color_BGR))

  rgb_code_tuple = convert_bgr_to_rgb(tuple(get_color_BGR))
  

  color_family_name = get_color_family_from_color_name(color_name)
  print("rgb code and color_name and color_family: ", image_name,rgb_code_tuple,color_name, color_family_name)

  # saving output in file 

  # Append data for each iteration to the respective lists
  image_names.append(image_name)
  rgb_code_tuples.append(rgb_code_tuple)
  color_names.append(color_name)
  color_family_names.append(color_family_name)

  
  # Create DataFrame from the lists
  df = pd.DataFrame({
      'image_name': image_names,
      'rgb_code_tuple': rgb_code_tuples,
      'color_name': color_names,
      'color_family_name': color_family_names
  })



  #save output
  file_path = "/drive/MyDrive/Fashion-Analytics-Project/t-shirt-articles_100-images/color-family-ciede2000.xlsx"
  df.to_excel(file_path, index=False) 
