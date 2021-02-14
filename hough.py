import numpy as np 
from skimage.draw import line
from skimage import feature
from skimage.color import gray2rgb
from numba import jit
import seaborn as sns
import matplotlib.pylab as plt

@jit(nopython=True)
def hough(image):
    dark_pixels = np.where(image == 0)
    coordinates = list(zip(dark_pixels[0], dark_pixels[1]))

    n = image.shape[0]
    m = image.shape[1]

    rho = np.arange(-np.sqrt(n*n + m*m), np.sqrt(n*n + m*m), 1)
    theta = np.arange(0,180,1)
    
    sin = np.sin(np.deg2rad(theta))
    cos = np.cos(np.deg2rad(theta))

    accumulator = np.zeros((len(rho), len(theta)), dtype=np.uint8)
    
    for p in range(len(coordinates)):
        for t in range(len(theta)):
            r = int(round(coordinates[p][1] * cos[t] + coordinates[p][0] * sin[t]))
            accumulator[r, t] += 1

    return accumulator

def lines(accumulator, image):
    n = image.shape[0]
    m = image.shape[1]

    edge_pixels = np.where(accumulator >= np.max(accumulator))
    coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

    for i in range(0, len(coordinates)):
        print(i)
        cos_theta = np.cos(np.deg2rad(coordinates[i][1]))
        sin_theta = np.sin(np.deg2rad(coordinates[i][1]))
        y1 = int(1)
        y2 = int(image.shape[0]-1)
        x1 = int(coordinates[i][0] / cos_theta) % m
        x2 = int((coordinates[i][0] - n*sin_theta)/cos_theta) % m

        rr,cc = line(y1,x1,y2,x2)
        cc[cc >= m] = m-1 
        cc[cc <= 0] = 0
        image[rr,cc] = [255,0,0]
    
    return image
    

from skimage.io import imread,imsave


image_gray = imread('/home/pavel/HoughTransform/photo.png', as_gray=True)

accumulator = hough(image_gray)

ax = sns.heatmap(accumulator, linewidth=1E-7)
plt.show()

image_res = lines(hough(image_gray), gray2rgb(image_gray))
imsave('/home/pavel/HoughTransform/lines.png', image_res)
