import math
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, y, sigma):
    hasil = (1/(2*math.pi*sigma**2)) * math.exp(-((x**2+y**2)/(2*sigma**2)))
    return hasil

def create_kernel_gaussian(panjang, sigma):
    hasil = np.zeros((panjang,panjang))
    tengah = panjang//2
    for i in range(-tengah,tengah+1):
        for j in range(-tengah,tengah+1):
            hasil[i+tengah][j+tengah] = gaussian(i,j,sigma)
    return hasil

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def convolution2d(image, kernel):
    p, l = image.shape
    p2, l2 = kernel.shape
    p_hasil = p - p2 + 1
    l_hasil = l - l2 + 1
    hasil = np.zeros((p_hasil, l_hasil))
    for i in range(p_hasil):
        for j in range(l_hasil):
            hasil[i][j] = np.sum(image[i:i+p2, j:j+l2]*kernel)
    return hasil

def non_max_suppression(g, sudut):
    p, l = g.shape
    hasil = np.zeros((p, l))
    for i in range(p):
        for j in range(l):
            try:
                x = g[i][j]
                if (0 <= sudut[i][j] < 22.5) or (157.5 <= sudut[i][j] <= 180):
                    y = g[i][j+1]
                    z = g[i][j-1]
                elif (22.5 <= sudut[i][j] < 67.5):
                    y = g[i+1][j-1]
                    z = g[i-1][j+1]
                elif (67.5 <= sudut[i][j] < 112.5):
                    y = g[i+1][j]
                    z = g[i-1][j]
                elif (112.5 <= sudut[i][j] < 157.5):
                    y = g[i-1][j-1]
                    z = g[i+1][j+1]
                if (x >= y) and (x >= z):
                    hasil[i][j] = x
                else:
                    hasil[i][j] = 0
            except:
                pass
    return hasil

def double_threshold(g, lowratio=0.05, highratio= 0.09):
    high = g.max()*highratio
    low = g.max()*lowratio
##    low = high*lowratio
    p, l = g.shape
    hasil = np.zeros((p,l))
    weak = 25
    strong = 255
    for i in range(p):
        for j in range(l):
            if g[i][j] >= high:
                hasil[i][j] = strong
            elif g[i][j] < low:
                hasil[i][j] = 0
            else:
                hasil[i][j] = weak
    return hasil

def edge_tracking_hysteresis(g):
    weak = 25
    strong = 255
    p, l = g.shape
    for i in range(p):
        for j in range(l):
            if (g[i][j] == weak):
                try:
                    if (g[i][j+1] == strong) or (g[i][j-1] == strong) or (g[i+1][j] == strong) or (g[i-1][j] == strong) or (g[i+1][j+1] == strong) or (g[i-1][j+1] == strong) or (g[i+1][j-1] == strong) or (g[i-1][j-1] == strong):
                        g[i][j] = strong
                    else:
                        g[i][j] = 0
                except:
                    pass
    return g

kernel_gaussian = create_kernel_gaussian(5, 1.4)
print(kernel_gaussian)
##img = plt.imread('bangun.png')
##img = plt.imread('2.jpg')
img = plt.imread('3x4.jpg')
gray = rgb2gray(img)
result_gaussian = convolution2d(gray, kernel_gaussian)

sobel_gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
gx = convolution2d(result_gaussian, sobel_gx)
gy = convolution2d(result_gaussian, sobel_gy)
g = np.sqrt(np.square(gx) + np.square(gy))
##sudut = np.arctan(abs(gy)/abs(gx))
##sudut = np.arctan2(abs(gy), abs(gx))
sudut = np.arctan2(gy, gx)*(180/np.pi)
sudut[sudut<0] += 180

##sudut = np.arctan2(gy, gx)
##sudut = 180 + (180/np.pi)*sudut
hasil_non_max = non_max_suppression(g, sudut)
hasil_double_threshold = double_threshold(hasil_non_max, 0.02, 0.08)
final = edge_tracking_hysteresis(hasil_double_threshold)

f, arr = plt.subplots(2,3)
arr[0][0].imshow(gray, 'gray')
arr[0][1].imshow(result_gaussian, 'gray')
arr[0][2].imshow(g, 'gray')
arr[1][0].imshow(hasil_non_max, 'gray')
arr[1][1].imshow(hasil_double_threshold, 'gray')
arr[1][2].imshow(final, 'gray')
plt.show()
