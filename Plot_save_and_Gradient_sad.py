from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

disp_range = [-60, 0]

left = Image.open("C:/Users/rmalinow/Code/Disparity_visualization/Cones_LEFT.tif")
right = Image.open("C:/Users/rmalinow/Code/Disparity_visualization/Cones_RIGHT.tif")


row = 200
k = 20


im1_bis = left.crop((85, 270, 115, 275)).resize((k*30, k*5), Image.NEAREST)
im2_bis = right.crop((40, 270, 70, 275)).resize((k*30, k*5), Image.NEAREST)

delta = 12

im1_bis = im1_bis.convert('RGB')
im1_draw = ImageDraw.Draw(im1_bis)
im1_draw.rectangle([(delta*k, 0), ((delta+5)*k, k*5)], outline="red", width=3)
im1_bis.save("D:/Users/rmalinow/Desktop/GIF_CNES/left.jpg")


im2_bis = im2_bis.convert('RGB')


def sad(w1, w2):
    return abs(w1-w2).sum()/np.product(w1.shape)


def census(w1, w2):
    w1_bool = w1 >= w1[2, 2]
    w2_bool = w2 >= w2[2, 2]

    return 25-(w1_bool == w2_bool).sum()/np.product(w1.shape)


def better_sad(w1, w2, grad1, grad2, theta=0.5, tau1=1000, tau2=1000):
    return (1 - theta) * np.min([sad(w1, w2), tau1]) + theta * np.min([sad(grad1, grad2), tau2])


cost_arr = np.array([np.nan for i in range(3, 12)])

grad_1 = signal.convolve2d(np.array(left), np.array([[-1, 0, 1]]), boundary="symm", mode='same')
grad_2 = signal.convolve2d(np.array(right), np.array([[-1, 0, 1]]), boundary="symm", mode='same')

# fig, ax = plt.subplots(2, 2, figsize=(6, 15))
# ax[0, 0].imshow(np.array(left), cmap='gray')
# ax[0, 1].imshow(np.array(right), cmap='gray')
# ax[1, 0].imshow(grad_1, cmap='gray')
# ax[1, 1].imshow(grad_2, cmap='gray')
# fig.show()
# plt.show()

for d in range(3, 12):
    im2_temp = im2_bis.copy()
    im2_draw = ImageDraw.Draw(im2_temp)
    im2_draw.rectangle((d*k, 0, (d+5) * k, k * 5), outline="red", width=3)
    im2_temp.save("D:/Users/rmalinow/Desktop/GIF_CNES/right%s.jpg" % (d-2))

    arr1 = np.array(left.crop((85, 270, 115, 275)))
    arr2 = np.array(right.crop((40, 270, 70, 275)))
    # cost_arr[d-3] = census(arr1[:, delta:delta+5], arr2[:, d:d+5])
    cost_arr[d-3] = better_sad(arr1[:, delta:delta+5], arr2[:, d:d+5], grad_1[270:275, delta:delta+5],  grad_2[270:275, d:d+5])

    #     PIL_image = Image.fromarray(np.uint8(np.hstack([arr1[:, delta:delta+5], arr2[:, d:d+5]]))).convert('L')
    #     PIL_image.show()

    plt.figure()
    plt.plot(range(3, 12), cost_arr, '-o')
    plt.ylim(55, 85)
    plt.xlim(2, 12)
    # plt.show()
    plt.savefig('D:/Users/rmalinow/Desktop/GIF_CNES/CNES_plot_%s.png' % (d-2))







