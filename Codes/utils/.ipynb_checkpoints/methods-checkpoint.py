# Copyright (c) 2021,2024 Yongkai Zhu <yongkai_zhu@hotmail.com>
# MIT License

import numpy as np
import psutil

from scipy import constants as C

def bytestoGB(size):
    size = size / 1000 / 1000 / 1000
    return size

def bytestoMB(size):
    size = size / 1000 / 1000
    return size

def check_mem(img_size):
    required_mem = img_size ** 2 * 8 / 1000000000
    mem = psutil.virtual_memory()
    available_mem = mem[1]
    available_mem = bytestoGB(available_mem)
    if available_mem < required_mem * 1.2:
        print('Ava: %.2f GB; Req: %.2f GB' %(available_mem, required_mem))
        raise ValueError('Pixel size is too small.')
    else:
        print('Ava: %.2f GB; Req: %.2f GB' %(available_mem, required_mem))
    return

def calc_Tb(flux, omega, freq):
    """
    flux -> brightness temperature [K]
    flux: [Jy]
    omega: [arcsec^2]
    freq: [MHz]
    """
    c = C.c   # [m/s]
    k_B = C.Boltzmann
    arcsec2rad = C.pi / 180.0 / 3600.0
    freq = freq * 1e6    # [MHz]->[Hz]
    omega = omega * (arcsec2rad * arcsec2rad)
    Sb = flux * 1e-26 / omega
    return 0.5 * Sb * c * c / (freq * freq * k_B)

def in_ellipse(x, y, f1x, f1y, f2x, f2y, a):
    """
    Return whether a pixel is in a ellipse.
    x, y: Coordinates of a pixel.
    f1x, f1y, f2x, f2y: Coordinates of the left and right focus points of the ellipse.
    a: Semimajor axis.
    """
    t1 = np.sqrt((x - f1x) ** 2 + (y - f1y) ** 2)
    t2 = np.sqrt((x - f2x) ** 2 + (y - f2y) ** 2)
    return (t1 + t2) < 2 * a

def draw_ellipse(shapes, center, a, b, phi):
    points1 = np.arange(0, shapes[0])
    points2 = np.arange(0, shapes[1])
    x, y = np.meshgrid(points1, points2)
    x_center, y_center = center
    x_rotation = (x-x_center)*np.cos(phi)+(y-y_center)*np.sin(phi)
    y_rotation = (x-x_center)*np.sin(phi)-(y-y_center)*np.cos(phi)
    distances = ((x_rotation)**2)/(a**2)+((y_rotation)**2)/(b**2)
    img = (distances <= 1).astype(int)
    return img

def boundary_conditions(lower_limit, upper_limit, *p):
    result = True
    for i in p:
        if np.sum(i<lower_limit) > 0 or np.sum(i>=upper_limit) > 0:
            result = False
            break
    return result

def skymodel_to_img(skymodel, img_size):
    img_size = img_size
    img = np.zeros([img_size, img_size])
    for source in skymodel:
        x = source[:, 0]
        y = source[:, 1]
        for i in range(len(x)):
            img[int(x[i]), int(y[i])] += source[i, 2]
    return img