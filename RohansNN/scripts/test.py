import tensorflow as tf
import numpy as np
import pylab as plt
import os
from ppo.RealTimeTracking.predict import Memory


def generate_movies(n_samples=50, n_frames=15):
    row = 100
    col = 120
    noisy_movies = np.zeros((n_samples, n_frames, row, col,3), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col,3), dtype=np.float)

    for i in range(n_samples):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)

        for j in range(n):
            # Initial position
            xstart = np.random.randint(20, 100)
            ystart = np.random.randint(20, 80)
            # Direction of motion
            directionx = np.random.randint(0, 10) - 1
            directiony = np.random.randint(0, 10) - 1

            # Size of the square
            w = np.random.randint(2, 10)

            pix_val = np.random.randint(0,255)
            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[i, t, x_shift - w: x_shift + w,
                             y_shift - w: y_shift + w,0] += pix_val

                # Make it more robust by adding noise.
                # The idea is that if during inference,
                # the value of the pixel is not exactly one,
                # we need to train the network to be robust and still
                # consider it as a pixel belonging to a square.
                if np.random.randint(0, 2):
                    noise_f = (-1)**np.random.randint(0, 2)
                    noisy_movies[i, t,
                                 x_shift - w - 1: x_shift + w + 1,
                                 y_shift - w - 1: y_shift + w + 1,0] += noise_f * 0.1
                    noisy_movies[i, t,
                                 x_shift - w - 1: x_shift + w + 1,
                                 y_shift - w - 1: y_shift + w + 1,1] += noise_f * 0.1
                    noisy_movies[i, t,
                                 x_shift - w - 1: x_shift + w + 1,
                                 y_shift - w - 1: y_shift + w + 1,2] += noise_f * 0.1

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[i, t, x_shift - w: x_shift + w, y_shift - w: y_shift + w,0] += pix_val
                shifted_movies[i, t, x_shift - w: x_shift + w, y_shift - w: y_shift + w,1] += pix_val
                shifted_movies[i, t, x_shift - w: x_shift + w, y_shift - w: y_shift + w,2] += pix_val

    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, ::, 20:80, 20:100,::]
    shifted_movies = shifted_movies[::, ::, 20:80, 20:100,::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray



if __name__ == "__main__":
    noisy_movies, shifted_movies = generate_movies()

    which = 40

    track = noisy_movies[which][:3, ::, ::, ::]

    detargs = {"conf" : 0.8, 'nms' : 0.4}
    trackargs = {"update_ms" : 5,"min_conf" : 0.2, "nms" : 1.0, "min_det_ht" : 0, "max_cos_dist" : 0.2, "nn_budget" : 100}
    imagenetwork = Memory(detargs, trackargs)


    image_out_NN = []
    for i in range(len(track)):
        image_in_NN = track[i,::,::,::]
        print(image_in_NN.shape)

        output = imagenetwork.predict(image_in_NN)
        output = rgb2gray(output)
        image_out_NN.append(output)
        
        print(output.shape)
        print('--------------')
    image_out_NN = np.array(image_out_NN)

    predict_image = seq.predict(track)
    ground_truth = noisy_movies[which][2, ::, ::, 0]


    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(131)
    toplot = image_out_NN[0,::,::,0]
    plt.imshow(toplot)
    ax = fig.add_subplot(132)
    toplot = image_out_NN[1,::,::,0]
    plt.imshow(toplot)
    ax = fig.add_subplot(133)
    toplot = image_out_NN[1,::,::,0]
    plt.imshow(toplot)
    plt.savefig('predictioon.png')