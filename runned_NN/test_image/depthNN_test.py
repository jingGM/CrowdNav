from tensorflow.keras.models import Sequential,save_model, load_model
from tensorflow.keras.layers import Conv3D,ConvLSTM2D,BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pylab as plt
import os
from depthNN import ImageNN




# Artificial data generation:
# Generate movies with 3 to 7 moving squares inside.
# The squares are of shape 1x1 or 2x2 pixels,
# which move linearly over time.
# For convenience we first create movies with bigger width and height (80x80)
# and at the end we select a 40x40 window.

# def generate_movies(n_samples=2200, n_frames=15):
#     row = 100
#     col = 120
#     noisy_movies = np.zeros((n_samples, n_frames, row, col,1), dtype=np.float)
#     shifted_movies = np.zeros((n_samples, n_frames, row, col,1), dtype=np.float)

#     for i in range(n_samples):
#         # Add 3 to 7 moving squares
#         n = np.random.randint(3, 8)

#         for j in range(n):
#             # Initial position
#             xstart = np.random.randint(20, 100)
#             ystart = np.random.randint(20, 80)
#             # Direction of motion
#             directionx = np.random.randint(0, 10) - 1
#             directiony = np.random.randint(0, 10) - 1

#             # Size of the square
#             w = np.random.randint(2, 10)

#             pix_val = np.random.randint(0,5)
#             for t in range(n_frames):
#                 x_shift = xstart + directionx * t
#                 y_shift = ystart + directiony * t
#                 noisy_movies[i, t, x_shift - w: x_shift + w,
#                              y_shift - w: y_shift + w,0] += pix_val

#                 # Make it more robust by adding noise.
#                 # The idea is that if during inference,
#                 # the value of the pixel is not exactly one,
#                 # we need to train the network to be robust and still
#                 # consider it as a pixel belonging to a square.
#                 if np.random.randint(0, 2):
#                     noise_f = (-1)**np.random.randint(0, 2)
#                     noisy_movies[i, t,
#                                  x_shift - w - 1: x_shift + w + 1,
#                                  y_shift - w - 1: y_shift + w + 1,0] += noise_f * 0.1

#                 # Shift the ground truth by 1
#                 x_shift = xstart + directionx * (t + 1)
#                 y_shift = ystart + directiony * (t + 1)
#                 shifted_movies[i, t, x_shift - w: x_shift + w,
#                                y_shift - w: y_shift + w,0] += pix_val

#     # Cut to a 40x40 window
#     noisy_movies = noisy_movies[::, ::, 20:80, 20:100,::]
#     shifted_movies = shifted_movies[::, ::, 20:80, 20:100,::]

#     return noisy_movies, shifted_movies

# def create_module():
#     seq = Sequential()
#     seq.add(ConvLSTM2D(filters=60, kernel_size=(5, 5),input_shape=(None, 60, 80,1),
#                        padding='same', return_sequences=True))
#     seq.add(BatchNormalization())

#     seq.add(ConvLSTM2D(filters=60, kernel_size=(5, 5),padding='same', return_sequences=True))
#     seq.add(BatchNormalization())

#     seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3),padding='same', return_sequences=True))
#     seq.add(BatchNormalization())

#     seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3),padding='same', return_sequences=True))
#     seq.add(BatchNormalization())

#     seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),activation='sigmoid', padding='same', 
#                    data_format='channels_last'))
#     seq.compile(loss='binary_crossentropy', optimizer='adadelta')
#     seq.summary()
#     return seq
    
# Train the network
# noisy_movies, shifted_movies = generate_movies(n_samples=2200)
#noisy_movies.shape



# checkpoint_path = "cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = ModelCheckpoint(checkpoint_path,verbose=1)

# save_path = "training_depth.h5"

#seq = create_module()
#seq.fit(noisy_movies[:2000], shifted_movies[:2000], batch_size=10, epochs=300, validation_split=0.05)
#save_model(seq, save_path)
#seq1=load_model(save_path)










depthN = ImageNN()
#predict_image = depthN.predict_image(track)


frames = 3
depths = []
for i in range(frames):
    depths.append(np.zeros((60,80,1)))

for imgindex in range(320,359):
    imagenow = np.genfromtxt("./csvs/ground{0:06d}.csv".format(imgindex), delimiter=',')
    imagenow = np.expand_dims(np.array(imagenow),axis=2)
    imagenow = imagenow*(1/imagenow.max())
    print(imagenow.shape)
    imagenext = np.genfromtxt("./csvs/ground{0:06d}.csv".format(imgindex+1), delimiter=',')
    imagenext = np.expand_dims(np.array(imagenext),axis=2)
    imagenext = imagenext*(1/imagenext.max())

    for j in range(frames):
        if j == (frames-1):
            depths[j] = imagenow
        else:
            depths[j] = depths[j+1]

    imagesinput = np.array(depths)
    imageoutput = depthN.predict_image(imagesinput)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    toplot = imageoutput[0,0,::,::,0]
    toplot = toplot*(1/toplot.max())
    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    toplot = imagenext[::,::,0]
    plt.imshow(toplot)
    plt.savefig('predictions_{0:06d}.png'.format(imgindex))




# which = 2020
# track = noisy_movies[which][:3, ::, ::, ::]

# print(track.shape)
# print(track.dtype)
# print("=======================")
# #predict_image = seq1.predict(track)


# ground_truth = noisy_movies[which][2, ::, ::, 0]

# np.savetxt("predict.csv", predict_image[0,0,::,::,0], delimiter=",")

# np.savetxt("ground.csv", ground_truth, delimiter=",")

# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(131)
# toplot = predict_image[0,0,::,::,0]
# plt.imshow(toplot)
# ax = fig.add_subplot(132)
# toplot = predict_image[0,1,::,::,0]
# plt.imshow(toplot)
# plt.savefig('predictions.png')
# ax = fig.add_subplot(133)
# toplot = predict_image[0,2,::,::,0]
# plt.imshow(toplot)
# plt.savefig('predictions.png')

# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(221)
# ground_truth = noisy_movies[which][0, ::, ::, 0]
# plt.imshow(ground_truth)
# ax = fig.add_subplot(222)
# ground_truth = noisy_movies[which][1, ::, ::, 0]
# plt.imshow(ground_truth)
# ax = fig.add_subplot(223)
# ground_truth = noisy_movies[which][2, ::, ::, 0]
# plt.imshow(ground_truth)
# ax = fig.add_subplot(224)
# ground_truth = noisy_movies[which][3, ::, ::, 0]
# plt.imshow(ground_truth)
# plt.savefig('groundtruth.png')


# for j in range(16):
#     new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
#     new = new_pos[::, -1, ::, ::, ::]
#     track = np.concatenate((track, new), axis=0)

# track2 = noisy_movies[which][::, ::, ::, ::]
# for i in range(6):
#     fig = plt.figure(figsize=(10, 5))

#     ax = fig.add_subplot(121)

#     if i >= 3:
#         ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
#     else:
#         ax.text(1, 3, 'Initial trajectory', fontsize=20)

#     toplot = track[i, ::, ::, 0]

#     plt.imshow(toplot)
#     ax = fig.add_subplot(122)
#     plt.text(1, 3, 'Ground truth', fontsize=20)

#     toplot = track2[i, ::, ::, 0]
#     if i >= 2:
#         toplot = shifted_movies[which][i - 1, ::, ::, 0]

    # plt.imshow(toplot)
    # plt.savefig('%i_animate.png' % (i + 1))

