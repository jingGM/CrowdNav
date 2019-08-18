from tensorflow.keras.models import Sequential,save_model, load_model
from tensorflow.keras.layers import Conv3D,ConvLSTM2D,BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pylab as plt
import os
import cv2




# Artificial data generation:
# Generate movies with 3 to 7 moving squares inside.
# The squares are of shape 1x1 or 2x2 pixels,
# which move linearly over time.
# For convenience we first create movies with bigger width and height (80x80)
# and at the end we select a 40x40 window.

# def generate_movies(n_samples=1500, n_frames=15):
#     row = 100
#     col = 120
#     Maxvalue = 5
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

#             pix_val = np.random.randint(0,Maxvalue)
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

#     noisy_movies[noisy_movies >= Maxvalue] = Maxvalue
#     shifted_movies[shifted_movies >= Maxvalue] = Maxvalue
#     for i in range(n_samples):
#         for f in range(n_frames):
#             cv2.imwrite(os.path.join("./frames","noise_{0:06d}_{0:06d}.jpg".format(i,f)),noisy_movies[i, f, 20:80, 20:100,0])
    
#     return noisy_movies, shifted_movies


def generate_input(row, col, img_num=6000, n_frames=15, loc="./Depth_Images_Training"):
    skipedfram = 1
    maxpixelvalue = 5
    n_samples = int(img_num/(n_frames*skipedfram))

    noisy_imgs = np.zeros((n_samples, n_frames, row, col,1), dtype=np.float)
    shifted_imgs = np.zeros((n_samples, n_frames, row, col,1), dtype=np.float)


    current_idx = 0

    current_frame = None
    #next_frame = cv2.imread(os.path.join(loc, "frame{}.jpg".format(1)), cv2.IMREAD_GRAYSCALE)
    next_frame = np.genfromtxt("./csvs/ground{0:06d}.csv".format(current_idx), delimiter=',')
    #next_frame = np.array(cv2.resize(next_frame, (80, 60)), dtype=np.float)
    next_frame = next_frame /maxpixelvalue

    for i in range(n_samples):
        for t in range(n_frames):

            current_frame = next_frame
            # next_frame = cv2.imread(os.path.join(loc, "frame{}.jpg".format(current_idx*skipedfram + skipedfram)), cv2.IMREAD_GRAYSCALE)
            next_frame = np.genfromtxt("./csvs/ground{0:06d}.csv".format(current_idx*skipedfram + skipedfram), delimiter=',')

            #next_frame = np.array(cv2.resize(next_frame, (80, 60)), dtype=np.float)
            next_frame = next_frame /maxpixelvalue

            noisy_imgs[i, t, ::, ::, 0] = current_frame
            shifted_imgs[i, t, ::, ::, 0] = next_frame
            current_idx += skipedfram
            
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    toplot = current_frame
    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    toplot = next_frame
    plt.imshow(toplot)
    plt.savefig('./generatedimages/{0}_{1}.png'.format(i,t))
    # for i in range(n_samples):
    #     for t in range(n_frames-1):
    #         if (noisy_imgs[i, t+1, ::, ::, 0] != shifted_imgs[i, t, ::, ::, 0]).any():
    #             print(noisy_imgs[i, t+1, ::, ::, 0] == shifted_imgs[i, t, ::, ::, 0])

    # for i in range(n_samples):
    #     for f in range(n_frames):
    #         cv2.imwrite(os.path.join("./frames","noise_{0:06d}_{0:06d}.jpg".format(i,f)),noisy_imgs[i, f, ::, ::, 0])
    #         # cv2.imwrite(os.path.join("./frames","noise_{0:06d}_{0:06d}-1.jp g".format(i,f)),shifted_imgs[i, f, ::, ::, 0])

    return noisy_imgs, shifted_imgs



def create_module():
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=60, kernel_size=(5, 5),input_shape=(None, 60, 80,1),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=60, kernel_size=(5, 5),padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3),padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3),padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3),padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),activation='sigmoid', padding='same', 
                   data_format='channels_last'))
    seq.compile(loss='mean_squared_error', optimizer='adadelta')
    seq.summary()
    return seq

class ImageNN(object):
    def __init__(self):
        self.imagemodule = load_model("training_depth_v2.h5")
        # self.imagemodule._make_predict_function()
        # self.graph = tf.get_default_graph()

    def predict_image(self,inputimage):
        # graph = tf.get_default_graph()
        
        # inputimageextract.astype('float64')     
        # print(inputimageextract.shape)
        # self.imagemodule._make_predict_function()
        # #print(inputimageextract.dtype)
        # with self.graph.as_default():
        inputimageextract = inputimage[np.newaxis,::,::,::,::]
        predicted_image = self.imagemodule.predict(inputimageextract)
        return predicted_image





if __name__ == "__main__":
    # Train the network
    # noisy_movies, shifted_movies = generate_movies(n_samples=1500)
    #noisy_movies.shape

    img_num = 50
    # image_row, image_col = cv2.imread("./Depth_Images_Training/frame0.jpg", cv2.IMREAD_GRAYSCALE).shape
    image_row, image_col = 60, 80

    # noisy_imgs = []
    # shifted_imgs = []

    # noisy_imgs.append(cv2.imread("./Depth_Images_Training/frame0.jpg"))

    # for i in range(1,img_num):
    #     if i == 1 or i == img_num:
    #         print("hi")
    #     noisy_imgs.append(cv2.imread("./Depth_Images_Training/frame{}.jpg".format(i)))
    #     shifted_imgs.append(cv2.imread("./Depth_Images_Training/frame{}.jpg".format(i)))
    #     print(i)

    # shifted_imgs.append(cv2.imread("./Depth_Images_Training/frame{}.jpg".format(img_num)))

    noisy_imgs, shifted_imgs = generate_input(image_row, image_col)

    seq = create_module()
    #load_path = "training_depth_v1.h5"
    #seq=load_model(load_path)

    seq.fit(noisy_imgs, shifted_imgs, batch_size=10, epochs=400, validation_split=0.05)


    checkpoint_path = "cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = ModelCheckpoint(checkpoint_path,verbose=1)

    save_path = "training_depth_v2.h5"
    save_model(seq, save_path)

    #seq=load_model(save_path)



    # which = 1450
    # track = noisy_movies[which][:3, ::, ::, ::]

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

    #     plt.imshow(toplot)
    #     plt.savefig('%i_animate.png' % (i + 1))

