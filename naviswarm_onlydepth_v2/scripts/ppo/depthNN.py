from tensorflow.keras.models import Sequential,save_model, load_model
from tensorflow.keras.layers import Conv3D,ConvLSTM2D,BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pylab as plt
import os




# Artificial data generation:
# Generate movies with 3 to 7 moving squares inside.
# The squares are of shape 1x1 or 2x2 pixels,
# which move linearly over time.
# For convenience we first create movies with bigger width and height (80x80)
# and at the end we select a 40x40 window.

def generate_movies(n_samples=2200, n_frames=7):
    row = 100
    col = 120
    noisy_movies = np.zeros((n_samples, n_frames, row, col,1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col,1), dtype=np.float)

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

            pix_val = np.random.randint(0,5)
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

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[i, t, x_shift - w: x_shift + w,
                               y_shift - w: y_shift + w,0] += pix_val

    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, ::, 20:80, 20:100,::]
    shifted_movies = shifted_movies[::, ::, 20:80, 20:100,::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies

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

    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),activation='sigmoid', padding='same', 
                   data_format='channels_last'))
    seq.compile(loss='binary_crossentropy', optimizer='adadelta')
    seq.summary()
    return seq
    
class ImageNN(object):
    def __init__(self):
        self.imagemodule = load_model("training_depth.h5")

    def predict_image(self,inputimage):
        predicted_image = self.imagemodule.predict(inputimage[np.newaxis,::,::,::,::])
        return predicted_image[0,2,::,::,::]



if __name__ == "__main__":

    # Train the network
    noisy_movies, shifted_movies = generate_movies(n_samples=2200)
    #noisy_movies.shape
    seq = create_module()
    seq.fit(noisy_movies[:2000], shifted_movies[:2000], batch_size=10, epochs=300, validation_split=0.05)


    checkpoint_path = "cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = ModelCheckpoint(checkpoint_path,verbose=1)

    save_path = "training_depth.h5"
    save_model(seq, save_path)

    #seq=load_model(save_path)



    which = 2004
    track = noisy_movies[which][:3, ::, ::, ::]

    for j in range(16):
        new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
        new = new_pos[::, -1, ::, ::, ::]
        track = np.concatenate((track, new), axis=0)

    track2 = noisy_movies[which][::, ::, ::, ::]
    for i in range(6):
        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_subplot(121)

        if i >= 3:
            ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
        else:
            ax.text(1, 3, 'Initial trajectory', fontsize=20)

        toplot = track[i, ::, ::, 0]

        plt.imshow(toplot)
        ax = fig.add_subplot(122)
        plt.text(1, 3, 'Ground truth', fontsize=20)

        toplot = track2[i, ::, ::, 0]
        if i >= 2:
            toplot = shifted_movies[which][i - 1, ::, ::, 0]

        plt.imshow(toplot)
        plt.savefig('%i_animate.png' % (i + 1))