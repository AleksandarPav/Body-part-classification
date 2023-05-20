import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


NUM_OF_CLASSES = 22


def main():
    train_dir = 'x_ray/images/train/'
    test_dir = 'x_ray/images/test/'
    print(len(os.listdir(path=test_dir)))
    df = pd.read_csv('x_ray/train_df.csv', usecols=['image_path', 'Target'])
    df['image_path'] = df['image_path'].apply(lambda x: x.split('/')[-1])

    df = create_dataset(df)
    X = store_images(images_dir=train_dir, images_names=df['image_path'])

    y = np.array(df.drop(['image_path', 'Target'], axis=1))

    model, X_train, X_val, y_train, y_val = create_model(X=X, y=y, test_size=0.1, n_epochs=10, batch_size=64)

    model = fit_save(model=model, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val, n_epochs=10, batch_size=8)

    X_test = store_images(images_dir=test_dir, images_names=os.listdir(path=test_dir))
    y_pred = model.predict(X_test)

    plt.show()


def create_dataset(df):
    dfc = pd.DataFrame(0, index=np.arange(len(df)), columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                                             '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                                                             '21'])

    row = 0
    for i, targets in enumerate(df['Target']):
        targets = [int(v) for v in targets.split()]
        for target in targets:
            dfc.loc[row][str(target)] = 1
        row += 1

    dfc['image_path'] = df['image_path']
    dfc['Target'] = df['Target']

    return dfc


def store_images(images_dir, images_names):
    images = []
    for i in tqdm(range(len(images_names))):
        img = image.image_utils.load_img(images_dir + images_names[i], target_size=(400, 400, 1),
                                         color_mode='grayscale')
        img = image.image_utils.img_to_array(img)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        images.append(img)

    return np.array(images)


def create_model(X, y, test_size, n_epochs, batch_size):
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, test_size=test_size)

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400, 400, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_OF_CLASSES, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model, X_train, X_val, y_train, y_val


def fit_save(model, X_train, X_val, y_train, y_val, n_epochs, batch_size):
    model.fit(X_train, y_train, epochs=n_epochs, validation_data=(X_val, y_val), batch_size=batch_size)

    model.save('model.h5')

    return model


if __name__ == '__main__':
    main()