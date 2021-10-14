import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dicom
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_name = 'static/model/mymodel_echocv_500-500-8_adam_16_0.9394.h5'
model = ''
model = load_model(model_name)
labels = {'plax': 0, 'psax-av': 1, 'psax-mv': 2, 'psax-ap': 3, 'a4c': 4, 'a5c': 5, 'a3c': 6, 'a2c': 7}
labels = dict((v, k) for k, v in labels.items())
batch_size = 8


def convert_images(images):
    # convert to N*H*W*3
    if len(images.shape) == 2:
        # only one GRAY frame
        images = np.stack([images] * 3).transpose(1, 2, 0)[None, ...]
    elif len(images.shape) == 3:
        if images.shape[-1] <= 3:
            images = images[None, ...]
            # only one RGB frame
        else:
            # video of GRAY frames
            images = np.repeat(images[..., None], 3, axis=-1)
    else:
        # video of RGB frames
        pass
    return images


def get_image(path):
    ds = dicom.dcmread(path)
    sample = '/' + path.split('/')[-1]
    save_path = os.path.join(os.path.abspath('static/processed_data'), path.split('/')[-1])
    # print(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        save_path = save_path + sample
        os.mkdir(save_path)
        for i, (img, label) in enumerate(zip(ds.pixel_array, ds.keys())):
            img = convert_images(img)
            img = img.reshape(img.shape[1:])
            img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
            plt.imsave(f'{save_path}/{label}.png', img, cmap='bone')

    # else:
    #     if not os.path.exists(save_path + sample):
    #         save_path = save_path + sample
    #         os.mkdir(save_path)
    #     else:
    #         files = glob.glob(save_path + sample + '/*')
    #         save_path = save_path + sample
    #         for f in files:
    #             os.remove(f)
    # for i, (img, label) in enumerate(zip(ds.pixel_array, ds.keys())):
    #     img = convert_images(img)
    #     img = img.reshape(img.shape[1:])
    #     img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    #     plt.imsave(f'{save_path}/{label}.png', img, cmap='bone')
    return save_path

def display(df, show_false=False):
    number_imgs = 16
    columns = int(np.ceil(np.sqrt((number_imgs))))
    rows = columns
    fig = plt.figure(figsize=(15, 15))
    if show_false == False:
        for i in range(1, columns * rows + 1):
            if i <= (number_imgs):
                fig.add_subplot(rows, columns, i)
                img_path = os.path.join(os.path.join('static/processed_data', df['Sample'][i]), df['image_path'][i])
                img = plt.imread(img_path)
                plt.imshow(img)
                plt.axis('off')
                # plt.tight_layout(True)
                plt.title('Prediction: ' + df['Prediction'][i] + '- Confidence: ' + str(df['Confidence'][i]))
        plt.show()
    else:
        mode = df['Prediction'].mode().values[0]
        new_df = df[df['Prediction'] != mode]
        if len(new_df) == 0:
            print("No false case!")
        else:
            for i in range(len(new_df)):
                columns = int(np.ceil(np.sqrt((number_imgs))))
                rows = columns
                fig.add_subplot(rows, columns, i)
                img_path = os.path.join(os.path.join('static/processed_data', new_df['Sample'][i]),
                                        new_df['image_path'][i])
                img = plt.imread(img_path)
                plt.imshow(img)
                plt.tight_layout(True)
                plt.axis('off')
                plt.title('Prediction: ' + new_df['Prediction'][i] + '- Confidence: ' + str(new_df['Confidence'][i]))
            plt.show()


def get_predictions(path):
    processed_path = get_image(path)
    input_shape = (224, 224, 3)
    batch_size = 8

    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        directory=processed_path,
        target_size=input_shape[:2],
        color_mode='rgb',
        class_mode=None,
        batch_size=batch_size,
        shuffle=False
    )
    STEP_SIZE_TEST = test_generator.n / batch_size
    test_generator.reset()
    pred = model.predict_generator(test_generator, verbose=1)
    # print('pred ', pred)
    confidence = ["{0:.3f}".format(np.amax(p)) for p in pred]
    predicted_class_indices = np.argmax(pred, axis=1)
    # print('predicted_class_indices:', predicted_class_indices)
    most_voted_class = max(set(predicted_class_indices), key=predicted_class_indices.tolist().count)
    print('most_voted_class:', most_voted_class)
    labels = {'plax': 0, 'psax-av': 1, 'psax-mv': 2, 'psax-ap': 3, 'a4c': 4, 'a5c': 5, 'a3c': 6, 'a2c': 7}
    labels = dict((v, k) for k, v in labels.items())
    predictions = labels[most_voted_class]
    print('prediction:', predictions)
    filenames = test_generator.filenames
    print(filenames)
    results = pd.DataFrame(
        {"Sample": [s.split('\\')[0] for s in filenames], "Image_tag": [s.split('\\')[-1] for s in filenames],
         "image_path": filenames,
         "Prediction": predictions, "Confidence": confidence})

    # display(results)
    return results


def main():
    results = get_predictions('static/raw_data/1.2.840.113663.1500.852470722028710568801388968764085170____IM_0024')
    display(results,show_false=False)


if __name__ == "__main__":
    main()

# ds = dicom.dcmread('static/raw_data/1.2.840.113663.1500.185026841092894878791319844862664559____IM_0002')
# images = convert_images(ds.pixel_array)
# cv2.imshow('hjhj',cv2.cvtColor(images[0],cv2.COLOR_YCrCb2RGB))
# cv2.waitKey(0)
