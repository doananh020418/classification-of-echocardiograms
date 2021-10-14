import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_name = 'static/model/mymodel_echocv_500-500-8_adam_16_0.9394.h5'
input_shape = (224, 224, 3)
batch_size = 8

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    directory='static/processed_data',
    target_size=input_shape[:2],
    color_mode='rgb',
    class_mode=None,
    batch_size=batch_size,
    shuffle=False
)
STEP_SIZE_TEST = test_generator.n / batch_size
test_generator.reset()

model = load_model(model_name)
pred = model.predict_generator(test_generator, verbose=1)
print('pred ', pred)
confidence = ["{0:.3f}".format(np.amax(p)) for p in pred]
predicted_class_indices = np.argmax(pred, axis=1)
print('predicted_class_indices:', predicted_class_indices)
most_voted_class = max(set(predicted_class_indices), key=predicted_class_indices.tolist().count)
print('most_voted_class:', most_voted_class)
labels = {'plax': 0, 'psax-av': 1, 'psax-mv': 2, 'psax-ap': 3, 'a4c': 4, 'a5c': 5, 'a3c': 6, 'a2c': 7}
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
print('prediction:', predictions)
filenames = test_generator.filenames
results = pd.DataFrame(
    {"Sample": [s.split('\\')[0] for s in filenames], "Image_tag": [s.split('\\')[-1] for s in filenames],
     'image_path': filenames, "Prediction": predictions, "Confidence": confidence})

results_file = "./results.csv"
results.to_csv(results_file, index=False)
