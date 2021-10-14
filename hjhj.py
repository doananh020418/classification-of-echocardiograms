import matplotlib.pyplot as plt
import pandas as pd
import os

df = pd.read_csv('results.csv')
# img_path = os.path.join(os.path.join('static/processed_data',df['Sample'][0]),df['image_path'][0])
# img = plt.imread(img_path)
# plt.imshow(img)
# plt.axis('off')
# plt.title('Prediction: '+df['Prediction'][0]+ '- Confidence: '+str(df['Confidence'][0]))
# plt.show()
mode = df['Prediction'].mode().values[0]
new_df = df[df['Prediction'] != mode]
print(new_df)
