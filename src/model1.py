from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import numpy as np


img_width = 150
img_height = 150
batch_size = 16
nbr_test_samples = 1000

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']



# test data generator for prediction
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        "/Users/liu/Code/data/kaggle_fish/raw_data/data",
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = False, # Important !!!
        classes = None,
        class_mode = None)

print(test_generator)

test_image_list = test_generator.filenames

# load json and create model
json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model1.h5")
print("Loaded model from disk")



predictions = model.predict_generator(test_generator, nbr_test_samples)

np.savetxt(('predictions.txt'), predictions)


print('Begin to write submission file ..')
f_submit = open(('submit.csv'), 'w')
f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 100 == 0:
        print('{} / {}'.format(i, nbr_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

print('Submission file successfully generated!')