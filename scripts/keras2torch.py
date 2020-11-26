from tensorflow import keras


model = keras.models.load_model('temp/test.pth.tar')
weights=model.get_weights()