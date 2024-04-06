import keras.models
from utils import prepare_data, compile_model, fit_model

data = prepare_data()
model = compile_model((5, 5))
model = fit_model(model=model, data=data)

keras.models.save_model(model, "model-3x3.h5")
