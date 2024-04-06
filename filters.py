from utils import prepare_data, compile_model, fit_model
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

data = prepare_data()

filters = [(3, 3), (5, 5), (7, 7), (9, 9), (13, 13), (15, 15), (19, 19), (23, 23), (25, 25), (31, 31)]
models = [0] * len(filters)
histories = [0] * len(filters)

plt.rcParams['figure.figsize'] = (15.0, 15.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['font.family'] = 'Times New Roman'

epochs = 3
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + epochs))

for i in range(len(filters)):
    print("----------------------------")
    print("kernel size = ", filters[i])
    models[i] = compile_model(filters[i])
    histories[i] = models[i].fit(data['x_train'], data['y_train'],
                                 batch_size=128, epochs=epochs,
                                 validation_data=(data['x_validation'], data['y_validation']),
                                 callbacks=[annealer], verbose=1)

legend_names = []
for i in range(len(histories)):
    plt.plot(histories[i].history['accuracy'], '-o', linewidth=3.0)
    legend_names.append('kernel_size {0}'.format(filters[i][0]))

plt.legend(legend_names, loc='lower right', fontsize='xx-large', borderpad=2)

plt.xlabel('Epoch', fontsize=20, fontname='Times New Roman')
plt.ylabel('Validation Accuracy', fontsize=20, fontname='Times New Roman')
plt.yscale('linear')
plt.ylim(0.85, 1.0)
plt.xlim(0.5, 5.3)
plt.title('Accuracy for different sizes of filters', fontsize=22)
plt.tick_params(labelsize=18)
plt.show()
