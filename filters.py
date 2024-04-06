from utils import prepare_data, compile_model, fit_model

data = prepare_data()

filters = [(3, 3), (5, 5), (9, 9), (13, 13), (15, 15), (19, 19), (23, 23), (25, 25), (31, 31)]
model = [0] * len(filters)

for i in range(len(filters)):
    print("----------------------------")
    print("kernel size = ", filters[i])
    model = compile_model(filters[i])
    fit_model(model=model, data=data)
