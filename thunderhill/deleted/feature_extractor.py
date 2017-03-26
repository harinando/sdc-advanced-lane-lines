def load_from_h5(model_file, out):
    _model = load_model(model_file)
    model = Model(input=_model.layers[0].input, output=_model.layers[out].output)

    for layer in model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='mse')
    return model