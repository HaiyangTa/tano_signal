
"""
    predict: A method to predict the RHI and FCNM.
    
    input Attributes
    ----------
    dictt: dictionary 
        the  dictionary that contains only one data cube spectra.
    model: model 
        model for prediction. 
    device: GPU or CPU 
        the device for calculation.
    num_row: int 
        number of row.
    num_column: int
        number of column.
"""






def predict(dictt, model, device, num_row, num_column):
    Fcnm = np.zeros((num_row,num_column,1))
    Rhi = np.zeros((num_row,num_column,1))
    for i in range(0, len(dictt)):
        o = dictt.get(i)
        spectra = np.array(o[0])
        spectra = spectra.reshape(1,1, 1,-1).astype(np.float32)
        spectra = torch.from_numpy(spectra)
        spectra = spectra.to(device)
        ground_truth = o[1]
        position = o[2]
        prediction = model(spectra)
        Fcnm[position[1], position[2]] =abs(prediction[0][0].cpu()).detach().numpy()
        Rhi[position[1], position[2]] =abs(prediction[0][1].cpu()).detach().numpy()
    return Fcnm, Rhi