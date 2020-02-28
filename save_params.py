import scipy.io as sio

def save_params_as_MAT(params,path=None):
    if type(params)==list:
        for i in range(len(params)):
            assert type(params[i])==dict, "params must be either a single dictionary or a list of dictionaries."
            if i==0:
                totalParams = params[i]
            else:
                totalParams.update(params[i])
    else:
        totalParams=params

    matlabTotalParams={}
    for key in totalParams.keys():
        if type(totalParams[key])==dict:
            for subkey in totalParams[key].keys():
                matlabTotalParams[key.replace(" ","_")+"_"+str(subkey)] = totalParams[key][subkey]
        elif totalParams[key] is None:
            matlabTotalParams[key.replace(" ","_")] = "None"
        else:
            matlabTotalParams[key.replace(" ","_")] = totalParams[key]

    if path is None:
        sio.savemat("totalParams.mat",matlabTotalParams)
    else:
        sio.savemat(path+"totalParams.mat",matlabTotalParams)
