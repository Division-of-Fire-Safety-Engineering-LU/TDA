import numpy as np
from lmfit import Parameters

def userdefined_models():
    """ This function returns a list of models, defined by the user.
    Every model in the list is a dictionary with
        name - (string)
        parameters - parameter class from lmfit
        function - a (lambda) function that returns a numpy array of v for an
                   array of k and depends on the parameters """
    
    # Tupple in parameters have to contain a name and a value but can contain
    # more: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    
    params = Parameters()
    params.add_many(('vf', 110), ('kj', 60))
    greenshields = {
        'name': 'Greenshields et al. (1935)',
        'parameters': params,
        'function': lambda k,vf,kj:vf*(1-np.where(k<=0,0, np.where(k>kj,kj,k))/kj),
        'keypoints' : [1]}
    
    params = Parameters()
    params.add_many(('vf', 110), ('kc', 20))
    underwood = {
        'name': 'Underwood (1961)',
        'parameters': params,
        'function': lambda k, vf, kc : vf*np.exp(-np.where(k<=0,0,k)/kc)}
    
    params = Parameters()
    params.add_many(('vf', 110), ('kc', 20))
    drake = {
        'name': 'Drake et al. (1965)',
        'parameters': params,
        'function': lambda k, vf, kc : vf*np.exp(-0.5*(np.where(k<=0,0,k)/kc)**2)}
    
    params = Parameters()
    params.add_many(('vf', 110), ('kc', 20), ('kj', 60))
    daganzo = {
        'name': 'Daganzo (1994)',
        'function': lambda k,vf,kc,kj:np.where(k>=kj,0,np.where(k<=kc,vf,vf*(kc/k)*(kj-k)/(kj-kc))),
        'parameters': params,
        'keypoints' : [1,2]}
      
    models = [greenshields, underwood, drake, daganzo]
    return models
