import functools
from copy import deepcopy
from collections import OrderedDict

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class IntermediateFeatureExtractor:

    def __init__(self, net, target_layers, evaluate=False, intrain=True):
        self.net = net
        self.evaluate = evaluate
        self.target_layers = target_layers
        self.intrain = intrain
        print("IFE initialized")
        
    def __call__(self, *args, **kwargs):
        if self.evaluate:
            self.net.eval()
        ret, handles = {}, []
        for var_name in self.target_layers:
            try:
                layer = rgetattr(self.net, var_name)
            except:
                print("var name not found")
            def hook(module, feature_in, feature_out, name=var_name):
                assert name not in ret
                if isinstance(feature_in, tuple):
                    feature_in = feature_in[0]
                if isinstance(feature_out, tuple):
                    feature_out = feature_out[0]
                ret[name] = [feature_in.detach(), feature_out.detach()]
            h = layer.register_forward_hook(hook)
            handles.append(h)
            
        output = self.net(*args, **kwargs)
        if not self.intrain:
            output = output.detach()
            
        for h in handles:
            h.remove()
            
        if self.evaluate:
            self.net.train()
        
        return ret, output