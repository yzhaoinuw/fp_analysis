# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:09:21 2023

@author: yzhao
"""

from scipy.io import loadmat


def run_inference(data, model_choice, num_class=None, output_path=None):
    if model_choice == "msda":
        from msda_v1 import run_inference_msda
        if num_class is None:
            num_class = 3
        run_inference_msda.infer(data, num_class=num_class, output_path=None)
        
    else:
        import sys
        
        if './sdreamer' not in sys.path:
            sys.path.append('./sdreamer')
            
        from sdreamer import run_inference_sdreamer
        
        run_inference_sdreamer.infer(data, output_path)

if __name__ == "__main__":
    model_choice = "msda"
    data = loadmat("C:\\Users\\yzhao\\python_projects\\sleep_scoring\\data.mat")
    run_inference(data, model_choice, num_class=4)