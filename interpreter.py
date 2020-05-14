# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 00:35:34 2020

@author: DELL
"""

import tensorflow as tf




interpreter = tf.lite.Interpreter(model_path="mobilenet_float_v1_224.tflite")
interpreter.allocate_tensors()

print("== Input details ==")
print("name:", interpreter.get_input_details()[0]['name'])
print("shape:", interpreter.get_input_details()[0]['shape'])
print("type:", interpreter.get_input_details()[0]['dtype'])

print("\n== Output details ==")
print("name:", interpreter.get_output_details()[0]['name'])
print("shape:", interpreter.get_output_details()[0]['shape'])
print("type:", interpreter.get_output_details()[0]['dtype'])