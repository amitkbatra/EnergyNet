import tensorflow as tf
import tf2onnx

def export_keras_model_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return tflite_model

def export_keras_model_to_onnx(model, input_dim):
    spec = (tf.TensorSpec((None, input_dim), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path="model.onnx")
    return model_proto

def export_tflite_model_to_onnx(model_path, input_dim):
    spec = (tf.TensorSpec((None, input_dim), tf.float32, name='input'),)
    model_proto, _ = tf2onnx.convert.from_tflite(model_path, output_path="model.onnx")
    return model_proto
