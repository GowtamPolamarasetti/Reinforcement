import numpy as np
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.models import load_model
    
    # Define AttentionLayer as provided by User
    @keras.utils.register_keras_serializable()
    class AttentionLayer(Layer):
        def __init__(self, units=32, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)
            self.units = units

        def build(self, input_shape):
            self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                     initializer="glorot_uniform",
                                     name="W")
            self.b = self.add_weight(shape=(self.units,), initializer="zeros", name="b")
            self.v = self.add_weight(shape=(self.units, 1),
                                     initializer="glorot_uniform", name="v")
            super(AttentionLayer, self).build(input_shape)

        def call(self, inputs, mask=None):
            score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
            score = tf.tensordot(score, self.v, axes=1)
            score = tf.squeeze(score, axis=-1)

            if mask is not None:
                minus_inf = -1e9
                score = tf.where(mask, score, minus_inf)

            alpha = tf.nn.softmax(score, axis=-1)
            alpha_expanded = tf.expand_dims(alpha, axis=-1)
            context = tf.reduce_sum(inputs * alpha_expanded, axis=1)
            return context

        def get_config(self):
            config = super(AttentionLayer, self).get_config()
            config.update({"units": self.units})
            return config

    class BiLSTMModel:
        def __init__(self, model_path='models/bilstm_50_trend.keras'):
            self.model_path = model_path
            self.model = None
            
        def load(self):
            if not os.path.exists(self.model_path):
                print(f"Error: Model file not found at {self.model_path}")
                return False
                
            try:
                self.model = load_model(
                    self.model_path,
                    custom_objects={'AttentionLayer': AttentionLayer}
                )
                print("BiLSTM Model loaded successfully.")
                return True
            except Exception as e:
                print(f"Failed to load BiLSTM model: {e}")
                return False

        def predict(self, sequence_str, current_trend):
            """
            Predict confidence using the sequence string from Renko data.
            """
            if self.model is None:
                return 0.5 # Default neutral confidence
                
            # Parse sequence
            # Expecting format like "1,-1,1,1,..."
            try:
                if isinstance(sequence_str, str):
                    seq = [int(x) for x in sequence_str.split(',')]
                else:
                     # Ensure it's iterable
                     seq = list(sequence_str)
                     
                if len(seq) == 50:
                    new_seq = seq[1:] + [int(current_trend)]
                    
                    input_tensor = np.array(new_seq).reshape(1, 50, 1)
                    prediction = self.model.predict(input_tensor, verbose=0)
                    return prediction[0][0]
                else:
                    return 0.5
            except Exception as e:
                # print(f"Prediction error: {e}")
                return 0.5

except ImportError:
    print("WARNING: TensorFlow not found. BiLSTM features will be disabled (set to 0.5).")
    class BiLSTMModel:
        def __init__(self, model_path=None):
            pass
        def load(self):
            print("BiLSTM Disabled (No TF).")
            return True
        def predict(self, sequence_str, current_trend):
            return 0.5

