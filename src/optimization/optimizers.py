import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
from ..training.train import baseline_model

class DefaultDenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def __init__(self, quantization_config="LastValueQuantizer", num_bits=8, symmetric=True, narrow_range=False, per_axis=False):
        self.quantization_config = quantization_config
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.narrow_range = narrow_range
        self.per_axis = per_axis

        if self.quantization_config == "LastValueQuantizer":
            self.quantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer(num_bits=self.num_bits, symmetric=self.symmetric, narrow_range=self.narrow_range, per_axis=self.per_axis)
        elif self.quantization_config == "MovingAverageQuantizer":
            self.quantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=self.num_bits, symmetric=self.symmetric, narrow_range=self.narrow_range, per_axis=self.per_axis)
        elif self.quantization_config == "AllValuesQuantizer":
            self.quantizer = tfmot.quantization.keras.quantizers.AllValuesQuantizer(num_bits=self.num_bits, symmetric=self.symmetric, narrow_range=self.narrow_range, per_axis=self.per_axis)

    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
        return [
            (
                layer.kernel,
                self.quantizer,
            )
        ]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        return [
            (
                layer.activation,
                self.quantizer,
            )
        ]

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        layer.activation = quantize_activations[0]

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {
            "quantization_config": self.quantization_config,
            "num_bits": self.num_bits,
            "symmetric": self.symmetric,
            "narrow_range": self.narrow_range,
            "per_axis": self.per_axis,
        }

def full_integer_quantization(trained_model, data_transformed):
    def dataset_generator():
        for data in tf.data.Dataset.from_tensor_slices((data_transformed)).batch(1).take(int(len(data_transformed) * 0.25)):
            yield [tf.dtypes.cast(data, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = dataset_generator
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    quantized_model = converter.convert()

    print("Applied Full-integer Quantization")

    # Save the model to disk
    quantized_tflite_model_file = 'model.tflite'
    with open(quantized_tflite_model_file, 'wb') as f:
        f.write(quantized_model)

    return quantized_tflite_model_file

def float16_quantization(trained_model, data_transformed):
    def dataset_generator():
        for data in tf.data.Dataset.from_tensor_slices((data_transformed)).batch(1).take(int(len(data_transformed) * 0.25)):
            yield [tf.dtypes.cast(data, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)
    converter.representative_dataset = dataset_generator
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.float16]
    quantized_model = converter.convert()

    print("Applied float16 Activations and int8 weights-Quantization")

    # Save the model to disk
    quantized_tflite_model_file = 'model.tflite'
    with open(quantized_tflite_model_file, 'wb') as f:
        f.write(quantized_model)

    return quantized_tflite_model_file

def float16_int8_quantization(trained_model, data_transformed):
    def dataset_generator():
        for data in tf.data.Dataset.from_tensor_slices((data_transformed)).batch(1).take(int(len(data_transformed) * 0.25)):
            yield [tf.dtypes.cast(data, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)
    converter.representative_dataset = dataset_generator
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    quantized_model = converter.convert()

    print("Applied float16 Activations and int8 weights-Quantization")

    # Save the model to disk
    quantized_tflite_model_file = 'model.tflite'
    with open(quantized_tflite_model_file, 'wb') as f:
        f.write(quantized_model)

    return quantized_tflite_model_file

def pruning(model, df_train, data_transformed, initial_sparsity, final_sparsity, begin_step, end_step, convert_to_tflite=True):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=initial_sparsity, final_sparsity=final_sparsity, begin_step=begin_step, end_step=end_step
        )
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(
        optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=["accuracy"]
    )

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
    ]

    history = model_for_pruning.fit(data_transformed, pd.get_dummies(df_train['tag']), validation_split=0.1, epochs=10, batch_size=256, callbacks=callbacks, verbose=0)

    pruned_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    if convert_to_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
        converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY, tf.lite.Optimize.DEFAULT]
        pruned_tflite_model = converter.convert()

        pruned_tflite_model_file = 'model.tflite'
        with open(pruned_tflite_model_file, 'wb') as f:
            f.write(pruned_tflite_model)

        return pruned_tflite_model_file, history
    else:
        return pruned_model, history

class DefaultDenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def __init__(self, quantization_config="LastValueQuantizer", num_bits=8, symmetric=True, narrow_range=False, per_axis=False):
        self.quantization_config = quantization_config
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.narrow_range = narrow_range
        self.per_axis = per_axis

        if self.quantization_config == "LastValueQuantizer":
            self.quantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer(num_bits=self.num_bits, symmetric=self.symmetric, narrow_range=self.narrow_range, per_axis=self.per_axis)
        elif self.quantization_config == "MovingAverageQuantizer":
            self.quantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=self.num_bits, symmetric=self.symmetric, narrow_range=self.narrow_range, per_axis=self.per_axis)
        elif self.quantization_config == "AllValuesQuantizer":
            self.quantizer = tfmot.quantization.keras.quantizers.AllValuesQuantizer(num_bits=self.num_bits, symmetric=self.symmetric, narrow_range=self.narrow_range, per_axis=self.per_axis)

    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, self.quantizer)]

    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, self.quantizer)]

    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {
            "quantization_config": self.quantization_config,
            "num_bits": self.num_bits,
            "symmetric": self.symmetric,
            "narrow_range": self.narrow_range,
            "per_axis": self.per_axis,
        }

def quantization_aware_training(model, df_train, data_transformed, quantization_config="LastValueQuantizer", num_bits=8, symmetric=True, narrow_range=False, per_axis=False):
    quantize_model = tfmot.quantization.keras.quantize_model
    
    # Annotate the baseline model
    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=lambda layer: tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config=DefaultDenseQuantizeConfig(quantization_config, num_bits, symmetric, narrow_range, per_axis)) if isinstance(layer, tf.keras.layers.Dense) else layer
    )

    # Apply quantization to the model
    q_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    q_aware_model.compile(
        optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=["accuracy"]
    )

    batch_size = 256
    n_epochs = 10
    validation_split = 0.1

    history = q_aware_model.fit(data_transformed, pd.get_dummies(df_train['tag']), validation_split=validation_split, epochs=n_epochs, batch_size=batch_size, verbose=0)

    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_model = converter.convert()

    quantized_tflite_model_file = 'model.tflite'
    with open(quantized_tflite_model_file, 'wb') as f:
        f.write(quantized_model)

    return quantized_tflite_model_file, history


class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

def apply_knowledge_distillation(teacher, df_train, data_transformed, input_dim, validation_split, epochs, batch_size, es_patience, es_restore_best_weights, convert_to_tflite=True, student_model_fn=None, reduction_factor=0.5, alpha=0.1, temperature=3):
    print("Applying Knowledge Distillation")

    best_student = None
    best_history = None
    best_score = 0

    # Grid search for alpha and temperature
    for current_alpha in [0.1, 0.3, 0.5]:
        for current_temperature in [2, 3, 4]:
            print(f"  Training with alpha={current_alpha}, temperature={current_temperature}")

            if student_model_fn:
                student = student_model_fn(input_dim=input_dim, n_output=2)
            elif reduction_factor is not None:
                teacher_num_params = teacher.count_params()
                student_num_params = int(teacher_num_params * reduction_factor)
                print(f"  Reducing model parameters from {teacher_num_params} to {student_num_params}")

                student = Sequential()
                student.add(Dense(int(input_dim * 4 * reduction_factor), input_dim=input_dim, activation='relu'))
                student.add(Dense(int(input_dim * 6 * reduction_factor), activation='relu'))
                student.add(Dense(int(input_dim * 3 * reduction_factor), activation='relu'))
                student.add(Dense(2, activation='softmax'))
                student.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            else:
                student = baseline_model(input_dim=input_dim, n_output=2)

            distiller = Distiller(student=student, teacher=teacher)
            distiller.compile(
                optimizer=keras.optimizers.Adam(),
                metrics=[keras.metrics.CategoricalAccuracy()],
                student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=False),
                distillation_loss_fn=keras.losses.KLDivergence(),
                alpha=current_alpha,
                temperature=current_temperature,
            )

            es = EarlyStopping(
                monitor="val_student_loss",
                mode="min",
                patience=es_patience,
                restore_best_weights=es_restore_best_weights,
            )
            history = distiller.fit(
                data_transformed,
                pd.get_dummies(df_train["tag"]),
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[es],
                verbose=0
            )

            # Evaluate the student model
            score = distiller.evaluate(data_transformed, pd.get_dummies(df_train["tag"]), verbose=0)
            print(f"  Evaluation score: {score}")

            # Select the best model based on a metric (e.g., accuracy)
            if score[1] > best_score:  # Assuming the second metric is accuracy
                best_score = score[1]
                best_student = distiller.student
                best_history = history

    if convert_to_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(best_student)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        distilled_tflite_model = converter.convert()

        distilled_tflite_model_file = 'model.tflite'
        with open(distilled_tflite_model_file, 'wb') as f:
            f.write(distilled_tflite_model)

        return distilled_tflite_model_file, best_history
    else:
        return best_student, best_history

    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.CategoricalAccuracy()],
        student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=False),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=alpha,
        temperature=temperature,
    )

    es = EarlyStopping(
        monitor="val_student_loss",
        mode="min",
        patience=es_patience,
        restore_best_weights=es_restore_best_weights,
    )
    history = distiller.fit(
        data_transformed,
        pd.get_dummies(df_train["tag"]),
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
    )

    if convert_to_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(distiller.student)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        distilled_tflite_model = converter.convert()

        distilled_tflite_model_file = 'model.tflite'
        with open(distilled_tflite_model_file, 'wb') as f:
            f.write(distilled_tflite_model)

        return distilled_tflite_model_file, history
    else:
        return distiller.student, history
