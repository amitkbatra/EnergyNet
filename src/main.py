import argparse
import os

from training.train import preprocess_data, baseline_model, train, save_model_keras, save_scaler
from optimization.optimizers import (
    full_integer_quantization,
    float16_quantization,
    float16_int8_quantization,
    pruning,
    quantization_aware_training,
    apply_knowledge_distillation
)
from profiling.profiler import (
    setup_logger,
    set_global_variables,
    benchmark_training,
    benchmark_inference,
    benchmark_load,
    perform_evaluation_tflite,
    get_average_stats,
    get_std_dev_stats,
    get_max_stats,
    save_stats_to_logfile,
    get_gzipped_model_size,
    delete_model
)
from deployment.exporter import export_keras_model_to_tflite, export_keras_model_to_onnx, export_tflite_model_to_onnx
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="EnergyNet: A Methodological Framework for Optimizing Energy Consumption of Deep Neural Networks")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Create a parser for the "train" command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    
    # Create a parser for the "optimize" command
    optimize_parser = subparsers.add_parser('optimize', help='Apply optimization techniques')
    optimize_parser.add_argument('--strategy', type=str, required=True, choices=['pruning', 'quantization', 'distillation'], help='Optimization strategy to apply')
    optimize_parser.add_argument('--input_model', type=str, required=True, help='Path to the input model')
    optimize_parser.add_argument('--output_model', type=str, required=True, help='Path to save the optimized model')

    # Create a parser for the "profile" command
    profile_parser = subparsers.add_parser('profile', help='Evaluate energy-performance trade-offs')
    profile_parser.add_argument('--input_model', type=str, required=True, help='Path to the optimized model')

    # Create a parser for the "deploy" command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy the optimized model')
    deploy_parser.add_argument('--input_model', type=str, required=True, help='Path to the optimized model')

    args = parser.parse_args()

    if args.command == 'train':
        config_path = args.config
        config = {}
        with open(config_path, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        df_train, df_test, gf, standard = preprocess_data(config['data_path'])

        model = baseline_model(input_dim=len(gf), n_output=2)
        model, history = train(model, data_transformed, df_train, config['validation_split'], config['epochs'], config['batch_size'], config['es_patience'], config['es_restore_best_weights'])
        save_model_keras(model, config['model_path'])
        save_scaler(standard, config['scaler_path'])

    elif args.command == 'optimize':
        input_model_path = args.input_model
        output_model_path = args.output_model
        
        df_train, df_test, gf, standard = preprocess_data(config['data_path'])
        
        model = tf.keras.models.load_model(input_model_path)

        if args.strategy == 'pruning':
            initial_sparsity = 0.5
            final_sparsity = 0.8
            begin_step = 0
            num_samples = len(df_train)
            batch_size = 256
            epochs = 10
            end_step = np.ceil(num_samples / batch_size).astype(np.int32) * epochs
            
            pruned_model_path, _ = pruning(model, df_train, data_transformed, initial_sparsity, final_sparsity, begin_step, end_step)
            
            import shutil
            shutil.copyfile(pruned_model_path, output_model_path)
            
        elif args.strategy == 'quantization':
            quantized_model_path, _ = quantization_aware_training(model, df_train, data_transformed)
            
            import shutil
            shutil.copyfile(quantized_model_path, output_model_path)
            
        elif args.strategy == 'distillation':
            input_dim = len(gf)
            validation_split = 0.2
            epochs = 10000
            batch_size = 1024
            es_patience = 20
            es_restore_best_weights = True
            
            distilled_model_path, _ = apply_knowledge_distillation(model, df_train, data_transformed, input_dim, validation_split, epochs, batch_size, es_patience, es_restore_best_weights)
            
            import shutil
            shutil.copyfile(distilled_model_path, output_model_path)

    elif args.command == 'profile':
        input_model_path = args.input_model
        
        df_train, df_test, gf, standard = preprocess_data(config['data_path'])
        
        model = tf.keras.models.load_model(input_model_path)
        
        setup_logger()
        set_global_variables()
        
        benchmark_training(device="CPU")
        benchmark_inference(device="CPU")
        benchmark_load(device="CPU")
        
        average_stats = get_average_stats()
        std_dev_stats = get_std_dev_stats()
        max_stats = get_max_stats()
        
        save_stats_to_logfile(average_stats, std_dev_stats, max_stats)
        
        model_size = get_gzipped_model_size()
        print(f"Model size (gzip): {model_size}")
        
        delete_model()
        
        perform_evaluation_tflite(input_model_path)

    elif args.command == 'deploy':
        input_model_path = args.input_model
        
        model = tf.keras.models.load_model(input_model_path)
        
        tflite_model = export_keras_model_to_tflite(model)
        onnx_model = export_keras_model_to_onnx(model)
        
        # save the models
        with open("model.tflite", 'wb') as f:
            f.write(tflite_model)
        
        onnx.save(onnx_model, "model.onnx")
        
        print("Model exported to model.tflite and model.onnx")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
