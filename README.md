### EnergyNet: Codebase of the paper "A methodological framework for optimizing the energy consumption of deep neural networks - A case study of a cyber threat detector"

---

#### **Project Overview**
This repository provides an **open-source methodological framework** designed to optimize the **energy consumption** of deep neural networks (DNNs) while maintaining system performance. The framework automates experimentation to minimize trial-and-error inefficiencies, systematically evaluates various optimization techniques, and generates models tailored to user-defined energy-performance trade-offs. 

This framework was validated using a real-world use case for cryptomining detection in a software-defined networking (SDN) controller, achieving up to **82% energy reduction** during inference with minimal accuracy loss.

For detailed methodology, see the paper: [Neural Computing and Applications](https://doi.org/10.1007/s00521-024-09588-z).

---

#### **Key Features**
- **Pre-configured Optimization Strategies**:
  - Model pruning
  - Weight quantization
  - Knowledge distillation
  - Neural architecture search (NAS)
- **Optimization Profiles**:
  - Energy-focused
  - Performance-focused
  - Balanced (user-defined trade-offs)
- **Extensibility**:
  - Easily integrate new optimization techniques or metrics.
- **Deployment-Ready Models**:
  - Optimized for production environments using TensorFlow Lite.

---

#### **Folder Structure**
```
├── src/
│   ├── training/                  # Scripts for model training
│   ├── optimization/              # Optimization algorithms
│   ├── profiling/                 # Energy and performance profiling tools
│   ├── deployment/                # Export to production-ready formats
│   └── utils/                     # Helper functions and utilities
├── configs/
│   └── optimization_profiles.yaml # Configuration for optimization profiles
└── README.md                      # Project README
```

---

#### **Getting Started**

##### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/amitkbatra/EnergyNet.git
   cd EnergyNet
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify installation:
   ```bash
   python src/main.py --help
   ```

##### **System Requirements**
- Linux-based OS
- CPU supporting Intel RAPL interface
- Python 3.10 or later

---

#### **Usage**

1. **Train a Baseline Model**:
   ```bash
   python src/training/train.py --config configs/baseline_config.yaml
   ```

2. **Apply Optimization Techniques**:
   ```bash
   python src/optimization/optimizers.py --strategy pruning --input_model path/to/model
   ```

3. **Evaluate Energy-Performance Trade-offs**:
   ```bash
   python src/profiling/profiler.py --input_model path/to/optimized_model
   ```

4. **Deploy Optimized Model**:
   ```bash
   python src/deployment/exporter.py --input_model path/to/optimized_model
   ```

---

#### **Configuration**
Modify `configs/optimization_profiles.yaml` to define custom optimization strategies or profiles:
```yaml
profiles:
  EXP0:
    post_training_optimizations: null
    training_aware_optimizations: null
  EXP1:
    post_training_optimizations:
      - full_integer_quantization
    training_aware_optimizations: null
  EXP2:
    post_training_optimizations:
      - float16_quantization
    training_aware_optimizations: null
  EXP3:
    post_training_optimizations:
      - float16_int8_quantization
    training_aware_optimizations: null
  EXP4:
    post_training_optimizations: null
    training_aware_optimizations:
      - pruning
  EXP5:
    post_training_optimizations: null
    training_aware_optimizations:
      - quantization_aware_training
  EXP6:
    post_training_optimizations: null
    training_aware_optimizations:
      - knowledge_distillation
  EXP7:
    post_training_optimizations: null
    training_aware_optimizations:
      - pruning
      - quantization_aware_training
  EXP8:
    post_training_optimizations: null
    training_aware_optimizations:
      - knowledge_distillation
      - pruning
  EXP9:
    post_training_optimizations: null
    training_aware_optimizations:
      - knowledge_distillation
      - quantization_aware_training
  EXP10:
    post_training_optimizations: null
    training_aware_optimizations:
      - knowledge_distillation
      - pruning
      - quantization_aware_training
  EXP11:
    post_training_optimizations: null
    training_aware_optimizations:
      - pruning
      - float16_quantization
  EXP12:
    post_training_optimizations: null
    training_aware_optimizations:
      - knowledge_distillation
      - float16_quantization
  EXP13:
    post_training_optimizations: null
    training_aware_optimizations:
      - knowledge_distillation
      - pruning
      - float16_quantization
```

---

#### **Key Dependencies**
- TensorFlow (>=2.9.2)
- TensorFlow Model Optimization
- psutil
- powerstat
- PyYAML

---

#### **Example**
Optimize a DNN model using the EXP7 profile:
```bash
python src/main.py \
    --input_model models/baseline.h5 \
    --profile EXP7 \
    --output_model models/optimized.h5
```

---

#### **Results**
View optimization results and energy metrics in:
```
results/
├── energy_metrics.csv
├── performance_summary.json
└── optimization_logs.txt
```

---

#### **Contributing**
We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

---

#### **License**
This project is licensed under the MIT License. See `LICENSE` for details.

---

#### **Acknowledgments**
This framework was developed as part of the research project *"A methodological framework for optimizing the energy consumption of deep neural networks"* by Amit Karamchandani, Alberto Mozo, Sandra Gómez-Canaval, and Antonio Pastor.

---

### **Citation**

If you are referencing the methodological framework described in this project, you can use the following citation statement:

APA Style: Karamchandani, A., Mozo, A., Gómez-Canaval, S., & Pastor, A. (2024). A methodological framework for optimizing the energy consumption of deep neural networks: A case study of a cyber threat detector. Neural Computing and Applications, 36(17), 10297–10338. https://doi.org/10.1007/s00521-024-09588-z

BibTeX:
```bibtex
@article{Karamchandani2024EnergyOptimization,
  author  = {Karamchandani, Amit and Mozo, Alberto and Gómez-Canaval, Sandra and Pastor, Antonio},
  title   = {A methodological framework for optimizing the energy consumption of deep neural networks: A case study of a cyber threat detector},
  journal = {Neural Computing and Applications},
  year    = {2024},
  volume  = {36},
  number  = {17},
  pages   = {10297--10338},
  doi     = {10.1007/s00521-024-09588-z}
}
```

---

#### **Contact**
For questions or feedback, please contact [amitkbatra@upm.es](mailto:amitkbatra@upm.es).
