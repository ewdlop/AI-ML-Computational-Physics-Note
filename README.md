# README

[![Dependabot Updates](https://github.com/ewdlop/AI-ML-Computational-Physics-Note/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/ewdlop/AI-ML-Computational-Physics-Note/actions/workflows/dependabot/dependabot-updates)
[![CodeQL](https://github.com/ewdlop/AI-ML-Computational-Physics-Note/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/ewdlop/AI-ML-Computational-Physics-Note/actions/workflows/github-code-scanning/codeql)

## Overview

```python
import time

def show_pleased():
    happy_face = '''
    ╔════════════╗
    ║   ╭────╮   ║
    ║  (ᵔᴥᵔ)    ║
    ║   \＿＿/    ║
    ╚════════════╝
    '''
    for _ in range(3):
        print("\033[93m" + happy_face + "\033[0m")  # Yellow color
        time.sleep(0.5)
        print("\033[91m" + happy_face.replace('ᵔᴥᵔ', '♥‿♥') + "\033[0m")  # Red heart eyes
        time.sleep(0.5)

    print("\n🌈 You're awesome! Have a wonderful day! 🌟")

show_pleased()
```

This repository contains various projects related to AI, ML, and computational physics. Below is a detailed description of the repository's content, including links to relevant submodules and external resources.

## Reinforcement Learning

Q-Learning written in C#. Made with Unity.
![Q-Learning](https://github.com/ewdlop/AI-Machine-Learning/assets/25368970/792b11b2-f4e5-44a2-bb99-2d661f3a077b)

## Natural Language Processing (NLP)

- [LangChain](https://docs.langchain.com/docs/)
- [Semantic-kernel](https://learn.microsoft.com/en-us/semantic-kernel/overview/)

## Computational Physics

- [Computational-Physics-Notes](https://github.com/ewdlop/Computational-Physcis-Notes)

## ComfyUI + ComfyUIManager + CIVITAI

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
- [CIVITAI](https://civitai.com/)
- [stable-diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui?tab=readme-ov-file)

## AIMA

The `AIMA` directory contains C# implementations of AI algorithms and models.

## Azure

The `Azure` directory contains projects related to Azure services and AI.

## BackgroundRemoval

The `BackgroundRemoval` directory contains scripts for background removal in images.

## Bayesian

The `Bayesian` directory contains Bayesian inference and modeling projects.

## bot

The `bot` directory contains chatbot implementations.

## CNTK

The `CNTK` directory contains projects using Microsoft Cognitive Toolkit.

## ComplexDynamics

The `ComplexDynamics` directory contains projects related to complex dynamic systems.

## DiffSharpProject

The `DiffSharpProject` directory contains projects using DiffSharp for automatic differentiation.

## Keras

The `Keras` directory contains projects using Keras for deep learning.

## LangChain

The `LangChain` directory contains projects using LangChain for NLP.

## MLDotNet

The `MLDotNet` directory contains projects using ML.NET for machine learning.

## NamedEntityRecognitionAndQuestionAnswering

The `NamedEntityRecognitionAndQuestionAnswering` directory contains projects for NER and QA.

## Neo4J

The `Neo4J` directory contains projects using Neo4j graph database.

## NeuralNetwork

The `NeuralNetwork` directory contains neural network implementations.

## NLP

The `NLP` directory contains various NLP projects.

## Notebook

The `Notebook` directory contains Jupyter notebooks for various ML and AI tasks.

## NTK

The `NTK` directory contains projects using NTK for neural tangent kernels.

## OpenCV

The `OpenCV` directory contains projects using OpenCV for computer vision.

## Pytorch

The `Pytorch` directory contains projects using PyTorch for deep learning.

## Q&A Chatbot Using LLM

The `Q&A Chatbot Using LLM` directory contains a Q&A chatbot implementation using LangChain and OpenAI.

## scikit-learn

The `scikit-learn` directory contains projects using scikit-learn for machine learning.

## SemanticKernel

The `SemanticKernel` directory contains projects using Semantic Kernel for AI.

## StableDiffusion

The `StableDiffusion` directory contains projects for stable diffusion models.

## TensorFlow

The `TensorFlow` directory contains projects using TensorFlow for deep learning.

## TorchSharpProject

The `TorchSharpProject` directory contains projects using TorchSharp for deep learning.

## USQL

The `USQL` directory contains projects using U-SQL for data processing.

## Submodules

- [MachineLearningInUnity](https://github.com/ewdlop/MachineLearningInUnity)
- [Data-Mining](https://github.com/ewdlop/Data-Mining)

## External Resources

- [NLTK Data](https://www.nltk.org/data.html#)
- [Neo4j Generative AI](https://neo4j.com/generativeai/?utm_source=discovery&utm_medium=PaidSearch&utm_campaign=GDB&utm_content=AMS-X-Awareness-Evergreen-Image&utm_term=&gclid=Cj0KCQiA5fetBhC9ARIsAP1UMgE2B8Ju-MmYgfsGp8mgkoEO7e8Q0ALl55NgRNuXfGUB_xKMrWuDB6EaAh3FEALw_wcB)
- [ONNX Runtime C# API Doc](https://onnxruntime.ai/docs/api/csharp/api)
- [Get Started with C# in ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-csharp.html)
- [Hugging Face Stable Diffusion Blog](https://huggingface.co/blog/stable_diffusion)
- [Stable Diffusion C# Tutorial for this Repo](https://onnxruntime.ai/docs/tutorials/csharp/stable-diffusion-csharp.html)

## Additional Information

### AI Workflow MLops

AI code to Hardware. transpiling.. compiler.. pipelining..

**XLA, Glow, TVM, nGraph, TensorRT, and IREE** are all frameworks and compilers used to optimize and accelerate machine learning models on various hardware backends. Here's an overview of each:

#### 1. **XLA (Accelerated Linear Algebra)**
   - **Developed by**: Google.
   - **Purpose**: XLA is a domain-specific compiler designed to optimize computations for machine learning, specifically for TensorFlow.
   - **How it works**: XLA takes computational graphs from TensorFlow and compiles them into optimized code for specific hardware like CPUs, GPUs, and TPUs (Tensor Processing Units). It does this by optimizing tensor operations, fusing operations, and reducing memory overhead.
   - **Target Hardware**: CPUs, GPUs, TPUs.
   - **Main Use Case**: Optimizing TensorFlow model performance, reducing inference time, and memory footprint.

#### 2. **Glow (Graph Lowering Compiler)**
   - **Developed by**: Facebook (now Meta).
   - **Purpose**: Glow is a machine learning compiler that optimizes neural network computations by lowering high-level graph representations into machine code.
   - **How it works**: Glow divides its compilation process into two phases:
     - High-level optimizations on the computational graph.
     - Low-level optimizations targeting specific hardware, converting to optimized machine code.
   - **Target Hardware**: CPUs, GPUs, and custom accelerators.
   - **Main Use Case**: Used for optimizing the performance of PyTorch models and various ML workloads on different hardware platforms.

#### 3. **TVM (Tensor Virtual Machine)**
   - **Developed by**: Apache (open-source project).
   - **Purpose**: TVM is a deep learning compiler that optimizes machine learning models and automates code generation for multiple hardware backends.
   - **How it works**: TVM uses machine learning techniques to generate highly optimized code for running models on hardware like CPUs, GPUs, and specialized accelerators (such as FPGAs and ASICs). It focuses on portability and performance optimization.
   - **Target Hardware**: CPUs, GPUs, FPGAs, ASICs, and more.
   - **Main Use Case**: TVM is used for optimizing deep learning models and ensuring efficient deployment on a variety of hardware platforms, especially in the context of edge computing and specialized AI hardware.

#### 4. **nGraph**
   - **Developed by**: Intel.
   - **Purpose**: nGraph is a deep learning compiler designed to optimize neural network computations for Intel hardware, including CPUs, GPUs, and FPGAs.
   - **How it works**: nGraph takes a deep learning model, optimizes it, and then generates machine code tailored to Intel hardware. It integrates with frameworks like TensorFlow and PyTorch.
   - **Target Hardware**: Intel CPUs, Intel GPUs, Intel FPGAs.
   - **Main Use Case**: Optimizing deep learning model inference on Intel hardware, specifically aiming for performance improvements in Intel architectures.

#### 5. **TensorRT (NVIDIA TensorRT)**
   - **Developed by**: NVIDIA.
   - **Purpose**: TensorRT is a deep learning inference library and optimizer designed to accelerate deep learning models on NVIDIA GPUs.
   - **How it works**: TensorRT optimizes pre-trained models by fusing layers, quantizing to lower precision (like FP16 or INT8), and generating highly efficient code for NVIDIA hardware.
   - **Target Hardware**: NVIDIA GPUs, including their specialized AI hardware (e.g., Jetson, Tesla).
   - **Main Use Case**: TensorRT is primarily used to optimize and accelerate the inference phase of deep learning models for production use, especially in high-performance applications like autonomous driving, robotics, and high-performance computing.

#### 6. **IREE (Intermediate Representation Execution Environment)**
   - **Developed by**: Google.
   - **Purpose**: IREE is a compiler and runtime for machine learning models, aimed at enabling efficient and portable execution across different hardware backends.
   - **How it works**: IREE converts machine learning models into an intermediate representation (IR) that can be optimized for different hardware backends (like CPUs, GPUs, or specialized accelerators). The focus is on high portability and performance across many platforms, including mobile, edge, and cloud environments.
   - **Target Hardware**: CPUs, GPUs, DSPs, and other specialized accelerators.
   - **Main Use Case**: IREE is designed for running machine learning models in real-time and resource-constrained environments, making it ideal for edge computing, mobile devices, and embedded systems.

### Comparison and Key Differences

| **Framework**  | **Purpose**                          | **Target Hardware**             | **Key Features**                                             | **Main Use Case**                                    |
|----------------|--------------------------------------|---------------------------------|--------------------------------------------------------------|-----------------------------------------------------|
| **XLA**        | TensorFlow graph optimization        | CPUs, GPUs, TPUs                | Tensor operation fusion, low-level hardware optimizations     | Optimizing TensorFlow models on TPUs and GPUs       |
| **Glow**       | Lowering computation graphs to code  | CPUs, GPUs, accelerators        | High- and low-level optimizations, PyTorch integration        | Optimizing PyTorch models on various hardware       |
| **TVM**        | General ML model optimization        | CPUs, GPUs, FPGAs, ASICs        | Automatic code generation, cross-hardware optimization        | Efficient deep learning model deployment on edge    |
| **nGraph**     | Intel-specific model optimization    | Intel CPUs, GPUs, FPGAs         | Optimizations for Intel architectures, supports TensorFlow    | Accelerating deep learning on Intel hardware        |
| **TensorRT**   | NVIDIA GPU optimization              | NVIDIA GPUs                     | Layer fusion, precision reduction (FP16/INT8), fast inference | High-performance inference on NVIDIA GPUs           |
| **IREE**       | Portable ML execution environment    | CPUs, GPUs, DSPs, accelerators  | Portable IR, optimized execution for mobile and edge devices  | Efficient ML execution on mobile/embedded systems   |

### Summary
- **XLA** is tightly integrated with TensorFlow and is excellent for optimizing models for TPUs and GPUs.
- **Glow** works well with PyTorch and focuses on graph lowering and optimization.
- **TVM** is a highly portable solution that can optimize across a wide range of hardware, from CPUs to ASICs.
- **nGraph** is Intel’s solution for optimizing deep learning workloads on their own hardware.
- **TensorRT** is specifically designed for fast inference on NVIDIA GPUs, making it ideal for production environments where performance is critical.
- **IREE** provides a portable execution environment with a focus on efficient performance across a wide variety of devices, including mobile and edge platforms.

Each of these tools plays a significant role in the **optimization of machine learning models**, often targeting different hardware and use cases, from cloud computing to mobile devices and edge AI deployment.
