# Tiny-AAResUNet: An Attention Augmented Convolution-based Tiny-Residual UNet for Road Extraction
Abstract: Recently remote sensing images have become more popular due to improved image quality and resolution. These images have been shown to be a valuable data source for road extraction applications like intelligent transportation systems, road maintenance, and road map making. In recent decades, the use of highly significant deep learning in automatic road extraction from these images has been a hot research area. However, highly accurate road extractions from remote sensing images remain a challenge because they are cluttered in the background and have widely different shapes and complex connectivities. This paper proposes novel tiny attention augmented convolution-based residual UNet architecture (Tiny-AAResUNet) for road extraction, which adopts powerful features of the self-attention mechanism and advantageous properties of residual UNet structure. The self-attention mechanism uses attention-augmented convolutional operation to capture long-range global information; however, traditional convolution has a fundamental disadvantage: it only performs on local information. Therefore, we use the attention-augmented convolutional layer as an alternative to standard convolution layers to obtain more discriminant feature representations. It allows the development of a network with fewer parameters. We also adopt improved residual units in standard ResUNet to the speedup training process and enhance the segmentation accuracy of the network. Experimental results on Massachusetts, DeepGlobe Challenge, and UAV Road Dataset show that the Tiny-AAResUNet performs well in road extraction, with Intersection over Union (IoU) (94.27%), lower trainable parameters (1.20 M), and inference time (1.14 sec). Comparative results on the proposed method have outperformed in road extraction with ten recently established deep learning approaches.

# Note:
The research paper will be uploaded after acceptance from the publication.

# Dataset Link:
1. Massachusetts Road Dataset: [Here](https://www.cs.toronto.edu/~vmnih/data/)
2. DeepGlobe Challenge Road Dataset: [Here](https://competitions.codalab.org/competitions/18467)
3. UAV Road Dataset: [Here](https://zenodo.org/record/7020196)

# Proposed Architecture Design:
![Tiny-AAResUNet Arch](https://github.com/patilparam/AA-ResUNet/assets/49902973/6f98f656-eef6-4b33-888a-4602dc676885)

# Paper Result on Massachusetts Road Dataset:
![Massachusetts_Results](https://github.com/patilparam/AA-ResUNet/assets/49902973/56e946b1-73b4-46e7-b9ef-4bd670359392)

# Paper Result on DeepGlobe Challenge Dataset:
![DeepGlobe_result](https://github.com/patilparam/AA-ResUNet/assets/49902973/58af69ff-5037-4743-92b8-8a92811be484)

# Ablation Result on UAV Road Dataset:
![ablation_result_uav](https://github.com/patilparam/AA-ResUNet/assets/49902973/0e14ad8d-70f1-4985-8ac3-443f3e48a8c0)

# Conclusion

This paper explores the advantage of self-attention for the road extraction model as a replacement for standard convolutions. We propose a novel two-dimensional approach of an individual self-attention mechanism for complex road structure satellite images that allows competitive, self-attentional semantic segmentation to be trained on road extraction tasks. We propose a self-attention mechanism based on the attention-augmented convolution layer in road region segmentation, and we prove that the proposed method outperforms the existing attention approaches. Comprehensive experiments show that the proposed Tiny-AAResUNet improves road extraction results consistently over a wide range of architectures and computing costs. Additionally, we used depthwise separable convolution, and attention-augmented convolution to speed up the training process and enhance the segmentation performance of Tiny-AAResUNet. We adopted the ResUNet architecture because of its extensive use and flexibility to scale across various computational constraints. Once each residual block of the original ResUNet, we use attention-augmented convolution with depthwise separable convolution. The proposed architecture uses the benefit of a combination of residual learning strength and UNet structure. This characteristic facilitates training and the development of basic yet effective neural networks. The proposed network beats ResUNet, and many other residual deep-learning road extraction algorithms in terms of parameters, GFLOPs, and inference speed.
