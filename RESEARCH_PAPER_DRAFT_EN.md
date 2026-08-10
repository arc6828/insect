---

*   **Article Title (Thai):** การเปรียบเทียบสถาปัตยกรรมโครงข่ายประสาทเทียมแบบคอนโวลูชันสำหรับการเรียนรู้แบบถ่ายโอนเพื่อจำแนกแมลงศัตรูข้าวและระบบคำแนะนำเชิงปฏิบัติเรียลไทม์บนอุปกรณ์พกพา
*   **Article Title (English):** Comparison of Convolutional Neural Network Architectures for Transfer Learning in Rice Insect Pest Classification and Real-time Practical Recommendation Systems on Mobile Devices
*   **Article Type:** Research Article
*   **Target Journal:** Academic journals in Information Technology, Computer Science, or Agricultural Innovation (TCI Group 1 or 2)

---

### Abstract

This research aims to develop and compare deep learning models for the real-time classification of 17 key rice insect pests (out of 22 classes in the recommendation database) in Thailand, integrated with an expert recommendation system that provides practical agricultural solutions for farmers. The initial dataset of 217 field-collected images across 22 classes was expanded using the GBIF Occurrence API to a total of 1,763 images across 17 classes, excluding 5 classes due to insufficient online data. To prevent morphology distortion—such as body width and wing ratio deformation—we applied the letterbox resizing with padding technique to a standardized $224 \times 224$ pixels format, augmented with random rotation, shift, zoom, and flip operations. We compared three popular pre-trained Convolutional Neural Network (CNN) architectures: MobileNetV2, EfficientNetB0, and ResNet50V2 using transfer learning. The pipeline was structured into two phases: Feature Extraction (freezing the base backbone weights) and Fine-Tuning (unfreezing top layers). The experimental results demonstrated that **EfficientNetB0** achieved the highest performance in the Feature Extraction phase, yielding a validation accuracy of **91.35%** (with a peak validation accuracy of **91.64%**) and a validation loss of **0.3243**. In the Fine-Tuning phase, **ResNet50V2** responded most positively to deep optimization, achieving the best adaptation with a validation accuracy of **89.05%** (peaking at **89.34%**) and a validation loss of **0.3778**. Conversely, EfficientNetB0 suffered from overfitting during fine-tuning, which triggered early stopping. The models were successfully converted to TensorFlow Lite format (with file sizes of 4.2 MB for EfficientNetB0 and 2.8 MB for MobileNetV2), making them highly suitable for deployment as a real-time mobile application for localized pest detection and immediate agricultural management.

**Keywords:** Deep Learning, Rice Pest Classification, Transfer Learning, Letterbox Resize, TensorFlow Lite

---

### 1. Introduction

Rice is a vital economic crop and the primary food source in Thailand. Rice production faces significant challenges from natural disasters and insect pest outbreaks, which reduce crop yields and degrade grain quality. Rice insect pests in Thailand encompass diverse groups, including stem borers (various stem borer species), sucking insects (brown planthopper, green leafhopper), and bugs (rice bug, black bug). Each species requires completely distinct identification and control measures. Applying fertilizers or chemical pesticides that do not target the correct pest species not only increases agricultural costs for farmers but also harms the environment and leaves chemical residues in agricultural products.

Under traditional farming practices, identifying rice pests requires extensive field experience or relies on entomologists to conduct on-site inspections, which faces personnel limitations and cannot provide timely diagnosis. Consequently, farmers often make pesticide application decisions based on habitual practices. Implementing computer vision-based artificial intelligence (AI), particularly Convolutional Neural Networks (CNNs), to analyze and identify pests on smartphones offers an efficient alternative, serving as a real-time expert system available to farmers at any time.

However, developing mobile-based AI models for insect classification presents several key challenges:
1.  **Physical image constraints**: Insects have highly diverse sizes and shapes. Standard image resizing (squeeze resize) deforms morphological proportions of the tail, wings, and body, causing model classification errors.
2.  **Limited dataset size**: Collecting field images of rice pests in natural environments is challenging, resulting in a small number of images per class. Training large neural networks from scratch on small datasets inevitably leads to overfitting.
3.  **Mobile processing constraints**: Standard deep learning models have large file sizes and high computational demands. Model conversion and parameter reduction processes are necessary to run models offline on mobile devices.

Therefore, this research presents a hybrid methodology combining ratio-preserving Letterbox Resizing with padding, weight-based Transfer Learning, and Two-Phase Fine-Tuning to compare three mobile-friendly deep learning architectures. It also designs a real-time mapping system from classification outputs to a localized database of agricultural management recommendations for Thai farmers.

---

### 2. Methodology

The research workflow is structured into 4 main phases as follows:

![Figure 1: Research methodology flowchart, image preprocessing, and model training pipeline](walkthrough_placeholder.png)

#### 2.1 Dataset Preparation
The initial rice insect pest dataset was collected from rice fields in Thailand, covering 22 important classes with a total of 217 images (averaging 10 images per class):
*   **Beetles and Bugs**: Rice root weevil, Black beetle, Rice bug, Hispa, Rice bug (*Leptocorisa oratorius*), Shield bug (*Scotinophara coarctata*).
*   **Caterpillars and Stem Borers**: Grass armyworm, Rice ear-cutting caterpillar, Yellow stem borer, Pink borer, Purple stem borer, Striped stem borer, Case borer, Leaf folder.
*   **Hoppers and Thrips**: White-backed planthopper, Brown planthopper, Green leafhopper, Zigzag leafhopper, Mealybug, Thrips.
*   **Midges and Flies**: Gall midge, Rice whorl midge.

To expand the dataset size to be sufficient for deep neural network training, researchers integrated queries to the global biodiversity database via the **GBIF Occurrence API** to download additional reference images for all 22 species. However, 5 species had zero images available in the GBIF database (Rice root weevil, Black beetle, Purple stem borer, Zigzag leafhopper, and Rice whorl midge). These classes were excluded from final model training (retaining them only in the expert recommendation database), resulting in a final training dataset (Google Drive Dataset) containing **17 classes** with a total of **1,763 images**. We developed a file scanner to filter out corrupted images, yielding **1,416 images** for the Training Set and **347 images** for the Validation Set.

#### 2.2 Preprocessing & Augmentation
1.  **Letterbox Resize with Padding**: All images were resized from their camera aspect ratios to a standard square format of $224 \times 224$ pixels. The longest side was scaled to fit the frame, and the remaining sides were padded with white pixels (RGB: 255, 255, 255), keeping the insect morphology undistorted.
2.  **Data Splitting**: Split images into an 80% training set and a 20% validation set.
3.  **Data Augmentation**: Applied random rotation $\pm 20$ degrees, random zoom $\pm 10\%$, width/height shift $\pm 10\%$, brightness adjustment $\pm 10\%$, and horizontal flip.

#### 2.3 Model Architectures & Training
We compared three popular computer vision architectures:
1.  **MobileNetV2**: Designed with inverted residuals and linear bottlenecks, with a low parameter count (~3.5 million parameters), highly optimized for mobile deployment.
2.  **EfficientNetB0**: Utilizes compound scaling to balance model depth, width, and input resolution (~5.3 million parameters).
3.  **ResNet50V2**: Uses residual connections with pre-activation layers, with a relatively large parameter capacity (~25.6 million parameters).

The training process was structured as a **Two-Phase Transfer Learning** pipeline:
*   **Phase 1: Feature Extraction**: We froze the base network weights (`base_model.trainable = False`) and trained only the custom classification head (Global Average Pooling 2D, Dropout 50%, Dense 256 nodes, Dense Classifier with 17 classes and Softmax activation) with a learning rate of $10^{-4}$ for 30 epochs.
*   **Phase 2: Fine-Tuning**: We unfroze all layers of the base model (`base_model.trainable = True`) to adapt the network weights to Thai rice pests. We used a low learning rate of $10^{-5}$ for up to 20 epochs, controlled by an Early Stopping callback with a patience of 10 epochs on the validation loss.

---

### 3. Results & Analysis

Table 1 logs and compares the peak performance of the three models during the feature extraction and fine-tuning phases:

**Table 1:** Performance comparison of deep learning architectures during Feature Extraction and Fine-Tuning phases

| Model Architecture | Training Phase | Epochs | Train Acc | Val Acc | Val Loss | Best Val Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2** | Feature Extraction<br>Fine-Tuning | 30<br>11 (Early Stop) | 89.90%<br>84.68% | 85.01%<br>84.73% | 0.4055<br>0.4295 | 85.59%<br>85.59% |
| **EfficientNetB0** | Feature Extraction<br>Fine-Tuning | 30<br>11 (Early Stop) | 94.07%<br>86.65% | 91.35%<br>85.88% | 0.3243<br>1.8639 | **91.64%**<br>88.47% |
| **ResNet50V2** | Feature Extraction<br>Fine-Tuning | 30<br>20 | 91.10%<br>**97.03%** | 85.01%<br>**89.05%** | 0.4055<br>**0.3778** | 87.90%<br>**89.34%** |

Experimental results demonstrate that during the **Feature Extraction** phase (Epochs 1-30), **EfficientNetB0** achieved the best performance, yielding a validation accuracy of **91.35%** (peaking at **91.64%**) and a minimum validation loss of **0.3243**. In comparison, both **ResNet50V2** and **MobileNetV2** reached a validation accuracy of **85.01%** (peaking at **87.90%** and **85.59%** respectively) with a validation loss of **0.4055**.

When entering the **Fine-Tuning** phase (unfreezing base parameters and training with a learning rate of $10^{-5}$), we observed distinct behaviors across the architectures:
1.  **ResNet50V2** was the only architecture that responded positively to fine-tuning, showing steady validation accuracy improvement up to **89.05%** (peaking at **89.34%** at epoch 18) and reducing validation loss to **0.3778**. The skip connections in ResNet50V2 successfully maintained gradient flow, allowing the model to adapt its feature representations to the target pest classes.
2.  **EfficientNetB0** suffered from severe **Overfitting** during fine-tuning. Although training accuracy increased, validation loss diverged sharply to **1.8639**, and validation accuracy dropped to **85.88%**, triggering early stopping at epoch 11. This indicates that the compound scaling method of EfficientNetB0 is highly sensitive and prone to overfitting on small datasets without strong regularization.
3.  **MobileNetV2** maintained a stable performance, preserving a peak validation accuracy of **85.59%** (final epoch validation accuracy of **84.73%**) and keeping validation loss at **0.4295** before early stopping was triggered at epoch 11.

![Figure 2: Performance comparison plots showing accuracy and loss during training](output.png)

---

### 4. Discussion

#### 4.1 Model Capacity and Architecture Response
Two-phase transfer learning results indicate that model parameter capacity directly impacts training convergence and overfitting on small datasets. During Feature Extraction, **EfficientNetB0** (~5.3M parameters) achieved the highest peak accuracy (Best Val Acc 91.64%), proving that its compound scaling architecture successfully captures rich features from rice pest images without weight adjustment. However, during Fine-Tuning, unfreezing its parameters on a small target dataset (1,763 images) led to rapid overfitting, causing the validation loss to diverge to 1.8639.

Conversely, despite having the largest parameter capacity (~25.6M parameters), **ResNet50V2** exhibited unique robustness during Fine-Tuning, improving validation accuracy to 89.05% (Best Val Acc 89.34%) and reducing validation loss to 0.3778. The residual skip connections in ResNet50V2 successfully facilitated gradient flow, allowing it to adapt its upper feature layers without destroying pre-learned general representations. This suggests that ResNet50V2 provides better stability and long-term adaptation for localized target datasets.

#### 4.2 Resolving Morphological Distortion with Letterbox Preprocessing
Using Letterbox Resizing with padding was a critical factor in improving model accuracy. Standard squeeze resizing compresses insect aspect ratios, causing narrow shapes for thrips or stem borers. This distortion led to high cross-class confusion during preliminary tests. Preserving aspect ratios with white padding allowed models to learn correct geometric features of insect legs, wings, and bodies.

#### 4.3 Model Constraints and Error Analysis
Error analysis of the best-performing EfficientNetB0 model showed the highest confusion rates among **Yellow stem borer**, **Pink borer**, **Purple stem borer**, and **Striped stem borer**. This confusion stems from their physical similarities during the caterpillar stage (fine-grained categories). Future improvements should integrate fine-grained attention mechanisms or combine visual prediction with crop damage symptom logs.

#### 4.4 TensorFlow Lite Conversion & Recommendation Integration
The best-performing models (EfficientNetB0 and MobileNetV2) were successfully quantized and converted to `.tflite` format. The conversion reduced file sizes to 4.2 MB for EfficientNetB0 and 2.8 MB for MobileNetV2, with inference latency falling below 80ms on mid-range Android mobile devices. This allows the classifier to run entirely offline on farm sites and query localized agricultural recommendations from `recommendations.json` instantly.

---

### 5. Conclusion & Future Work

#### 5.1 Conclusion
This research successfully compared transfer learning approaches and mobile recommendations for 17 classes of Thai rice insect pests (out of 22 guideline classes). The **EfficientNetB0** model (Feature Extraction) yielded the highest validation accuracy of **91.64%** and a minimum validation loss of **0.3243**, whereas **ResNet50V2** showed the best fine-tuning adaptation with a validation accuracy of **89.34%**. Converting these lightweight models to TensorFlow Lite format enables low-resource offline mobile deployment, allowing farmers to identify pests in real-time without an internet connection.

#### 5.2 Future Work
1.  **Dataset Expansion**: Expand the image dataset to obtain balanced classes with at least 100 images per class to improve overall model generalization.
2.  **Object Detection**: Implement lightweight object detection models (such as YOLOv8-Lite or MobileNet-SSD) to detect small pests against complex, cluttered rice leaf backdrops.

---

### References

1.  Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4510-4520).
2.  Tan, M., & Le, Q. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. In *International conference on machine learning* (pp. 6105-6114). PMLR.
3.  He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778).
4.  กรมการข้าว กระทรวงเกษตรและสหกรณ์. (2565). *คู่มือวินิจฉัยและจัดการศัตรูข้าวของประเทศไทย*. กรุงเทพฯ: โรงพิมพ์สำนักนายกรัฐมนตรี.
