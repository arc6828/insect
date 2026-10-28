# ARCHITECTURE-DEPENDENT EFFECTS OF FINE-TUNING ON RICE INSECT PEST CLASSIFICATION: A COMPARISON OF MOBILENETV2, EFFICIENTNETB0, AND RESNET50V2

**Authors:** Natradee Anupong, Chavalit Koweerawong, and Daorathar Weerapan\*  
**Affiliation:** Computer Science Program, Faculty of Science and Technology, Valaya Alongkorn Rajabhat University under the Royal Patronage, Pathum Thani, 13180, Thailand  
**E-mail & ORCID:**  
- `natradee@vru.ac.th`, ORCID iD: 0009-0007-2848-2098  
- `chavalit@vru.ac.th`, ORCID iD: 0009-0008-3048-4225  
- `daorathar@vru.ac.th`, ORCID iD: 0009-0001-3625-3806 (\*Corresponding author)  

*Received: ................, ......; Revised: ................, ......; Accepted: ................, ...... [For editorial use only]*  
*DOI: https://doi.org/10.55766/sujstxxxxx [for editorial use only]*  
*Suranaree J. Sci. Technol. xx(x):xxx-xxx [for editorial use only]*  
*Running Head: Provide a shortened version of the title, not exceeding 60 characters, including spaces. (9 pt)*

---

## Abstract

Accurate identification of insect pests is the first step in integrated pest management for rice, yet access to entomological expertise in the field is limited. This study compared MobileNetV2, EfficientNetB0, and ResNet50V2 on a dataset of 1,763 images representing 17 rice insect pest species retrieved through taxonomically resolved records from the Global Biodiversity Information Facility (GBIF). All models were trained using the same two-phase protocol: feature extraction with a frozen backbone followed by full-backbone fine-tuning at a lower learning rate. With the frozen backbone, EfficientNetB0 achieved the highest validation accuracy (92.80%). Fine-tuning produced architecture-dependent outcomes: ResNet50V2 improved from 86.46% to 90.20%, MobileNetV2 remained at 86.17%, and EfficientNetB0 declined to 89.63%. For EfficientNetB0, validation loss increased nearly sixfold within four epochs while training accuracy also fell, a pattern inconsistent with conventional overfitting. A taxonomic error analysis showed that 77.8% of misclassifications occurred within the same insect order, significantly more than the 37.5% expected if errors were spread evenly across all incorrect classes. These findings indicate that fine-tuning does not necessarily improve performance, that its effects differ across architectures. The exported fine-tuned EfficientNetB0 model achieved a macro-F1 score of 0.8835 and was converted to a 4.2 MB TensorFlow Lite file suitable for offline use.

**Keywords:** Fine-tuning, Image classification, Convolutional neural networks, Rice insect pests, Transfer learning

---

## Introduction

In Thailand, rice cultivation covers approximately 9,600 km2, and contributes substantially to national food security and rural household income. Thailand is also among the world’s ten largest rice producers and one of its leading rice exporters (Office of Agricultural Economics, 2026). Rice constitutes the dietary foundation for more than half of the global population and serves as the principal agricultural commodity in Southeast Asian economies. Rice production in Southeast Asia experiences recurring yield losses caused by insect pests, making accurate pest identification a fundamental first step in integrated pest management (IPM). However, distinguishing among visually similar species can be difficult even when their management requirements are comparable. For instance, brown planthoppers and white-backed planthoppers may call for near-identical interventions but their identification can depend on subtle morphological characteristics such as the pale marking on the pronotum. While confusing a leaffolder caterpillar with a stem borer leads to treatments applied to the wrong part of the plant at the wrong time. The limited availability of extension personnel with specialized entomological expertise further restricts access to reliable field identification, which is why image-based automatic identification has attracted sustained attention for over two decades (Martineau et al., 2017). In Thailand, the costs of incorrect pest management have been documented: insecticide-induced brown planthopper outbreaks in the central plain during the 2010–2012 crop seasons affected thousands of farmers, with reported yield losses of 4.3 t/ha in Ang Thong and Suphan Buri and 3.5 t/ha in Chainat (Escalada et al., 2015). Those outbreaks attributed to insecticide misuse, accurate identification that supports targeted rather than prophylactic spraying has direct economic value.

Image-based insect recognition has consequently developed from systems based on manually designed color, texture, and shape descriptors toward convolutional neural networks (CNNs). Early deep-learning approaches for paddy pests incorporated explicit localization before classification, reflecting the difficulty of separating insects from complex agricultural backgrounds. Liu et al. (2016) combined saliency-based localization with CNN classification for paddy-field pests, demonstrating the relevance of foreground isolation to pest recognition.

Transfer learning offers a practical approach when a target dataset is too small to train a deep network from scratch. Pretrained representations can provide useful visual features for target domains with comparatively few labeled examples (Pan & Yang, 2010; Kornblith et al., 2019). A conventional workflow first trains a classification head while keeping the pretrained feature extractor frozen and then unfreezes part or all of the backbone for fine-tuning. Although this procedure often improves target-domain performance, the results depend on several factors, including the target data, optimization procedure, and architectural components.

Deep convolutional networks have substantially improved image-based pest classification, although reported performance varies considerably across datasets. On IP102, the largest public pest benchmark, a fine-tuned ResNet50 reaches only 49.40% accuracy across 102 classes (Wu et al., 2019). In contrast, higher accuracy has been reported on more narrowly defined agricultural datasets. For example, Li et al. (2022) report 88.68% on 14 rice pest classes after GAN-based augmentation expanded their training set to approximately 14,000 images. These results illustrate the gap between large-scale benchmarks and the conditions often encountered in applied research, a pretrained backbone is adapted to a dataset of only a few thousand images with strong class imbalance. In such settings, the benefit of fine-tuning is usually assumed rather than verified.

However, this assumption may not always hold. During fine-tuning, the behavior of a pretrained model can be affected by how its internal components respond to the target dataset. If the target data differ substantially from the pretraining data, these changes may influence the quality of the learned representations and the model's final performance (Chollet, 2020; Ioffe & Szegedy, 2015). Training may appear to proceed normally, yet the resulting model may not achieve the expected improvement. In agricultural image classification, studies do not always describe the fine-tuning procedure in sufficient detail, making it difficult to determine which training configurations contribute to the reported results. Reporting these settings clearly would make it easier to compare results across studies and repeat the experiments.

The present study investigates this issue through a controlled comparison of three widely used CNN architectures: MobileNetV2, EfficientNetB0, and ResNet50V2. Each model uses a different approach to feature extraction. MobileNetV2 uses inverted residual blocks and linear bottlenecks to improve computational efficiency (Sandler et al., 2018). EfficientNetB0 combines compound scaling with mobile inverted bottleneck convolutional blocks and squeeze-and-excitation (SE) attention (Tan & Le, 2019; Hu et al, 2018). ResNet50V2 uses pre-activation residual connections to support effective training of deep neural networks (He et al., 2016).

### Image-based insect pest classification

Early insect pest classification systems relied on handcrafted features, such as the Scale-Invariant Feature Transform (SIFT), and the Histogram of Oriented Gradients (HOG). Martineau et al. (2017) reviewed these methods and discussed their limitations. Deep learning substantially raised the attainable accuracy. Liu et al. (2016) combined a saliency map with a deep CNN to localize and classify paddy field pests, the approach highlighted the importance of focusing on the insect rather than irrelevant background details. Wu et al. (2019) then established IP102 as the community benchmark dataset containing 102 insect pest classes. Their work also demonstrated the difficulty of classifying pests in real-world images, where even fine-tuned deep learning models achieved a top-1 accuracy of only around 50%. Work restricted to rice pests report considerably higher accuracy on cleaner datasets. Rahman et al. (2020) classified rice diseases and pests from field images with compact CNNs designed for deployment, and Li et al. (2022) built IP_RicePests, showing that transfer learning raised the accuracy of VGG16 to 84.39% and that ARGAN-based augmentation further raised it to 88.68% across 14 classes. Beyond pest classification at the broader category level, Hansen et al. (2020) demonstrated species-level classification of 361 carabid beetle species from 65,841 standardized museum images, reaching 51.9% species-level accuracy. Their findings indicated that classification becomes more difficult when species are taxonomically similar. More recently, transformer-based models such as GNViT have been applied to pest recognition (Venkatasaichandrakanth & Iyapparaja, 2024), although their data requirements make them better suited to large datasets than to small ones.

Research from Thailand has approached the problem from complementary angles. At the population level, Skawsang et al. (2019) forecast peak brown planthopper light-trap catches in the Central Plain over the 2006–2016 dry seasons by combining ground meteorological observations with satellite-derived crop phenology; an artificial neural network (RMSE 1.686 on log-transformed catches) outperformed random forest (1.737) and multiple linear regression (2.015). Such forecasts indicate when to scout but not which pest has been found. At the image level, Temniranrat et al. (2021) of the National Electronics and Computer Technology Center (NECTEC) and Kasetsart University deployed a YOLOv3 rice disease detector to about 470 farmers and specialists through a LINE chatbot, with five disease classes and 8,767 training images. For insects, Kittichai et al. (2021) showed that a two-stage YOLOv3 pipeline could identify the species and sex of mosquito vectors trapped in Kalasin, Bangkok, Chonburi, and Prachuap Khiri Khan with 98.9% accuracy from 10,564 images. To our knowledge, no comparable image-based classifier has yet been published for the insect pests of Thai rice paddies, a gap that the present study begins to address.

### Transfer learning and the fine-tuning

Transfer learning rests on the observation that features learned on large source datasets remain useful on small target datasets (Pan & Yang, 2010; Kornblith et al., 2019). A common approach is to train a new classification head while keeping the pretrained backbone frozen, then unfreeze and fine-tune at a low learning rate. However, the details of this process can affect the final performance, particularly when the target dataset is small or differs from the original training data. For example, batch normalization layers rely on statistics estimated from the training data, and updating these statistics on a small target dataset may disturb previously learned features (Chollet, 2020; Ioffe & Szegedy, 2015). The three CNN architectures examined in this study also differ in their internal designs. EfficientNetB0 uses mobile inverted bottleneck convolutional blocks (MBConv) and squeeze-and-excitation (SE) blocks to adjust the importance of different feature channels (Hu et al., 2018; Tan & Le, 2019). ResNet50V2 uses pre-activation residual units in which an identity shortcut carries the signal directly across each block (He et al., 2016). MobileNetV2 uses lightweight inverted residual blocks and has far fewer parameters than the other two models (Sandler et al., 2018). These architectural differences may influence how each model responds to fine-tuning on a small dataset. Therefore, this study compares the three architectures under the same experimental conditions.

### Research Overview

This study examines how fine-tuning affects the performance of three commonly used CNN architectures for rice pest classification: MobileNetV2, EfficientNetB0, and ResNet50V2. The models are evaluated using a taxonomically verified dataset of 17 rice insect pest species. To ensure a fair comparison, all models are trained using the same dataset, classification head, optimizer, data augmentation, and data split. The study follows the standard two-phase transfer learning protocol with the backbone fully unfrozen in phase two. This setting commonly used in applied work. The study also considers the practical deployment of the selected model through TensorFlow Lite for offline use under field conditions. Together, these analyses provide an empirical basis for understanding the effects of fine-tuning on different CNN architectures and evaluating their suitability for rice pest classification.

---

## Objectives

The objectives of this study were:
1. To compare the classification performance of MobileNetV2, EfficientNetB0, and ResNet50V2 under an identical two-phase transfer learning protocol on a taxonomically verified dataset of 17 rice insect pest species.
2. To analyze the taxonomic patterns of model misclassifications as a validity check on classification behavior and as a means of detecting potential label inconsistencies, and to assess the practical deployability of the selected model as a compact TensorFlow Lite file for offline use under field conditions.

---

## Materials and Methods

### Dataset construction

The study began with a target list of 22 economically important rice pest species compiled from the Thai Department of Agricultural Extension’s field manual. Rather than scraping images by keyword, we resolved each scientific name to a canonical taxon identifier through the GBIF Species Match API and retrieved occurrence records with still images through the Occurrence Search API, with a maximum of 150 images per species per species to prevent any single class from dominating the dataset. Checking the taxonomic information before downloading images helped revealed several naming issues. Three species could only be matched at the genus level, while three others required approximate name matching. In addition, *Recilia dorsalis* is treated as a junior synonym under the genus *Maiestas* in the current GBIF backbone. These cases highlight the risk of mislabeling when images are collected using scientific names alone. Each downloaded file was checked for integrity and named using its occurrence key to help identify duplicate records across repeated downloads.

Five of the 22 species returned no usable images at all and were excluded from the dataset. These included small insects and species with concealed life stages, such as the rice root weevil and the rice shoot fly, which may be less frequently represented in citizen-science image records. The final dataset contains 1,763 images across 17 species (Table 1, Figure 1). The number of images per class ranged from 13 (*Dicladispa armigera*) to the cap of 150, giving an imbalance ratio of 11.5:1 with a coefficient of variation of 50.2%, a long-tailed distribution similar from IP102 (Wu et al., 2019 ; Zhang et al., 2023). One additional issue concerns the class directory for *Sogatella furcifera* which contains a Thai spelling variant that affects the alphabetical ordering of class indices. Correcting the directory name may alter the assigned indices for two classes. Therefore, the original class mapping should be preserved, or an explicit label mapping should be applied, to ensure that the model predictions are interpreted correctly. If the class indices change during preprocessing, the model may need to be retrained or its output labels remapped accordingly.

**Table 1. Composition of the dataset**

| Species | Order / Family | Images | Val. n |
| :--- | :--- | :---: | :---: |
| *Tetroda denticulifera* | Hemiptera / Pentatomidae | 46 | 9 |
| *Spodoptera mauritia* | Lepidoptera / Noctuidae | 150 | 30 |
| *Mythimna separata* | Lepidoptera / Noctuidae | 148 | 29 |
| *Scirpophaga incertulas* | Lepidoptera / Crambidae | 148 | 29 |
| *Sesamia inferens* | Lepidoptera / Noctuidae | 65 | 13 |
| *Chilo suppressalis* | Lepidoptera / Crambidae | 56 | 11 |
| *Nymphula depunctalis* | Lepidoptera / Crambidae | 150 | 30 |
| *Cnaphalocrocis medinalis* | Lepidoptera / Crambidae | 146 | 29 |
| *Sogatella furcifera* | Hemiptera / Delphacidae | 150 | 30 |
| *Nilaparvata lugens* | Hemiptera / Delphacidae | 149 | 29 |
| *Nephotettix virescens* | Hemiptera / Cicadellidae | 75 | 15 |
| *Pseudococcus saccharicola* | Hemiptera / Pseudococcidae | 54 | 10 |
| *Stenchaetothrips biformis* | Thysanoptera / Thripidae | 144 | 28 |
| *Dicladispa armigera* | Coleoptera / Chrysomelidae | 13 | 2 |
| *Orseolia oryzae* | Diptera / Cecidomyiidae | 22 | 4 |
| *Leptocorisa oratoria* | Hemiptera / Alydidae | 150 | 30 |
| *Scotinophara coarctata* | Hemiptera / Pentatomidae | 97 | 19 |
| **Total** | | **1,763** | **347** |

*(Figure 1. Images per class, sorted)*

### Preprocessing, data split, and augmentation

Insect identification often hinges on body proportions, such as leg length relative to body length and the wing-to-abdomen ratio. To preserve these proportions, images were resized without distorting their aspect ratio horizontally or vertically. Each image was scaled so that its longer side measured 224 pixels, then placed at the center of a white $224 \times 224$ canvas. White padding adds no texture that the network could exploit and matches the pale backgrounds typical of specimen photography. Pixel values were initially scaled to the range $[0, 1]$ during image loading. Each model then received inputs in the format required by its architecture. MobileNetV2 and ResNet50V2 used pixel values scaled to $[-1, 1]$, whereas EfficientNetB0 received values in the $[0, 255]$ range and performed normalization internally.

The data were split 80:20 into training (1,416 images) and validation (347 images) sets using the `validation_split` option of the Keras `ImageDataGenerator`. This option operates independently within each class directory, assigning the first 20% of files in sorted filename order (i.e., by GBIF occurrence key) to the validation set and the remainder to the training set. Class proportions were therefore preserved, which is equivalent to a stratified split, and the split is fully reproducible without a random seed. A fixed seed (42) was used to shuffle the training stream. Shuffling was disabled for the validation generator so that per-image predictions remained aligned with the ground-truth order, which is necessary for the confusion matrix in Section 4.3 to be valid. Augmentation was applied to the training data only: rotations within $\pm 20^\circ$, width and height shifts of 10%, zoom of 10%, brightness jitter in $[0.9, 1.1]$, and horizontal flips. These settings were chosen to introduce moderate variation while retaining the visual characteristics needed for species identification, because excessive augmentation may remove or distort fine image details that help distinguish closely related species (Shorten & Khoshgoftaar, 2019).

### Architectures and the controlled two-phase protocol

This study compares three CNN architectures: MobileNetV2, EfficientNetB0, and ResNet50V2, with 2.26 million, 4.05 million, and 23.56 million backbone parameters, respectively. These three backbones were chosen because they embody distinct design philosophies, depthwise separable convolution, compound scaling with channel attention, and pre-activation residual learning. All three are also potential candidates for mobile deployment after model conversion. Each backbone was initialized with ImageNet weights (Deng et al., 2009) and given the identical head, consisting of global average pooling, dropout at 0.5, a 256-unit ReLU layer, and a 17-class softmax layer. With the head, optimizer, loss, batch size, augmentation, split, and seed all held fixed, the backbone was the only manipulated variable.

Training followed the standard two-phase procedure (Table 2). Phase 1 froze the backbone and trained the head up to 30 epochs using the Adam optimizer at a learning rate of $10^{-4}$. Phase 2 made the entire backbone trainable and continued training at $10^{-5}$ for up to 50 epochs. Both phases used early stopping on validation loss with restoration of the best weights, together with learning-rate reduction on plateau. In phase 2, all backbone layers were unfrozen without special handling of any layer type. This setting was used to examine how the three architectures respond to a commonly applied fine-tuning procedure, not to recommend it as the best practice. The aim was to compare their behavior under the same training conditions and identify differences in performance after fine-tuning.

**Table 2. Training configuration**

| Setting | Phase 1: feature extraction | Phase 2: fine-tuning |
| :--- | :--- | :--- |
| Backbone weights | ImageNet, frozen | Unfrozen (all layers) |
| Initial learning rate (Adam) | $1 \times 10^{-4}$ | $1 \times 10^{-5}$ |
| Maximum epochs | 30 | 50 |
| Early stopping (val. loss) | patience 8, restore best | patience 10, restore best |
| ReduceLROnPlateau | factor 0.2, patience 4 | factor 0.2, patience 5 |
| Batch size / input | $16 \text{ / } 224 \times 224 \times 3$ | $16 \text{ / } 224 \times 224 \times 3$ |
| Epochs actually run | 30 / 30 / 30 | 16 / 11 / 34 (MNetV2 / EffB0 / RNet50V2) |

### Evaluation

Because accuracy estimated from 347 validation images is subject to considerable sampling uncertainty, every accuracy is reported with a 95% Wilson score interval (Wilson, 1927). Given the 11.5:1 class imbalance, we also report per-class precision, recall, and F1 score, together with their macro and weighted averages.

For each class $c$, precision, recall, and F1 were computed from the numbers of true positives ($TP_c$), false positives ($FP_c$), and false negatives ($FN_c$) in the confusion matrix (Equations 1 and 2) and averaged over the $K = 17$ classes either uniformly or weighted by the class support $n_c$, with $N = 347$ (Equation 3). For the multiclass setting, the Matthews correlation coefficient (MCC) was computed with the $K$-category generalization of Gorodkin (2004) (Equation 4) and Cohen’s kappa (Cohen, 1960) was computed with Equation 5, where $c$ is the number of correctly classified images, $s$ is the total number of images, $p_k$ is the number of times class $k$ was predicted, and $t_k$ is the number of times it truly occurred. The 95% Wilson score interval for an observed accuracy $\hat{p}$ on $n$ images is given by Equation 6, with $z = 1.96$.

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c} \tag{1}$$

$$\text{F1}_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c} \tag{2}$$

$$\text{Macro-F1} = \frac{1}{K} \sum_{c=1}^{K} \text{F1}_c, \quad \text{Weighted-F1} = \sum_{c=1}^{K} \frac{n_c}{N} \text{F1}_c \tag{3}$$

$$\text{MCC} = \frac{c \cdot s - \sum_k p_k t_k}{\sqrt{(s^2 - \sum_k p_k^2)(s^2 - \sum_k t_k^2)}} \tag{4}$$

$$\kappa = \frac{p_o - p_e}{1 - p_e}, \quad p_o = \frac{c}{s}, \quad p_e = \frac{\sum_k p_k t_k}{s^2} \tag{5}$$

$$\text{CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}} \tag{6}$$

In addition, the experiment includes an analysis that has not been frequently used in this literature. Using the order and family memberships in Table 1, each misclassification was categorized according to the taxonomic relationship between the true and predicted species: same family, same order but different family, or different order. If a classifier relied on backgrounds or other shortcuts, its errors would be scattered without regard to biology. To judge whether this pattern exceeds what would occur by chance, we compared it with a null model, that is, a reference scenario in which errors ignore taxonomy: whenever an image is misclassified, the wrong label is equally likely to be any of the 16 incorrect classes. For example, Hemiptera is represented by seven classes, so a misclassified hemipteran image has six same-order alternatives among its 16 incorrect classes, and $6/16 = 37.5\%$ of its errors would be expected to remain within Hemiptera by chance. A within-order rate well above this value indicates that the model relies on morphology, because taxonomic relatedness broadly tracks morphological similarity. The analysis requires nothing beyond the confusion matrix and requires no gradient-based attribution tools.

Formally, the expected within-order fraction under this null model is given by Equation 7, where $e_c$ is the number of misclassified validation images of class $c$ and $m_c$ is the number of classes in the order of class $c$. In the present data every class with at least one error belongs to Hemiptera or Lepidoptera, each represented by seven classes, so Equation 7 reduces to $6/16 = 37.5\%$.

$$E_{\text{within}} = \frac{\sum_c e_c \cdot \frac{m_c - 1}{K - 1}}{\sum_c e_c} \tag{7}$$

---

## Results

### Overall comparison

Figure 2 shows the full training trajectories, and Table 3 summarizes them. With frozen backbones (phase 1), EfficientNetB0 performed best, reaching 92.80% validation accuracy (322/347) with a validation loss of 0.2876 and the smallest generalization gap of the three models, 1.63 percentage points at the final epoch. MobileNetV2 and ResNet50V2 performed similarly, at 86.17% and 86.46%, respectively. None of the three models triggered early stopping in phase 1. All reached their minimum validation loss at exactly epoch 30, so the phase-1 results reported here are likely conservative estimates of the frozen-feature ceiling.

The two leading configurations, frozen-feature EfficientNetB0 and fine-tuned ResNet50V2, differ by 2.60 percentage points (9 images), and their confidence intervals overlap. Therefore, although EfficientNetB0 with frozen features was the leading model, but the available results do not establish that it is significantly better than ResNet50V2. Its margin over MobileNetV2 is larger (6.63 percentage points, 23 images), and the two confidence intervals do not overlap. Nevertheless, a paired statistical test on the per-image predictions, such as McNemar’s test, is still needed to determine whether this difference is significant.

*(Figure 2. Accuracy (top) and cross-entropy loss (bottom) over both training phases and the dashed line marks the start of fine-tuning at epoch 30)*

**Table 3. Summary of results**

| Model (phase) | Train acc. | Final val. acc. | Gap | Best val. acc. | 95% CI | Min. val. loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| MobileNetV2 (1) | 90.11% | 86.17% | +3.94 pp | 86.17% (299/347) | 82.14–89.41% | 0.3790 |
| MobileNetV2 (2) | 88.06% | 85.88% | +2.18 pp | 86.17% (299/347) | 82.14–89.41% | 0.3989 |
| EfficientNetB0 (1) | 94.42% | 92.80% | +1.62 pp | 92.80% (322/347) | 89.58–95.07% | 0.2876 |
| EfficientNetB0 (2) | 84.75% | 84.73% | +0.02 pp | 89.63% (311/347) | 85.97–92.41% | 0.3667 |
| ResNet50V2 (1) | 89.90% | 86.46% | +3.44 pp | 86.46% (300/347) | 82.45–89.66% | 0.4104 |
| ResNet50V2 (2) | 98.80% | 90.20% | +8.60 pp | 90.20% (313/347) | 86.62–92.90% | 0.3866 |

Differences in architecture may help explain this pattern. EfficientNetB0 combines MBConv blocks with squeeze-and-excitation modules that compute channel weights from global feature statistics (Hu et al., 2018; Tan & Le, 2019); changes in feature distributions during fine-tuning may therefore propagate through the whole network. In contrast, ResNet50V2 uses identity shortcuts that carry the signal directly across each residual block (He et al., 2016), which may help preserve information flow when the residual branches change during training. These architectural differences offer plausible explanations for the observed results, but controlled experiments are needed to determine which factors account for the contrasting responses to fine-tuning.

### Effects of fine-tuning

Phase 2 produced three qualitatively different outcomes from the same training protocol (Figure 3). ResNet50V2 behaved as conventionally expected: validation accuracy climbed from 86.46% to 90.20% and validation loss fell to 0.3866, with early stopping triggered after 34 phase-2 epochs. The improvement came at a cost visible in Table 3: training accuracy reached 98.80%, widening the generalization gap to 8.60 percentage points, which reflects ordinary overfitting pressure that the callbacks contained.

EfficientNetB0 degraded markedly. Validation loss jumped from 0.3667 at the first fine-tuning epoch to 2.1738 by the fifth, which is a 492.8% increase in four epochs, and remained near 1.6 even after plateau-triggered learning-rate reductions to $2 \times 10^{-6}$. The best validation accuracy in phase 2 was 89.63%, and validation accuracy at the final epoch was 84.73%. The decisive observation is that training accuracy also fell, from 94.42% to 84.75%, leaving a gap of only 0.02 percentage points. Overfitting typically widens the train–validation gap. However, the gap narrowed because both accuracies declined. In other words, the model was not memorizing the training set, as it appeared to be losing the representation it had acquired before fine-tuning. One possible explanation is that the normalization statistics learned during pretraining were replaced by statistics estimated from small mini-batches of the target data, but this mechanism was not tested directly in the present study.

*(Figure 3. Effect of fine-tuning)*

MobileNetV2 was largely unaffected by fine-tuning. Its best validation accuracy remained identical to that in phase 1 (86.17%), while its minimum validation loss increased slightly (from 0.3790 to 0.3989). With 2.26 million backbone parameters, it appears to have extracted most of what its capacity allows during phase 1. This result suggests that fine-tuning provided little additional benefit under the present training conditions. Its smaller parameter count may also leave less to destabilize.

### Per-class behavior of the exported model

The per-class evaluation was performed on the EfficientNetB0 checkpoint restored by early stopping after fine-tuning (89.63%), which is the model that was exported, rather than on the 92.80% phase-1 model. The deployed model therefore lost 3.17 percentage points (11 images) relative to the best phase-1 model, and this loss occurred without any warning during training.

Macro-F1 (0.8835) was 0.0116 lower than weighted-F1 (0.8951), reflecting the class imbalance, and the chance-corrected coefficients (MCC 0.8884; Cohen’s kappa 0.8881) fall within the range conventionally interpreted as almost perfect agreement (Chicco & Jurman, 2020). The per-class results (Figure 5, Table 4) reveal substantial variation between species. *Stenchaetothrips biformis* was classified perfectly across 28 validation images. Its distinct appearance as the only thrips species in the dataset may have helped the model distinguish it from the other classes. By comparison, the perfect scores of *Dicladispa armigera* and *Orseolia oryzae*, with two and four validation images respectively, These small sample sizes limit the conclusions that can be drawn from their results. At the other end, *Chilo suppressalis* achieved a recall of only 0.5455. The model failed to identify approximately 45% of the validation images belonging of this species. This result is particularly relevant because the species is an economically important stem borer. Two other classes, *Sesamia inferens* and *Pseudococcus saccharicola*, achieved perfect recall but had lower precision values of 0.7647 and 0.7692, respectively. In other words, the model identified all validation images belonging to these classes but also incorrectly assigned images from other classes to them. Such errors may reflect visual similarities between species, but further analysis would be needed to determine their underlying causes.

**Table 4. Per-class metrics of the exported model**

| Species | Precision | Recall | F1 | n |
| :--- | :---: | :---: | :---: | :---: |
| *Tetroda denticulifera* | 0.7500 | 0.6667 | 0.7059 | 9 |
| *Spodoptera mauritia* | 0.9355 | 0.9667 | 0.9508 | 30 |
| *Mythimna separata* | 0.8148 | 0.7586 | 0.7857 | 29 |
| *Scirpophaga incertulas* | 0.9231 | 0.8276 | 0.8727 | 29 |
| *Sesamia inferens* | 0.7647 | 1.0000 | 0.8667 | 13 |
| *Chilo suppressalis* | 0.7500 | 0.5455 | 0.6316 | 11 |
| *Nymphula depunctalis* | 0.9677 | 1.0000 | 0.9836 | 30 |
| *Cnaphalocrocis medinalis* | 0.9655 | 0.9655 | 0.9655 | 29 |
| *Sogatella furcifera* | 0.8387 | 0.8667 | 0.8525 | 30 |
| *Nilaparvata lugens* | 1.0000 | 0.9655 | 0.9825 | 29 |
| *Nephotettix virescens* | 0.9231 | 0.8000 | 0.8571 | 15 |
| *Pseudococcus saccharicola* | 0.7692 | 1.0000 | 0.8696 | 10 |
| *Stenchaetothrips biformis* | 1.0000 | 1.0000 | 1.0000 | 28 |
| *Dicladispa armigera* | 1.0000 | 1.0000 | 1.0000 | 2 |
| *Orseolia oryzae* | 1.0000 | 1.0000 | 1.0000 | 4 |
| *Leptocorisa oratoria* | 0.9310 | 0.9000 | 0.9153 | 30 |
| *Scotinophara coarctata* | 0.7273 | 0.8421 | 0.7805 | 19 |
| **Macro average** | **0.8859** | **0.8885** | **0.8835** | **347** |
| **Weighted average** | **0.8988** | **0.8963** | **0.8951** | **347** |

*(Figure 4. Confusion matrix of the exported EfficientNetB0 model on the 347-image validation set)*

*(Figure 5. Per-class F1 of the exported model, sorted, with validation support in parentheses)*

### Taxonomic patterns in classification errors

The 36 misclassifications showed a clear pattern rather than being distributed uniformly across the incorrect classes (Figure 6). Ten errors (27.8%) involved species from the same family as the true species, such as confusion between the brown planthopper and the white-backed planthopper, or between two noctuid armyworms. A further 18 errors (50.0%) A further 18 errors (50.0%) involved species from the same order but a different family. Overall, 28 of the 36 errors (77.8%) occurred within the same order, compared with the expected rate of 37.5% under the uniform-error null model described in Section 3.4. An exact binomial test indicated that this difference was statistically significant ($p = 9.9 \times 10^{-7}$). Because taxonomic proximity broadly reflects morphological resemblance, errors that respect taxonomy are what would be expected from a model relying on body shape, wing venation, and coloration, but not from exploiting backgrounds. Hansen et al. (2020) observed the same coupling between taxonomy and classification difficulty in museum beetles. In the present study, taxonomic error analysis is used as an additional diagnostic measure for examining the consistency of the model’s predictions with known biological relationships.

*(Figure 6. Taxonomic structure of the 36 misclassifications)*

The eight cross-order errors deserve separate scrutiny, and one cluster is particularly informative. One notable pattern involved *Mythimna separata*, a caterpillar, was twice predicted as *Sogatella furcifera*, a winged planthopper, and this class alone accounted for seven errors (recall 0.7586). These errors may reflect unusual visual similarities, image-quality issues, or incorrect labels in the source records. Their causes cannot be determined from the confusion matrix alone.

**Table 5. Context from the rice pest literature (n.r. = not reported)**

| Study | Data | Classes / images | Best model | Acc. |
| :--- | :--- | :---: | :--- | :---: |
| Wu et al. (2019) | IP102 | 102 / 75,222 | ResNet50, fine-tuned | 49.40% |
| Li et al. (2022) | IP102 rice subset | 14 / n.r. | ResNet50 + TL | 65.35% |
| Li et al. (2022) | IP_RicePests | 14 / 8,248 | VGG16 + TL | 84.39% |
| Li et al. (2022) | IP_RicePests + ARGAN | 14 / ~14,000 | VGG16 + TL | 88.68% |
| This work | GBIF, taxon-verified | 17 / 1,763 | EfficientNetB0, frozen | 92.80% |

### Comparison with prior work and deployment

Table 5 points to the potential importance of data quality. The frozen-feature EfficientNetB0 model achieved 92.80% validation accuracy across 17 classes using substantially fewer images than the dataset used by Li et al. (2022), whose dataset was expanded through GAN-based augmentation. This difference may reflect factors beyond the choice of CNN architecture, including taxonomic verification of image labels, the predominance of single-subject images, and aspect-ratio-preserving resizing. However, the present the results do not establish the individual contribution of these factors.

Three limitations should be considered when interpreting this comparison. First, the reported accuracy was measured on a validation set that was also used for early stopping, which may lead to an optimistic estimate of model performance. Second, the studies used different class inventories, making direct comparisons difficult. Third, images obtained through GBIF may be cleaner and less complex than the field images in datasets such as IP102. Although this may make model training easier, differences between the training images and real-world field conditions could reduce performance after deployment.

Regarding deployment, the exported model converts to a 4.2 MB TensorFlow Lite file under dynamic-range quantization, which is small enough for fully offline use on ordinary smartphones. However, the accuracy of the quantized model and its on-device inference latency were not evaluated in this study.

### Per-class behavior of the exported model

The per-class evaluation was performed on the EfficientNetB0 checkpoint restored by early stopping after fine-tuning (89.63%), which is the model that was exported, rather than on the 92.80% phase-1 model. The deployed model therefore lost 3.17 percentage points (11 images) relative to the best phase-1 model, and this loss occurred without any warning during training.

The per-class evaluation was performed on the EfficientNetB0 checkpoint restored by early stopping after fine-tuning (89.63%), which is the model that was exported, rather than on the 92.80%

---

## Discussion

### Overall comparison

The comparison was designed to rank three CNN architectures for rice pest classification, but the results also highlight the importance of the training procedure used in such comparisons. Had only the fine-tuning phase been considered, ResNet50V2 would have appeared to be the best-performing model, with a validation accuracy of 90.20%, followed by EfficientNetB0 (89.63%). The frozen-feature phase reverses this ordering, with EfficientNetB0 reaching 92.80%, 2.60 percentage points higher. In other words, the ranking of architectures depends on the training procedure, and a comparison that does not describe that procedure in sufficient detail cannot be verified. We therefore recommend that comparative studies describe their fine-tuning settings in detail, including which layers were unfrozen and how they were updated, and report the results of both phases rather than only the better one.

The possibility that fine-tuning can degrade a pretrained model is not new and has been noted in framework documentation (Chollet, 2020). Nevertheless, a quantified demonstration on an applied dataset is valuable precisely because the degradation is silent and superficially resembles overfitting. In this experiment, however, the generalization gap narrowed to 0.02 percentage points because both training and validation accuracy declined, a pattern that distinguishes this degradation from conventional overfitting.

The next finding concerns the role of data quality. With its backbone frozen and only the classification head trained, EfficientNetB0 achieved 92.80% validation accuracy across 17 species using 1,416 training images. This result is consistent with previous work showing that pretrained features can transfer effectively to new image classification tasks (Kornblith et al., 2019). One potential advantage of the present dataset is that scientific names were checked against a taxonomic reference before image collection, which helped reduce ambiguity in species identification. In this comparison, a model trained on a small, carefully labeled dataset achieved higher accuracy than those reported for larger datasets, although the datasets are not directly comparable. Moreover, the experiment did not isolate the effect of taxonomic verification from other factors, such as image characteristics, class composition, and model architecture. The results therefore support the value of careful dataset construction but do not establish that label quality alone explains the model's performance.

The taxonomic error analysis provided additional insight into the model’s predictions, with 77.8% of misclassifications occurring between species in the same order. It also highlighted cases that may warrant label verification, including *Mythimna separata*. These findings suggest a potential direction for deployment: a pest advisory system could present the top three species candidates alongside family-level information, rather than relying on a single species prediction. Cases with low confidence could be referred to an expert for further assessment. This analysis is relatively simple to perform, requiring only a confusion matrix and a taxonomic reference table, and could serve as a practical tool for examining classification errors and identifying potentially problematic labels.

The number of validation images was limited for some species. Two classes had fewer than five validation images, making their perfect classification scores difficult to interpret reliably. This limitation highlights the need for more representative samples, particularly for underrepresented species. A similar limitation has been reported in other insect classification tasks, including the Thai mosquito classifier of Kittichai et al. (2021), in which species represented by few wild-caught specimens were among the most difficult to classify.

For future mobile deployment, the distinction between curated images and field photographs is also important, because photographs taken in rice fields may differ in background complexity, lighting, insect orientation, and image quality. Thai experience with a deployed rice diagnostic system gives a sense of the likely size of this effect: the LINE-based disease detector of Temniranrat et al. (2021) achieved 95.6% accuracy on its curated test set but 78.86% on photographs submitted by its users, a drop of roughly 17 percentage points.

---

## Conclusions

The possibility that fine-tuning can degrade a pretrained model is not new and has been noted in framework documentation (Chollet, 2020). State conclusion (do not This study evaluated three CNN architectures for classifying 17 rice insect pest species using a taxonomically verified dataset of 1,763 images. With its backbone frozen, EfficientNetB0 achieved the highest validation accuracy (92.80%). The exported fine-tuned EfficientNetB0 model (89.63%) achieved a macro-F1 score of 0.8835 and an MCC of 0.8884, and its TensorFlow Lite version had a file size of 4.2 MB, making it a potential candidate for offline mobile applications. However, fine-tuning reduced EfficientNetB0's validation accuracy by 3.17 percentage points. The accompanying decline in both training and validation performance suggests a broader degradation rather than conventional overfitting, although its underlying cause requires further experimental confirmation. Under the same training protocol, fine-tuning improved ResNet50V2 by 3.75 percentage points, while MobileNetV2 showed no change in its best validation accuracy. The taxonomic analysis also revealed that misclassifications occurred within the same insect order more frequently than expected by chance (77.8% vs. 37.5%). These findings show that fine-tuning does not necessarily improve classification performance and that its effects may differ substantially across architectures. Combining conventional performance metrics with taxonomic error analysis may also provide a practical way to investigate classification errors and identify records that warrant further label verification. Before the model can be recommended for field use, its performance must be assessed on an independent test set and on images captured under real rice-field conditions.

---

## Declarations & Back Matter

### Funding
This project is funded by Valaya Aongkorn Rajabhat University under the Royal Patronage.

### Author Contributions
Conceptualization: N.A., C.K.; Methodology: N.A., C.K.; Investigation: N.A., C.K., D.W.; Data Curation: C.K., D.W.; Writing – Original Draft: N.A.; Writing – Review & Editing: N.A., D.W.; Supervision: C.K., D.W. All authors have read and agreed to the published version of the manuscript.

### Ethics Statement
Ethical approval was not required for this study because it did not involve human participants, animals, identifiable personal data, or materials requiring institutional ethical review.

### Declaration of Generative AI Use (12 pt Bold)
Generative AI was used solely for language editing, grammar improvement, and assistance with improving the clarity of the manuscript. The authors reviewed and edited all AI-assisted content and take full responsibility for the accuracy, integrity, and originality of the final manuscript.

### Conflict of Interest
The authors declare that they have no conflicts of interest related to this work.

### Data Availability Statement
No new data were generated or analyzed in support of this research.

### Acknowledgements
This work was supported by Valaya Aongkorn Rajabhat University under the Royal Patronage (VRU).

---

## References

- Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21(1), Article 6. https://doi.org/10.1186/s12864-019-6413-7
- Chollet, F. (2020). *Transfer learning & fine-tuning*. Keras documentation. Retrieved September 14, 2026, from https://keras.io/guides/transfer_learning/
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46. https://doi.org/10.1177/001316446002000104
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 248–255). IEEE. https://doi.org/10.1109/CVPR.2009.5206848
- Escalada, M. M., Luecha, M., & Heong, K. L. (2015). Social impacts of planthopper outbreaks in Thailand. In K. L. Heong, J. Cheng, & M. M. Escalada (Eds.), *Rice planthoppers: Ecology, management, socio economics and policy* (pp. 191–207). Springer. https://doi.org/10.1007/978-94-017-9535-7_10
- GBIF.org. (2025). *GBIF occurrence download*. Global Biodiversity Information Facility. https://www.gbif.org
- Gorodkin, J. (2004). Comparing two K-category assignments by a K-category correlation coefficient. *Computational Biology and Chemistry*, 28, 367–374. https://doi.org/10.1016/j.compbiolchem.2004.09.006
- Hansen, O. L. P., Svenning, J.-C., Olsen, K., Dupont, S., Garner, B. H., Iosifidis, A., Price, B. W., & Høye, T. T. (2020). Species-level image classification with convolutional neural network enables insect identification from habitus images. *Ecology and Evolution*, 10(2), 737–747. https://doi.org/10.1002/ece3.5921
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in deep residual networks. In *Computer Vision – ECCV 2016* (pp. 630–645). Springer. https://doi.org/10.1007/978-3-319-46493-0_38
- Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 7132–7141). IEEE. https://doi.org/10.1109/CVPR.2018.00745
- Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *Proceedings of the 32nd International Conference on Machine Learning* (pp. 448–456). PMLR. https://proceedings.mlr.press/v37/ioffe15.html
- Kittichai, V., Pengsakul, T., Chumchuen, K., Samung, Y., Sriwichai, P., Phatthamolrat, N., Tongloy, T., Jaksukam, K., Chuwongin, S., & Boonsang, S. (2021). Deep learning approaches for challenging species and gender identification of mosquito vectors. *Scientific Reports*, 11(1), Article 4838. https://doi.org/10.1038/s41598-021-84219-4
- Kornblith, S., Shlens, J., & Le, Q. V. (2019). Do better ImageNet models transfer better? In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 2661–2671). IEEE. https://doi.org/10.1109/CVPR.2019.00277
- Li, Z., Jiang, X., Jia, X., Duan, X., Wang, Y., & Mu, J. (2022). Classification method of significant rice pests based on deep learning. *Agronomy*, 12(9), Article 2096. https://doi.org/10.3390/agronomy12092096
- Liu, Z., Gao, J., Yang, G., Zhang, H., & He, Y. (2016). Localization and classification of paddy field pests using a saliency map and deep convolutional neural network. *Scientific Reports*, 6, Article 20410. https://doi.org/10.1038/srep20410
- Martineau, C., Conte, D., Raveaux, R., Arnault, I., Munier, D., & Venturini, G. (2017). A survey on image-based insect classification. *Pattern Recognition*, 65, 273–284. https://doi.org/10.1016/j.patcog.2016.12.020
- Office of Agricultural Economics. (2026). *Agricultural Statistics of Thailand 2025*. Ministry of Agriculture and Cooperatives: Bangkok, Thailand.
- Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE Transactions on Knowledge and Data Engineering*, 22(10), 1345–1359. https://doi.org/10.1109/TKDE.2009.191
- Rahman, C. R., Arko, P. S., Ali, M. E., Khan, M. A. I., Apon, S. H., Nowrin, F., & Wasif, A. (2020). Identification and recognition of rice diseases and pests using convolutional neural networks. *Biosystems Engineering*, 194, 112–120. https://doi.org/10.1016/j.biosystemseng.2020.03.020
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 4510–4520). IEEE. https://doi.org/10.1109/CVPR.2018.00474
- Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1), Article 60. https://doi.org/10.1186/s40537-019-0197-0
- Skawsang, S., Nagai, M., Tripathi, N. K., & Soni, P. (2019). Predicting rice pest population occurrence with satellite-derived crop phenology, ground meteorological observation, and machine learning: A case study for the Central Plain of Thailand. *Applied Sciences*, 9(22), Article 4846. https://doi.org/10.3390/app9224846
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *Proceedings of the 36th International Conference on Machine Learning* (pp. 6105–6114). PMLR. https://proceedings.mlr.press/v97/tan19a.html
- Temniranrat, P., Kiratiratanapruk, K., Kitvimonrat, A., Sinthupinyo, W., & Patarapuwadol, S. (2021). A system for automatic rice disease detection from rice paddy images serviced via a Chatbot. *Computers and Electronics in Agriculture*, 185, Article 106156. https://doi.org/10.1016/j.compag.2021.106156
- Venkatasaichandrakanth, P., & Iyapparaja, M. (2024). GNViT: An enhanced image-based groundnut pest classification using Vision Transformer (ViT) model. *PLOS ONE*, 19(3), Article e0301174. https://doi.org/10.1371/journal.pone.0301174
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953
- Wu, X., Zhan, C., Lai, Y.-K., Cheng, M.-M., & Yang, J. (2019). IP102: A large-scale benchmark dataset for insect pest recognition. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 8787–8796). IEEE. https://doi.org/10.1109/CVPR.2019.00899

