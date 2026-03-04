# Mathematical Foundations of Object Detection for Autonomous Driving

> A self-contained mathematical reference for every algorithm and metric
> implemented in this project.

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Bounding-Box Representation and Geometry](#2-bounding-box-representation-and-geometry)
3. [Intersection over Union (IoU)](#3-intersection-over-union-iou)
4. [Anchor-Based Region Proposals](#4-anchor-based-region-proposals)
5. [Bounding-Box Regression](#5-bounding-box-regression)
6. [Faster R-CNN](#6-faster-r-cnn)
7. [YOLOv8 — Anchor-Free Single-Stage Detection](#7-yolov8--anchor-free-single-stage-detection)
8. [Non-Maximum Suppression (NMS)](#8-non-maximum-suppression-nms)
9. [Loss Functions](#9-loss-functions)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Data Augmentation Mathematics](#11-data-augmentation-mathematics)
12. [Optimisation and Learning-Rate Scheduling](#12-optimisation-and-learning-rate-scheduling)
13. [Mixed-Precision Training](#13-mixed-precision-training)
14. [Feature Pyramid Networks (FPN)](#14-feature-pyramid-networks-fpn)
15. [References](#15-references)

---

## 1. Problem Formulation

Given an input image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$, the
object detection task requires predicting a set of detections:

$$
\mathcal{D} = \bigl[(b_i,\, c_i,\, s_i)\bigr]_{i=1}^{N}
$$

where, for each detection $i$:

| Symbol | Meaning |
|--------|---------|
| $b_i = (x_1^i, y_1^i, x_2^i, y_2^i) \in \mathbb{R}^4$ | Bounding box (top-left and bottom-right corners) |
| $c_i \in \{1, \dots, K\}$ | Class label ($K$ foreground classes) |
| $s_i \in [0, 1]$ | Confidence score |

The goal of training is to learn parameters $\theta$ that minimise a
multi-task loss combining **localisation** and **classification** terms:

$$
\theta^{*} = \arg\min_{\theta}\;
\frac{1}{N}\sum_{i=1}^{N}
\bigl[\mathcal{L}_{\text{cls}}(c_i, \hat{c}_i;\theta)
 + \lambda\,\mathcal{L}_{\text{loc}}(b_i, \hat{b}_i;\theta)\bigr]
$$

where $\lambda$ controls the relative weight of localisation versus
classification.

---

## 2. Bounding-Box Representation and Geometry

### 2.1 Corner form

A box is stored as $(x_1, y_1, x_2, y_2)$ with the constraint
$x_2 > x_1,\; y_2 > y_1$.

**Area:**

$$
\text{Area}(b) = (x_2 - x_1)(y_2 - y_1)
$$

### 2.2 Centre form

An equivalent parameterisation used inside regression heads:

$$
c_x = \frac{x_1 + x_2}{2},\quad
c_y = \frac{y_1 + y_2}{2},\quad
w = x_2 - x_1,\quad
h = y_2 - y_1
$$

---

## 3. Intersection over Union (IoU)

IoU measures the overlap between a predicted box $B_p$ and a
ground-truth box $B_g$:

$$
\text{IoU}(B_p, B_g)
= \frac{|B_p \cap B_g|}{|B_p \cup B_g|}
= \frac{|B_p \cap B_g|}{|B_p| + |B_g| - |B_p \cap B_g|}
$$

where $|\cdot|$ denotes area. The intersection is computed as:

$$
|B_p \cap B_g|
= \max\bigl(0,\;\min(x_2^p, x_2^g) - \max(x_1^p, x_1^g)\bigr)
\;\times\;
\max\bigl(0,\;\min(y_2^p, y_2^g) - \max(y_1^p, y_1^g)\bigr)
$$

**Properties:**

* $\text{IoU} \in [0, 1]$
* $\text{IoU} = 1$ iff $B_p = B_g$ (perfect overlap)
* $\text{IoU} = 0$ iff $B_p \cap B_g = \varnothing$ (no overlap)

### 3.1 Pairwise IoU Matrix

For two sets of boxes $\{B^a_i\}_{i=1}^N$ and $\{B^b_j\}_{j=1}^M$ we
construct the $N \times M$ matrix $\mathbf{J}$ where
$J_{ij} = \text{IoU}(B^a_i, B^b_j)$. This is used for matching
predictions to ground truths in evaluation.

---

## 4. Anchor-Based Region Proposals

In Faster R-CNN, the Region Proposal Network (RPN) generates proposals from
a set of **anchors** tiled over the feature map.

### 4.1 Anchor Generation

At each spatial location $(i, j)$ on a feature map of stride $s$, anchors
are centred at:

$$
c_x = s \cdot j + \frac{s}{2},\qquad
c_y = s \cdot i + \frac{s}{2}
$$

For each of $|\mathcal{S}|$ scales and $|\mathcal{R}|$ aspect ratios, the
anchor dimensions are:

$$
w_k = s \cdot s_k \cdot \sqrt{r_k},\qquad
h_k = s \cdot s_k \;/\; \sqrt{r_k}
$$

where $s_k \in \mathcal{S}$ and $r_k \in \mathcal{R}$.

### 4.2 Anchor Assignment

An anchor $a$ is labelled **positive** if:

$$
\text{IoU}(a, g^{*}) \geq \tau_{\text{pos}} \quad (\text{typically } 0.7)
$$

or it has the highest IoU with any ground-truth box $g^*$. It is
**negative** if:

$$
\max_{g}\;\text{IoU}(a, g) < \tau_{\text{neg}} \quad (\text{typically } 0.3)
$$

All other anchors are ignored during training.

---

## 5. Bounding-Box Regression

The network predicts **offsets** $(t_x, t_y, t_w, t_h)$ relative to an
anchor or proposal $(a_x, a_y, a_w, a_h)$:

$$
\hat{x} = a_x + t_x \cdot a_w,\qquad
\hat{y} = a_y + t_y \cdot a_h
$$

$$
\hat{w} = a_w \cdot \exp(t_w),\qquad
\hat{h} = a_h \cdot \exp(t_h)
$$

Ground-truth targets are the inverse:

$$
t_x^* = \frac{g_x - a_x}{a_w},\quad
t_y^* = \frac{g_y - a_y}{a_h},\quad
t_w^* = \ln\frac{g_w}{a_w},\quad
t_h^* = \ln\frac{g_h}{a_h}
$$

---

## 6. Faster R-CNN

Faster R-CNN is a **two-stage** detector comprising:

1. **Backbone + FPN** — extracts multi-scale feature maps.
2. **Region Proposal Network (RPN)** — proposes candidate boxes.
3. **RoI Head** — classifies and refines proposals.

### 6.1 RPN Losses

For $N_{\text{cls}}$ sampled anchors the RPN minimises:

$$
\mathcal{L}_{\text{RPN}}
= \frac{1}{N_{\text{cls}}}\sum_i \mathcal{L}_{\text{cls}}(p_i, p_i^{*})
+ \frac{\lambda}{N_{\text{reg}}}\sum_i p_i^{*}\;
\text{smooth}_{L_1}(t_i - t_i^{*})
$$

where $p_i$ is the predicted objectness probability, $p_i^{*} \in \{0, 1\}$
is the anchor label, and $t_i, t_i^{*}$ are predicted and target box
offsets respectively.

### 6.2 RoI Pooling / RoI Align

Given a proposal of arbitrary size on the feature map, RoI Align extracts a
fixed-size feature by bilinear interpolation at regularly sampled points,
avoiding the quantisation errors of RoI Pooling.

For a proposal mapped to feature coordinates $(x_1', y_1', x_2', y_2')$,
the output grid of size $G \times G$ samples at:

$$
x_{g} = x_1' + \frac{(g_x + 0.5)(x_2' - x_1')}{G},\quad
y_{g} = y_1' + \frac{(g_y + 0.5)(y_2' - y_1')}{G}
$$

### 6.3 Fast R-CNN Head

The classification head applies softmax over $K + 1$ classes (including
background):

$$
p(c \mid \mathbf{f}) = \text{softmax}\bigl(\mathbf{W}_{\text{cls}}\,\mathbf{f} + \mathbf{b}_{\text{cls}}\bigr)
$$

The regression head predicts class-specific box offsets
$t^c \in \mathbb{R}^4$ for each foreground class $c$.

### 6.4 Multi-Task Loss (Fast R-CNN Head)

$$
\mathcal{L}_{\text{head}}
= \mathcal{L}_{\text{cls}}(p, c^{*})
+ \lambda\,[c^{*} \geq 1]\;\text{smooth}_{L_1}(t^{c^{*}}, t^{*})
$$

where $[c^{*} \geq 1]$ is an Iverson bracket that ignores background
proposals in the regression term.

---

## 7. YOLOv8 — Anchor-Free Single-Stage Detection

YOLOv8 is an **anchor-free** detector that directly predicts box centres
and dimensions at each feature-map cell.

### 7.1 Grid Decoding

At feature level $\ell$ with stride $s_\ell$, the model outputs per-cell
offsets $(\sigma(o_x), \sigma(o_y))$ and distances $(l, t, r, b)$ to the
four sides of the bounding box:

$$
\hat{x} = (g_x + \sigma(o_x)) \cdot s_\ell,\qquad
\hat{y} = (g_y + \sigma(o_y)) \cdot s_\ell
$$

$$
x_1 = \hat{x} - l \cdot s_\ell,\quad
y_1 = \hat{y} - t \cdot s_\ell,\quad
x_2 = \hat{x} + r \cdot s_\ell,\quad
y_2 = \hat{y} + b \cdot s_\ell
$$

where $\sigma$ is the sigmoid function and $(g_x, g_y)$ is the grid cell
index.

### 7.2 Task-Aligned Assignment (TAL)

YOLOv8 uses **Task-Aligned Assignment** to match predictions to
ground truths. For a candidate prediction with classification score $s$ and
IoU $u$ with a ground-truth, the alignment metric is:

$$
t = s^{\alpha} \cdot u^{\beta}
$$

Candidates with the highest $t$ values are assigned as positives.

### 7.3 Distribution Focal Loss (DFL)

Rather than regressing a single scalar, YOLOv8 represents each box side
distance as a discrete probability distribution over $n$ bins
$\{0, 1, \dots, n-1\}$. The predicted distance is the expectation:

$$
\hat{d} = \sum_{i=0}^{n-1} i \cdot P(i)
$$

The Distribution Focal Loss encourages the distribution to concentrate
around the target value $y$:

$$
\mathcal{L}_{\text{DFL}}(P, y)
= -\bigl[(y_{\lceil\rceil} - y)\log P(y_{\lfloor\rfloor})
+ (y - y_{\lfloor\rfloor})\log P(y_{\lceil\rceil})\bigr]
$$

where $y_{\lfloor\rfloor} = \lfloor y \rfloor$ and
$y_{\lceil\rceil} = \lceil y \rceil$.

---

## 8. Non-Maximum Suppression (NMS)

After decoding, many overlapping boxes may predict the same object. NMS
keeps only the most confident detection per object cluster.

**Algorithm:**

1. Sort detections by score in descending order.
2. Select the highest-scoring box $b_1$ and add it to the output set
   $\mathcal{K}$.
3. Remove all remaining boxes $b_j$ for which
   $\text{IoU}(b_1, b_j) > \tau_{\text{NMS}}$.
4. Repeat from step 2 until no boxes remain.

The standard threshold is $\tau_{\text{NMS}} = 0.5$.

---

## 9. Loss Functions

### 9.1 Cross-Entropy Loss (Classification)

$$
\mathcal{L}_{\text{CE}}(p, y)
= -\sum_{c=0}^{K} y_c \log p_c
$$

where $y$ is the one-hot ground-truth vector and $p$ is the predicted
probability distribution.

### 9.2 Binary Cross-Entropy with Logits

For objectness or per-class binary predictions:

$$
\mathcal{L}_{\text{BCE}}(z, y)
= -\bigl[y \log\sigma(z) + (1 - y)\log(1 - \sigma(z))\bigr]
$$

### 9.3 Focal Loss

Addresses class imbalance by down-weighting easy examples:

$$
\mathcal{L}_{\text{FL}}(p_t)
= -\alpha_t (1 - p_t)^{\gamma} \log(p_t)
$$

where $p_t = p$ if $y = 1$ else $1 - p$, $\alpha_t$ is a balancing
factor, and $\gamma \geq 0$ is the focusing parameter (typically 2.0).

### 9.4 Smooth L1 Loss (Localisation)

$$
\text{smooth}_{L_1}(x) =
\begin{cases}
0.5\,x^2 & \text{if } |x| < 1 \\
|x| - 0.5 & \text{otherwise}
\end{cases}
$$

This is less sensitive to outliers than the standard $L_2$ loss and more
stable than $L_1$ near zero.

### 9.5 CIoU Loss (YOLOv8 Localisation)

Complete IoU loss adds a distance penalty and an aspect-ratio consistency
term to the standard IoU loss:

$$
\mathcal{L}_{\text{CIoU}}
= 1 - \text{IoU}
+ \frac{\rho^2(\mathbf{b}_p, \mathbf{b}_g)}{c^2}
+ \alpha v
$$

where:

* $\rho(\mathbf{b}_p, \mathbf{b}_g)$ is the Euclidean distance between box
  centres.
* $c$ is the diagonal of the smallest enclosing box.
* $v = \frac{4}{\pi^2}\bigl(\arctan\frac{w_g}{h_g} - \arctan\frac{w_p}{h_p}\bigr)^2$
  measures aspect-ratio consistency.
* $\alpha = \frac{v}{(1 - \text{IoU}) + v}$ is a balancing weight.

---

## 10. Evaluation Metrics

### 10.1 Precision and Recall

For a given confidence threshold and IoU threshold $\tau$:

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}},\qquad
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

A predicted box is a **True Positive (TP)** if $\text{IoU}(B_p, B_g)
\geq \tau$ and it is the highest-scoring match for that ground-truth box.
Otherwise it is a **False Positive (FP)**.

### 10.2 Precision-Recall Curve

By sweeping the confidence threshold from high to low, we obtain ordered
pairs $(\text{recall}_k,\, \text{precision}_k)$ that trace out the PR
curve.

### 10.3 Average Precision (AP)

We use the **all-point interpolation** method. First, the precision is made
monotonically non-increasing:

$$
p_{\text{interp}}(r) = \max_{r' \geq r} p(r')
$$

Then AP is the area under this interpolated curve:

$$
\text{AP} = \sum_{k=0}^{n-1} (r_{k+1} - r_k)\;p_{\text{interp}}(r_{k+1})
$$

### 10.4 Mean Average Precision (mAP)

mAP averages AP over all $K$ foreground classes:

$$
\text{mAP} = \frac{1}{K}\sum_{c=1}^{K} \text{AP}_c
$$

We report **mAP@0.5** (IoU threshold 0.5) as the primary metric.

### 10.5 F1 Score

The harmonic mean of precision and recall:

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

---

## 11. Data Augmentation Mathematics

### 11.1 Horizontal Flip

For an image of width $W$, the horizontal flip maps pixel
$(x, y) \mapsto (W - 1 - x,\; y)$.

Bounding-box coordinates transform as:

$$
x_1' = W - x_2,\qquad x_2' = W - x_1,\qquad y_1' = y_1,\qquad y_2' = y_2
$$

### 11.2 Resize with Aspect-Ratio Preservation

Given a target minimum dimension $d_{\min}$ and optional maximum dimension
$d_{\max}$, the scale factor is:

$$
s = \frac{d_{\min}}{\min(H, W)}
$$

If $d_{\max}$ is specified:

$$
s = \min\!\left(s,\;\frac{d_{\max}}{\max(H, W)}\right)
$$

New dimensions: $W' = \lfloor W \cdot s + 0.5 \rfloor$,
$H' = \lfloor H \cdot s + 0.5 \rfloor$, and all box coordinates scale
by $s$.

### 11.3 Colour Jitter

Each adjustment samples a random factor $f$ from a uniform distribution:

| Transform | Sampling | Operation |
|-----------|----------|-----------|
| Brightness | $f \sim U(1 - \beta,\; 1 + \beta)$ | $\mathbf{I}' = f \cdot \mathbf{I}$ |
| Contrast | $f \sim U(1 - \gamma,\; 1 + \gamma)$ | $\mathbf{I}' = f \cdot \mathbf{I} + (1 - f) \cdot \bar{I}$ |
| Saturation | $f \sim U(1 - \sigma,\; 1 + \sigma)$ | Blend towards greyscale |
| Hue | $h \sim U(-\delta,\; \delta)$ | Rotate HSV hue channel by $h$ |

The order of application is randomised to increase diversity.

### 11.4 Target Sanitisation

Degenerate boxes (zero or negative area) are removed before training:

$$
\text{keep}_i = \bigl[(x_2^i - x_1^i) > 0\bigr] \;\wedge\; \bigl[(y_2^i - y_1^i) > 0\bigr]
$$

### 11.5 ImageNet Normalisation

After conversion to $[0, 1]$ float tensors, optional channel-wise
normalisation is applied:

$$
\hat{I}_c = \frac{I_c - \mu_c}{\sigma_c}
$$

with ImageNet statistics $\boldsymbol{\mu} = (0.485, 0.456, 0.406)$ and
$\boldsymbol{\sigma} = (0.229, 0.224, 0.225)$.

---

## 12. Optimisation and Learning-Rate Scheduling

### 12.1 Stochastic Gradient Descent with Momentum

Parameter update at iteration $t$:

$$
\mathbf{v}_t = \mu\,\mathbf{v}_{t-1} + \nabla_\theta \mathcal{L}(\theta_{t-1})
$$

$$
\theta_t = \theta_{t-1} - \eta\,\mathbf{v}_t - \eta\,\lambda\,\theta_{t-1}
$$

where $\eta$ is the learning rate, $\mu$ is the momentum coefficient, and
$\lambda$ is the weight-decay (L2 regularisation) coefficient.

### 12.2 Step Learning-Rate Decay

The learning rate is reduced by a factor $\gamma$ every $T$ epochs:

$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t / T \rfloor}
$$

### 12.3 Cosine Annealing

The learning rate follows a cosine schedule over $T_{\max}$ epochs:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})
\Bigl(1 + \cos\!\bigl(\frac{\pi\,t}{T_{\max}}\bigr)\Bigr)
$$

This provides a smooth decay with warm restarts if combined with periodic
resets.

### 12.4 Gradient Clipping

To prevent exploding gradients, the parameter-gradient vector is rescaled
when its L2 norm exceeds a threshold $g_{\max}$:

$$
\hat{\mathbf{g}} =
\begin{cases}
\mathbf{g} & \text{if } \|\mathbf{g}\|_2 \leq g_{\max} \\[4pt]
g_{\max} \cdot \dfrac{\mathbf{g}}{\|\mathbf{g}\|_2} & \text{otherwise}
\end{cases}
$$

### 12.5 Early Stopping

Training is halted if the monitored metric (validation loss) does not
improve for $P$ consecutive epochs (the *patience*). This regularises the
model by preventing overfitting.

---

## 13. Mixed-Precision Training

Automatic Mixed Precision (AMP) uses **float16** for the forward pass and
loss computation while keeping a **float32** master copy of the weights for
gradient accumulation.

### 13.1 Loss Scaling

Because float16 has limited dynamic range ($\approx 6 \times 10^{-8}$ to
$6.5 \times 10^4$), gradients are scaled before the backward pass:

$$
\hat{\mathcal{L}} = S \cdot \mathcal{L}
$$

After `.backward()`, the optimizer unscales the gradients:

$$
\hat{\nabla}_\theta = \frac{1}{S}\;\nabla_\theta \hat{\mathcal{L}}
$$

The scale factor $S$ is dynamically adjusted: increased when no overflow is
detected and halved upon overflow.

---

## 14. Feature Pyramid Networks (FPN)

The Faster R-CNN backbone in this project uses a ResNet-50 with FPN. FPN
builds a multi-scale feature pyramid by combining bottom-up and top-down
pathways.

### 14.1 Bottom-Up Pathway

ResNet produces feature maps $\{C_2, C_3, C_4, C_5\}$ at strides
$\{4, 8, 16, 32\}$ pixels.

### 14.2 Top-Down Pathway and Lateral Connections

Starting from the coarsest level:

$$
P_5 = \text{Conv}_{1\times1}(C_5)
$$

$$
P_l = \text{Conv}_{1\times1}(C_l) + \text{Upsample}_{2\times}(P_{l+1}),
\quad l \in \{2, 3, 4\}
$$

Each $P_l$ is then passed through a $3\times3$ convolution to reduce
aliasing. The result is a set of feature maps $\{P_2, P_3, P_4, P_5\}$ all
with 256 channels, enabling detection at multiple scales.

---

## 15. References

1. S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards
   Real-Time Object Detection with Region Proposal Networks," *NeurIPS*,
   2015.

2. T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and
   S. Belongie, "Feature Pyramid Networks for Object Detection," *CVPR*,
   2017.

3. G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," 2023.
   https://github.com/ultralytics/ultralytics

4. Z. Zheng, P. Wang, W. Liu, J. Li, R. Ye, and D. Ren, "Distance-IoU
   Loss: Faster and Better Learning for Bounding Box Regression," *AAAI*,
   2020.

5. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss
   for Dense Object Detection," *ICCV*, 2017.

6. A. Geiger, P. Lenz, and R. Urtasun, "Are we ready for Autonomous
   Driving? The KITTI Vision Benchmark Suite," *CVPR*, 2012.

7. F. Yu *et al.*, "BDD100K: A Diverse Driving Dataset for Heterogeneous
   Multitask Learning," *CVPR*, 2020.
