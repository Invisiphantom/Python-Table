

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source .bashrc

sudo chmod -R 777 /opt/
pip install numpy pandas matplotlib jupyter import-ipynb scikit-learn
pip install torch torchvision torchaudio torchmetrics tensorboard torch-tb-profiler
pip install transformers datasets tokenizers
```


这里的代码实现的是 **条件随机场（CRF）的对数似然函数的梯度计算**，其数学公式如下：

---

### **1. 梯度公式**
对于特征 \( f \) 的权重 \( \mathbf{w}_f \)，梯度的计算公式为：
\[
\nabla_{\mathbf{w}_f} \mathcal{L} = \underbrace{\sum_{t=1}^T \mathbb{I}[f \in \mathcal{F}(x, y_t, t)]}_{\text{真实特征出现次数}} - \underbrace{\mathbb{E}_{P(y'|x)}\left[\sum_{t=1}^T \mathbb{I}[f \in \mathcal{F}(x, y'_t, t)]\right]}_{\text{特征模型期望}}
\]
其中：
- \( \mathbb{I}[\cdot] \) 是指示函数（特征 \( f \) 是否在位置 \( t \) 被激活）。
- \( \mathcal{F}(x, y_t, t) \) 是位置 \( t \) 标签 \( y_t \) 激活的特征集合。
- \( y \) 是真实标签序列，\( y' \) 是所有可能的预测序列。

---

### **2. 代码与公式的对应关系**
#### **(1) 真实特征计数**
```python
for f in true_features:
    gradient[f] += 1  # 对应公式的第一项 ∑𝕀[f ∈ F(x, y_t, t)]
```
- **物理意义**：统计特征 \( f \) 在真实标签序列 \( y \) 中出现的总次数。

#### **(2) 模型特征期望**
```python
for f in expected_features:
    gradient[f] -= expected_features[f]  # 对应公式的第二项 𝔼[∑𝕀[f ∈ F(x, y'_t, t)]]
```
- **物理意义**：减去特征 \( f \) 在所有可能预测序列 \( y' \) 中的期望出现次数（通过前向-后向算法计算）。

---

### **3. 直观解释**
- **梯度为正**（`true_count > expected_count`）：  
  当前特征在真实路径中比模型预测的更活跃，应增加其权重。
- **梯度为负**（`true_count < expected_count`）：  
  当前特征在模型中过活跃，应降低其权重。
- **梯度为零**：模型预测与真实情况一致。

---

### **4. 具体例子**
假设：
- **特征**：\( f_1 = \text{"B-PER::U02:张"} \)（当标签为 `B-PER` 且当前词为“张”时激活）。
- **真实序列**：\( y = [\text{B-PER}, \text{I-PER}] \)，\( f_1 \) 在 \( t=0 \) 激活（计数=1）。
- **模型预测**：
  - \( P(y'_0=\text{B-PER}|x)=0.7 \)，\( P(y'_0=\text{O}|x)=0.3 \)。
  - \( f_1 \) 仅在 `B-PER` 激活，因此模型期望为 \( 0.7 \times 1 + 0.3 \times 0 = 0.7 \)。

**梯度计算**：
\[
\nabla_{\mathbf{w}_{f_1}} = 1 (\text{真实}) - 0.7 (\text{模型期望}) = 0.3
\]

---

### **5. 数学推导**
#### **(1) 对数似然函数**
\[
\mathcal{L}(\mathbf{w}) = \log P(y|x) = \sum_{t=1}^T \left( \mathbf{T}(y_{t-1}, y_t) + \sum_f \mathbf{w}_f f(x, y_t, t) \right) - \log Z(x)
\]

#### **(2) 对 \( \mathbf{w}_f \) 求导**
\[
\nabla_{\mathbf{w}_f} \mathcal{L} = \sum_{t=1}^T f(x, y_t, t) - \frac{\partial \log Z(x)}{\partial \mathbf{w}_f}
\]
其中：
\[
\frac{\partial \log Z(x)}{\partial \mathbf{w}_f} = \mathbb{E}_{P(y'|x)}\left[\sum_{t=1}^T f(x, y'_t, t)\right]
\]

---

### **6. 代码实现细节**
- **`true_features`**：通过遍历真实标签序列 `tags` 提取所有激活的特征。
- **`expected_features`**：通过前向-后向算法计算所有位置、所有标签对的特征期望概率和。
- **稀疏存储**：使用 `defaultdict` 仅存储非零特征，节省内存。

---

### **7. 总结**
| 数学公式                                                                 | 代码实现                          | 作用                           |
|--------------------------------------------------------------------------|-----------------------------------|--------------------------------|
| \( \sum_t \mathbb{I}[f \in \mathcal{F}(x, y_t, t)] \)                   | `for f in true_features: grad[f] += 1` | 统计真实特征出现次数           |
| \( \mathbb{E}_{P(y'|x)}[\sum_t \mathbb{I}[f \in \mathcal{F}(x, y'_t, t)]] \) | `for f in expected_features: grad[f] -= prob` | 减去特征模型期望             |
| \( \nabla_{\mathbf{w}_f} \mathcal{L} = \text{true\_count} - \text{expected\_count} \) | `gradient[f] = true - expected`   | 更新权重                       |

该梯度公式是 CRF 训练的核心，确保模型通过梯度上升最大化真实标签序列的对数似然。