
# 一. Viterbi算法

Viterbi算法用于求解最优标签序列，其核心是动态规划
$\delta_t(j) = \max_{i} \left[ \delta_{t-1}(i) + \mathbf{T}(i,j) + \mathbf{E}_t(j) \right]$
- $\delta_t(j)$：到时间步$t$、标签$j$的最大得分
- $\mathbf{T}(i,j)$：从标签$i$转移到$j$的分数
- $\mathbf{E}_t(j)$：时间步$t$标签$j$的发射分数


# 二. 隐马尔可夫模型 (HMM)

### HMM参数估计的数学原理与实现

#### 1. HMM的三个核心参数
隐马尔可夫模型(HMM)由以下三个概率分布定义：
1. 初始概率(π)：每个标签作为序列开头的概率  
 $$\pi_i = P(y_1 = i)$$
2. 转移概率(A)：从标签 \(i\) 转移到标签 \(j\) 的概率  
 $$a_{ij} = P(y_t = j | y_{t-1} = i)$$
3. 发射概率(B)：标签 \(i\) 生成观测词 \(w\) 的概率  
 $$b_i(w) = P(x_t = w | y_t = i)$$

---

#### 2. 参数估计方法(极大似然估计)
通过统计训练数据中的频率，用加一平滑(Laplace Smoothing)解决零概率问题。

##### (1) 初始概率估计

$$
  \hat{\pi}_i = \frac{\text{Count}(y_1 = i) + 1}{\text{总句子数} + \text{标签类别数}}
$$


##### (2) 转移概率估计

$$
  \hat{a}_{ij} = \frac{\text{Count}(i \to j) + 1}{\text{Count}(i \to \text{任意标签}) + \text{标签类别数}}
$$

##### (3) 发射概率估计

$$
  \hat{b}_i(w) = \frac{\text{Count}(i \text{生成} w) + 1}{\text{Count}(i) + \text{词汇表大小}}
$$



# 三. 条件随机场 (CRF)

### 1. 前向概率(α)的递推
定义：$\alpha_t(j)$表示在时间步$t$，以标签$j$结尾的所有可能路径的非归一化概率和。

递推公式：
$$
\alpha_t(j) = \sum_{i=1}^N \alpha_{t-1}(i) \cdot \mathbf{T}(i,j) \cdot \mathbf{E}_t(j)
$$
- $\alpha_{t-1}(i)$：在时间步$t-1$，以标签$i$结尾的所有路径的概率和。
- $\mathbf{T}(i,j)$：从标签$i$转移到$j$的转移概率(未归一化)。
- $\mathbf{E}_t(j)$：在时间步$t$，观测到标签$j$的发射概率(未归一化)。

### 2. 后向概率(β)的递推
定义：$\beta_t(i)$表示从时间步$t$开始，以标签$i$为起点的所有可能路径的非归一化概率和。

递推公式：
$$
\beta_t(i) = \sum_{j=1}^N \mathbf{T}(i,j) \cdot \mathbf{E}_{t+1}(j) \cdot \beta_{t+1}(j)
$$
- $\mathbf{T}(i,j)$：从标签$i$转移到$j$的转移概率。
- $\mathbf{E}_{t+1}(j)$：在时间步$t+1$，观测到标签$j$的发射概率。
- $\beta_{t+1}(j)$：从$t+1$步开始，以$j$为起点的路径概率和。


### 3. 配分函数(Z)的计算
定义：$Z$是所有可能路径的总概率，用于归一化。
$$
Z = \sum_{j=1}^N \alpha_T(j)
$$


### 4. 联合概率计算
### (1) 单个标签的边缘概率
$$
P(y_t = j | x) = \frac{\alpha_t(j) \cdot \beta_t(j)}{Z}
$$
- 分子$\alpha_t(j) \cdot \beta_t(j)$：所有经过$y_t = j$的路径概率和。
- 分母$Z$：归一化因子，确保概率总和为 1。

### (2) 相邻标签的联合概率
$$
P(y_{t-1} = i, y_t = j | x) = \frac{\alpha_{t-1}(i) \cdot \mathbf{T}(i,j) \cdot \mathbf{E}_t(j) \cdot \beta_t(j)}{Z}
$$
- 分子：所有满足$y_{t-1} = i$且$y_t = j$的路径概率和。
- 分母$Z$：归一化因子。

### 5. 权重梯度
$$
\frac{\partial \log P(y|x)}{\partial w_f} = \text{Count}(f \text{ in true path}) - \mathbb{E}_{P(y|x)}[\text{Count}(f)]
$$
- $\text{Count}(f \text{ in true path})$：特征$f$在真实标签序列中出现的次数。
- $\mathbb{E}_{P(y|x)}[\text{Count}(f)]$：特征$f$在所有可能路径中的期望出现次数(通过前向-后向算法计算)。



### 6. 转移矩阵梯度
$$
\frac{\partial \log P(y|x)}{\partial T(i,j)} = \text{Count}(i \to j \text{ in true path}) - \mathbb{E}_{P(y|x)}[\text{Count}(i \to j)]
$$
- $\text{Count}(i \to j \text{ in true path})$：真实路径中从标签$i$转移到$j$的次数。
- $\mathbb{E}_{P(y|x)}[\text{Count}(i \to j)]$：所有路径中$i \to j$转移的期望次数。


# 四. CRF+Transformer

### 1. 输入表示(Input Representation)
- Token Embedding：  
  将输入序列$X = (x_1, \dots, x_T)$映射为向量：
$$
  \mathbf{E}(x_t) \in \mathbb{R}^{d_{\text{model}}}
$$
- Positional Encoding(正弦/余弦函数)：  
  添加位置信息：
$$
  \mathbf{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d_{\text{model}}}}\right), \quad \mathbf{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d_{\text{model}}}}\right)
$$
  最终得到：
$$
  \mathbf{h}_t^{(0)} = \mathbf{E}(x_t) + \mathbf{PE}(t)
$$

---

### 2. 编码器(Encoder)
每层编码器包含两个子层：

#### (1) 多头自注意力(Multi-Head Self-Attention)
- Query/Key/Value 投影：
$$
  \mathbf{Q} = \mathbf{h}^{(l-1)}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{h}^{(l-1)}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{h}^{(l-1)}\mathbf{W}_V
$$
- 缩放点积注意力：
$$
  \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$
- 多头拼接：
$$
  \text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}_O
$$

#### (2) 前馈神经网络(Feed-Forward Network)
$$
\text{FFN}(\mathbf{x}) = \text{ReLU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
$$

#### (3) 残差连接与层归一化
$$
\mathbf{h}^{(l)} = \text{LayerNorm}\left(\mathbf{h}^{(l-1)} + \text{Dropout}(\text{SubLayer}(\mathbf{h}^{(l-1)}))\right)
$$

---

### 3. 解码器(Decoder)
每层解码器包含三个子层：

#### (1) 掩码多头自注意力(Masked Self-Attention)
- 防止未来信息泄露：
$$
  \text{Mask}(i,j) = \begin{cases} 
  0 & \text{if } i \geq j \\
  -\infty & \text{if } i < j 
  \end{cases}
$$
  注意力权重：
$$
  \mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \text{Mask}\right)
$$

#### (2) 编码器-解码器注意力(Cross-Attention)
- Query 来自解码器，Key/Value 来自编码器：
$$
  \mathbf{Q} = \mathbf{h}_{\text{dec}}^{(l-1)}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{h}_{\text{enc}}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{h}_{\text{enc}}\mathbf{W}_V
$$

#### (3) 前馈神经网络与残差连接
- 与编码器相同

---

### 4. 输出层(Projection Layer)
- 线性变换 + Softmax：
$$
  P(y_t | y_{<t}, X) = \text{softmax}(\mathbf{h}_t^{(L)}\mathbf{W}_{\text{proj}})
$$
