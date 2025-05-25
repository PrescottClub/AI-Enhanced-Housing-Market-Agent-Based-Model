# AI增强住房市场多智能体仿真系统使用指南

## 🎯 项目概述

本项目是对原有住房市场ABM的AI增强版本，集成了多种人工智能技术来提升仿真的智能性和预测能力。

### 🚀 AI增强功能

1. **强化学习智能体** - 投资者使用深度Q网络进行投资决策
2. **机器学习预测** - 随机森林模型预测市场趋势
3. **大语言模型顾问** - 提供市场分析和政策建议
4. **智能决策系统** - 居民基于AI分析做出住房决策
5. **动态市场评分** - 房产智能体实时评估市场价值

## 📦 安装和配置

### 1. 环境设置

```bash
# 创建虚拟环境
python -m venv ai_housing_env
source ai_housing_env/bin/activate  # Windows: ai_housing_env\Scripts\activate

# 安装依赖
pip install -r requirements_ai_enhanced.txt
```

### 2. 可选配置

```python
# 配置OpenAI API（可选）
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

# 配置CUDA（如果有GPU）
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 🎮 快速开始

### 基础运行

```python
from ai_enhanced_housing_model import run_ai_enhanced_simulation

# 运行120步仿真
model, results = run_ai_enhanced_simulation(steps=120)
```

### 自定义参数运行

```python
from ai_enhanced_housing_model import AIEnhancedHousingMarketModel

# 创建自定义模型
model = AIEnhancedHousingMarketModel(
    num_residents=1000,    # 居民数量
    num_investors=150,     # 投资者数量
    width=40,              # 网格宽度
    height=40              # 网格高度
)

# 逐步运行并观察AI决策
for step in range(100):
    model.step()
    
    if step % 10 == 0:
        analysis = model.get_comprehensive_analysis()
        print(f"Step {step}: AI预测趋势 = {analysis['ai_prediction']['price_trend']:.3f}")
```

## 🧠 AI组件详解

### 1. 强化学习投资者

```python
class AIEnhancedInvestor(Agent):
    """每个投资者都有自己的深度Q网络"""
    
    def __init__(self, unique_id, model, capital):
        super().__init__(unique_id, model)
        # RL智能体：状态维度10，动作维度4
        self.rl_agent = ReinforcementLearningAgent(
            state_dim=10,  # 市场状态、个人状态、预测等
            action_dim=4   # 买入、卖出、持有、等待
        )
```

**特点：**
- 🎯 基于市场状态智能决策
- 📈 从经验中持续学习
- ⚡ 实时适应市场变化

### 2. 市场预测系统

```python
class MarketPredictor:
    """使用随机森林预测市场趋势"""
    
    def predict_market_trends(self, model):
        # 提取7维特征向量
        features = self.extract_features(model)
        
        # 预测价格和需求趋势
        price_trend = self.price_model.predict(features)
        demand_trend = self.demand_model.predict(features)
```

**预测特征：**
- 🏠 平均房价和收入水平
- 📊 空置率和投资者行为
- 🏛️ 政策限制程度
- ⏰ 时间序列特征

### 3. LLM市场顾问

```python
class LLMAdvisor:
    """大语言模型提供专业分析"""
    
    def analyze_market_situation(self, market_data):
        # 生成专业市场分析报告
        return {
            "market_analysis": "市场现状专业分析",
            "risk_assessment": "风险评估报告", 
            "investment_advice": "投资建议",
            "policy_advice": "政策建议"
        }
```

**分析维度：**
- 📈 市场趋势分析
- ⚠️ 风险评估
- 💡 投资建议
- 🏛️ 政策建议

### 4. AI增强居民

```python
class AIEnhancedResident(Agent):
    """居民基于AI预测做决策"""
    
    def ai_enhanced_housing_decision(self):
        # 获取市场预测
        market_prediction = self.ai_advisor.predict_market_trends(self.model)
        
        # 综合考虑个人和市场因素
        decision_factors = {
            'current_satisfaction': self.satisfaction,
            'income_trend': self.income_growth_rate,
            'market_price_trend': market_prediction['price_trend'],
            'risk_tolerance': self.risk_tolerance
        }
```

## 📊 数据分析和可视化

### 获取仿真数据

```python
# 运行仿真
model, results = run_ai_enhanced_simulation(steps=120)

# 获取数据框
data = model.datacollector.get_model_vars_dataframe()

# 查看AI预测准确性
print(data[['Average Property Value', 'AI Market Prediction']].corr())
```

### 可视化AI性能

```python
import matplotlib.pyplot as plt

# 绘制AI预测 vs 实际价格
plt.figure(figsize=(12, 6))
plt.plot(data['Average Property Value'], label='实际房价', linewidth=2)
plt.plot(data['AI Market Prediction'], label='AI预测', linewidth=2, alpha=0.7)
plt.legend()
plt.title('AI预测准确性分析')
plt.show()

# 投资者AI决策分析
plt.figure(figsize=(10, 6))
plt.plot(data['Investment Performance'], label='AI投资表现', color='green')
plt.title('AI增强投资者表现')
plt.show()
```

## 🔧 高级配置

### 1. 自定义RL参数

```python
# 创建自定义强化学习智能体
custom_rl_agent = ReinforcementLearningAgent(
    state_dim=15,          # 增加状态维度
    action_dim=6,          # 增加动作选择
    hidden_dim=256,        # 增大网络容量
)

# 调整学习参数
custom_rl_agent.epsilon = 0.05      # 降低探索率
custom_rl_agent.optimizer = optim.Adam(
    custom_rl_agent.q_network.parameters(), 
    lr=0.0005              # 调整学习率
)
```

### 2. 集成真实API

```python
# 集成OpenAI API
class RealLLMAdvisor(LLMAdvisor):
    def __init__(self, api_key):
        super().__init__(api_key)
        import openai
        openai.api_key = api_key
    
    def _call_real_llm(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content
```

### 3. 添加新的AI功能

```python
# 自定义AI组件
class SentimentAnalyzer:
    """市场情绪分析器"""
    
    def analyze_market_sentiment(self, model):
        # 分析居民满意度分布
        satisfactions = [r.satisfaction for r in model.schedule.agents 
                        if isinstance(r, AIEnhancedResident)]
        
        # 计算市场情绪指标
        sentiment_score = np.mean(satisfactions)
        volatility = np.std(satisfactions)
        
        return {
            'sentiment': 'positive' if sentiment_score > 0.6 else 'negative',
            'confidence': 1 - volatility,
            'score': sentiment_score
        }
```

## 📈 性能优化建议

### 1. GPU加速

```python
# 确保PyTorch使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 将模型移至GPU
rl_agent.q_network = rl_agent.q_network.to(device)
```

### 2. 并行化处理

```python
from joblib import Parallel, delayed

def run_parallel_simulations(num_runs=10):
    """并行运行多次仿真"""
    
    def single_run(run_id):
        model = AIEnhancedHousingMarketModel(800, 120, 30, 30)
        for _ in range(120):
            model.step()
        return model.datacollector.get_model_vars_dataframe()
    
    # 并行执行
    results = Parallel(n_jobs=-1)(
        delayed(single_run)(i) for i in range(num_runs)
    )
    
    return results
```

### 3. 模型保存和加载

```python
# 保存训练好的模型
torch.save(investor.rl_agent.q_network.state_dict(), 'trained_investor_model.pth')

# 加载预训练模型
investor.rl_agent.q_network.load_state_dict(
    torch.load('trained_investor_model.pth')
)
```

## 🔍 故障排除

### 常见问题

1. **CUDA内存不足**
   ```python
   # 减少批处理大小
   rl_agent._train_batch(batch_size=16)  # 默认32
   ```

2. **模型收敛慢**
   ```python
   # 调整学习率和网络结构
   optimizer = optim.Adam(network.parameters(), lr=0.01)  # 增大学习率
   ```

3. **预测准确性低**
   ```python
   # 增加训练数据
   if len(historical_data) < 50:  # 增加最小训练数据量
       return default_prediction
   ```

## 🚀 扩展建议

### 1. 多模态AI集成

- 🖼️ 图像识别：分析卫星图像预测区域发展
- 🗣️ 语音处理：集成语音政策公告分析
- 📄 文本挖掘：分析新闻情感对市场影响

### 2. 高级强化学习

- 🔄 Actor-Critic网络
- 🎯 多智能体强化学习（MARL）
- 🧠 注意力机制

### 3. 实时数据集成

- 📡 爬取实时房价数据
- 📊 接入经济指标API
- 🏛️ 政策文档自动分析

## 📚 参考资料

- [Mesa文档](https://mesa.readthedocs.io/)
- [PyTorch教程](https://pytorch.org/tutorials/)
- [强化学习基础](https://spinningup.openai.com/)
- [scikit-learn用户指南](https://scikit-learn.org/stable/user_guide.html)

---

**💡 提示**: 这个AI增强系统为住房市场仿真提供了强大的智能分析能力。通过持续的模型训练和参数调优，可以获得更准确的市场预测和更智能的决策制定。 