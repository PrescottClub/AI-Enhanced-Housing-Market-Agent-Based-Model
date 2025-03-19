# 住房市场多智能体模拟系统 (Housing Market ABM)

## 项目概述

本项目实现了一个高级多智能体模拟系统（Agent-Based Model, ABM），用于模拟城市住房市场动态与绅士化过程。通过复杂的智能体交互模型，系统能够真实再现不同市场条件下的房产价值波动、收入不平等发展和社区多样性变化。特别关注了户籍政策（hukou）对中国特色住房市场的影响，以及不同类型投资者如何塑造市场动态。

## 主要特性

- **多层次智能体系统**：
  - 居民（按教育水平、生命阶段、家庭规模和户籍状态分类）
  - 投资者（具有不同投资策略、风险承受能力和投资期限）
  - 房产（分为公寓、住宅、豪华房和经济适用房）
  - 商业实体（影响周围居民收入和房产价值）
  - 政府（实施各种住房政策和市场干预措施）

- **复杂市场动态模拟**：
  - 房产价值受位置、邻里特征和市场周期影响的精确模型
  - 基于多因素的居民满意度和迁移决策系统
  - 投资者根据预期回报和市场情绪做出复杂购买/出售决策
  - 模拟房地产市场周期、繁荣与萧条

- **政府政策影响**：
  - 动态户籍限制政策实施
  - 贷款限制和利率调整
  - 土地释放和经济适用房计划
  - 税收政策和市场干预措施

- **高级数据收集与分析**：
  - 追踪基尼系数和邻里多样性等绅士化指标
  - 通过参数研究探索市场规模和投资者数量的影响
  - 使用Matplotlib进行全面的可视化分析

## 技术要求

- Python 3.7+
- NumPy
- Matplotlib
- Pandas
- Mesa（多智能体建模框架）

## 安装

1. 克隆此仓库：
   ```
   git clone https://github.com/username/housing-market-abm.git
   ```
2. 安装所需依赖包：
   ```
   pip install -r requirements.txt
   ```

## 使用方法

1. 运行主模拟：
   ```
   python run_simulation.py
   ```
2. 进行参数研究：
   ```
   python parameter_study.py
   ```
3. 也可直接运行Jupyter notebook：
   ```
   jupyter notebook housing_market_simulation.ipynb
   ```

## 模型详解

### 主要智能体类型

**居民智能体**
- 具有收入、户籍状态、教育水平和家庭规模等属性
- 基于收入变化、位置偏好、物业满意度等因素做出住房决策
- 随时间推移建立社区依附感，影响住房稳定性

**投资者智能体**
- 实现四种投资策略：价值型、增长型、机会型和保守型
- 考虑风险承受能力、市场情绪和投资期限做出决策
- 通过资本配置和资产组合管理优化回报

**房产智能体**
- 模拟四种房产类型：公寓、住宅、豪华房和经济适用房
- 基于位置、邻里特征、市场周期和政府政策调整价值
- 包含详细属性如年龄、状况和大小

**商业智能体**
- 根据当地收入水平产生和消亡
- 影响周边居民收入和房产价值
- 形成经济活动集聚区

**政府智能体**
- 根据市场状况实施政策，包括：
  - 购买限制（如户籍要求）
  - 贷款条件和利率调整
  - 经济适用房开发
  - 房产税调整
  - 直接市场干预措施

### 模拟过程

模拟以月为单位运行，通常模拟10年（120个月）的市场演变。每个步骤中：
1. 所有智能体按随机顺序行动
2. 政府评估市场状况并可能实施新政策
3. 房产价值根据多种因素更新
4. 居民更新满意度并可能搬迁
5. 投资者根据市场情况购买或出售房产
6. 商业实体可能扩张或关闭
7. 收集关键指标数据用于分析

## 分析结果

模型产生多种关键输出：
- 平均房产价值、基尼系数和邻里多样性的时间序列
- 居民收入、房产价值和满意度水平的分布
- 不同参数设置下这些指标的可视化
- 投资者表现与策略效果分析
- 政府政策有效性评估

## 模型应用

本模型可应用于：
- 预测不同政策对住房市场和社会平等的影响
- 研究投资行为如何影响市场稳定性和价格
- 分析绅士化过程的发展和缓解策略
- 探索户籍政策对城市住房市场的特殊影响
- 识别可能导致市场泡沫或危机的条件

## 贡献

欢迎对模型改进或扩展功能的贡献。请随时提交问题或拉取请求。

## 许可

本项目采用MIT许可证 - 详情请参见[LICENSE](LICENSE)文件。

## 参考文献

本项目研究基础：
- Benenson, I., & Torrens, P. (2004). *Geosimulation: Automata-based modeling of urban phenomena*. John Wiley & Sons.
- Schelling, T. C. (1971). Dynamic models of segregation. *Journal of mathematical sociology, 1*(2), 143-186.
- Batty, M. (2007). *Cities and complexity: Understanding cities with cellular automata, agent-based models, and fractals*. MIT press.
- Wu, F. (2010). Housing environment preference of young consumers in Guangzhou, China: Using the analytic hierarchy process. *Property Management, 28*(3), 174-192.
- Huang, Y., & Clark, W. A. (2002). Housing tenure choice in transitional urban China: A multilevel analysis. *Urban Studies, 39*(1), 7-32.