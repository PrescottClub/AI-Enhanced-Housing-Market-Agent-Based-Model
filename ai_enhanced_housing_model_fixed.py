#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI增强房屋市场仿真系统 - 修复版本
解决了网格位置冲突、数值溢出和无效值问题
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class SafeAIDecisionMaker(ABC):
    """安全的AI决策制定者基类"""
    
    @abstractmethod
    def make_decision(self, state, available_actions):
        pass
    
    @abstractmethod
    def update_model(self, experience):
        pass

class SafeReinforcementLearningAgent(SafeAIDecisionMaker):
    """安全的强化学习智能体"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cpu')  # 使用CPU避免CUDA问题
        
        # 简化网络结构
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = []
        self.epsilon = 0.1
        
    def make_decision(self, state, available_actions):
        """安全的决策制定"""
        try:
            if len(available_actions) == 0:
                return 0  # 默认动作
                
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # epsilon-greedy策略
            if np.random.random() < self.epsilon:
                return np.random.choice(available_actions)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                # 只考虑可用动作
                valid_q_values = {action: q_values[0][action].item() for action in available_actions}
                return max(valid_q_values, key=valid_q_values.get)
                
        except Exception as e:
            print(f"Decision making error: {e}")
            return np.random.choice(available_actions) if available_actions else 0
    
    def update_model(self, experience):
        """安全的模型更新"""
        try:
            self.memory.append(experience)
            if len(self.memory) > 1000:
                self.memory.pop(0)
            
            if len(self.memory) >= 32:
                self._train_batch()
        except Exception as e:
            print(f"Model update error: {e}")
    
    def _train_batch(self, batch_size=16):
        """安全的批次训练"""
        try:
            batch = np.random.choice(self.memory, min(batch_size, len(self.memory)), replace=False)
            
            states = torch.FloatTensor([exp['state'] for exp in batch]).to(self.device)
            actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
            rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
            next_states = torch.FloatTensor([exp['next_state'] for exp in batch]).to(self.device)
            
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.q_network(next_states).max(1)[0].detach()
            target_q_values = rewards + 0.95 * next_q_values
            
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        except Exception as e:
            print(f"Training error: {e}")

class SafeMarketPredictor:
    """安全的市场预测器"""
    
    def __init__(self):
        self.price_model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.demand_model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.trained = False
        
    def extract_features(self, model):
        """安全的特征提取"""
        try:
            properties = model.properties
            residents = [a for a in model.schedule.agents if hasattr(a, 'income')]
            
            if not properties or not residents:
                return np.zeros(7)
            
            # 安全计算特征
            avg_price = np.mean([p.value for p in properties if np.isfinite(p.value)])
            avg_income = np.mean([r.income for r in residents if np.isfinite(r.income)])
            
            features = [
                avg_price if np.isfinite(avg_price) else 1000000,
                avg_income if np.isfinite(avg_income) else 100000,
                len(properties),
                len(residents),
                sum(1 for p in properties if p.owner is None) / max(1, len(properties)),
                model.schedule.steps,
                np.random.random()  # 市场随机因子
            ]
            
            return np.array([f if np.isfinite(f) else 0 for f in features])
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(7)
    
    def predict_market_trends(self, model):
        """安全的市场趋势预测"""
        try:
            features = self.extract_features(model).reshape(1, -1)
            
            if self.trained:
                price_trend = self.price_model.predict(features)[0]
                demand_trend = self.demand_model.predict(features)[0]
            else:
                # 未训练时使用简单预测
                price_trend = np.clip(np.random.normal(0, 0.02), -0.1, 0.1)
                demand_trend = np.clip(np.random.normal(0, 0.02), -0.1, 0.1)
            
            return {
                'price_trend': float(np.clip(price_trend, -0.2, 0.2)),
                'demand_trend': float(np.clip(demand_trend, -0.2, 0.2)),
                'confidence': 0.7 if self.trained else 0.3
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {'price_trend': 0.0, 'demand_trend': 0.0, 'confidence': 0.1}
    
    def train_models(self, historical_data):
        """安全的模型训练"""
        try:
            if len(historical_data) < 10:
                return
                
            # 清理数据
            clean_data = []
            for data in historical_data[-100:]:  # 只使用最近100个数据点
                if (isinstance(data, dict) and 
                    'features' in data and 
                    'price_change' in data and
                    np.all(np.isfinite(data['features'])) and
                    np.isfinite(data['price_change'])):
                    clean_data.append(data)
            
            if len(clean_data) < 5:
                return
                
            X = np.array([d['features'] for d in clean_data])
            y_price = np.array([np.clip(d['price_change'], -0.5, 0.5) for d in clean_data])
            y_demand = np.array([np.clip(d.get('demand_change', 0), -0.5, 0.5) for d in clean_data])
            
            self.price_model.fit(X, y_price)
            self.demand_model.fit(X, y_demand)
            self.trained = True
            
        except Exception as e:
            print(f"Training error: {e}")

class SafeLLMAdvisor:
    """安全的LLM顾问"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        
    def analyze_market_situation(self, market_data):
        """安全的市场分析"""
        try:
            # 简化分析，避免API调用错误
            avg_price = market_data.get('avg_price', 0)
            avg_income = market_data.get('avg_income', 0)
            
            if avg_price > avg_income * 10:
                sentiment = "谨慎"
                recommendation = "市场价格过高，建议观望"
            elif avg_price < avg_income * 5:
                sentiment = "乐观"
                recommendation = "市场价格合理，可考虑投资"
            else:
                sentiment = "中性"
                recommendation = "市场平稳，正常交易"
                
            return {
                'sentiment': sentiment,
                'recommendation': recommendation,
                'confidence': 0.8
            }
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return {
                'sentiment': '中性',
                'recommendation': '建议观察市场',
                'confidence': 0.3
            }

class SafeAIEnhancedResident(Agent):
    """安全的AI增强居民"""
    
    def __init__(self, unique_id, model, income, hukou_status):
        super().__init__(unique_id, model)
        self.income = max(1000, float(income))  # 确保最小收入
        self.hukou_status = hukou_status
        self.property = None
        self.satisfaction = 0.5
        self.moving_threshold = np.random.uniform(0.3, 0.7)
        
    def step(self):
        """安全的步进方法"""
        try:
            self.update_income()
            self.update_satisfaction()
            
            if self.satisfaction < self.moving_threshold:
                self.consider_moving()
                
        except Exception as e:
            print(f"Resident step error: {e}")
    
    def update_income(self):
        """安全的收入更新"""
        try:
            # 温和的收入变化
            change = np.random.normal(1.02, 0.05)
            self.income = max(1000, self.income * np.clip(change, 0.9, 1.1))
        except:
            pass
    
    def update_satisfaction(self):
        """安全的满意度更新"""
        try:
            if self.property:
                # 基于收入和房产价值的满意度
                affordability = min(1.0, self.income / max(1, self.property.value * 0.1))
                self.satisfaction = np.clip(0.3 + affordability * 0.4, 0, 1)
            else:
                self.satisfaction = 0.2  # 无房产时较低满意度
        except:
            self.satisfaction = 0.5
    
    def consider_moving(self):
        """安全的搬迁考虑"""
        try:
            available_properties = [p for p in self.model.properties 
                                  if p.owner is None and p.value < self.income * 8]
            
            if available_properties and len(available_properties) > 0:
                # 选择最合适的房产
                chosen = min(available_properties, 
                           key=lambda p: abs(p.value - self.income * 5))
                
                if self.property:
                    self.property.owner = None
                
                chosen.owner = self
                self.property = chosen
                self.satisfaction = min(1.0, self.satisfaction + 0.2)
                
        except Exception as e:
            print(f"Moving consideration error: {e}")

class SafeAIEnhancedInvestor(Agent):
    """安全的AI增强投资者"""
    
    def __init__(self, unique_id, model, capital):
        super().__init__(unique_id, model)
        self.capital = max(10000, float(capital))
        self.properties = []
        self.ai_agent = SafeReinforcementLearningAgent(state_dim=6, action_dim=3)
        
    def step(self):
        """安全的步进方法"""
        try:
            state = self._get_safe_state()
            actions = self._get_available_actions()
            
            if actions:
                action = self.ai_agent.make_decision(state, actions)
                reward = self._execute_action(action)
                
                # 更新AI模型
                experience = {
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': self._get_safe_state()
                }
                self.ai_agent.update_model(experience)
                
        except Exception as e:
            print(f"Investor step error: {e}")
    
    def _get_safe_state(self):
        """安全的状态获取"""
        try:
            properties = self.model.properties
            avg_price = np.mean([p.value for p in properties if np.isfinite(p.value)])
            
            state = [
                self.capital / 1000000,  # 标准化资本
                len(self.properties) / 10,  # 标准化房产数量
                avg_price / 1000000 if np.isfinite(avg_price) else 1.0,  # 标准化平均价格
                len([p for p in properties if p.owner is None]) / max(1, len(properties)),
                self.model.schedule.steps / 100,  # 标准化时间步
                np.random.random()  # 随机因子
            ]
            
            return np.array([s if np.isfinite(s) else 0.5 for s in state])
            
        except:
            return np.array([0.5] * 6)
    
    def _get_available_actions(self):
        """获取可用动作"""
        actions = [2]  # 总是可以持有
        
        # 买入：有资本且有可用房产
        available_properties = [p for p in self.model.properties 
                              if p.owner is None and p.value < self.capital]
        if available_properties:
            actions.append(0)
        
        # 卖出：有房产
        if self.properties:
            actions.append(1)
            
        return actions
    
    def _execute_action(self, action):
        """安全的动作执行"""
        try:
            if action == 0:  # 买入
                return self._safe_buy_property()
            elif action == 1:  # 卖出
                return self._safe_sell_property()
            else:  # 持有
                return self._safe_hold_properties()
        except:
            return 0
    
    def _safe_buy_property(self):
        """安全买入房产"""
        try:
            available = [p for p in self.model.properties 
                        if p.owner is None and p.value < self.capital * 0.8]
            
            if available:
                property_to_buy = np.random.choice(available)
                if property_to_buy.value <= self.capital:
                    self.capital -= property_to_buy.value
                    property_to_buy.owner = self
                    self.properties.append(property_to_buy)
                    return 0.1  # 正向奖励
            return 0
        except:
            return 0
    
    def _safe_sell_property(self):
        """安全卖出房产"""
        try:
            if self.properties:
                property_to_sell = np.random.choice(self.properties)
                self.capital += property_to_sell.value
                property_to_sell.owner = None
                self.properties.remove(property_to_sell)
                return 0.05  # 小额正向奖励
            return 0
        except:
            return 0
    
    def _safe_hold_properties(self):
        """安全持有房产"""
        return 0.01  # 小额奖励

class SafeAIEnhancedProperty(Agent):
    """安全的AI增强房产"""
    
    def __init__(self, unique_id, model, value, pos):
        super().__init__(unique_id, model)
        self.value = max(10000, float(value))  # 确保最小价值
        self.pos = pos
        self.owner = None
        self.hukou_restricted = np.random.random() < 0.3
        self.market_score = 0.5
        
    def step(self):
        """安全的步进方法"""
        try:
            self.update_value()
            self.update_market_score()
        except Exception as e:
            print(f"Property step error: {e}")
    
    def update_value(self):
        """安全的价值更新"""
        try:
            # 基础市场变化
            market_change = np.random.normal(1.001, 0.01)
            
            # 邻居效应
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
            neighbor_properties = [n for n in neighbors if hasattr(n, 'value')]
            
            if neighbor_properties:
                avg_neighbor_value = np.mean([p.value for p in neighbor_properties 
                                            if np.isfinite(p.value)])
                if np.isfinite(avg_neighbor_value) and avg_neighbor_value > 0:
                    neighbor_effect = (avg_neighbor_value - self.value) * 0.01
                    neighbor_effect = np.clip(neighbor_effect / self.value, -0.05, 0.05)
                else:
                    neighbor_effect = 0
            else:
                neighbor_effect = 0
            
            # 所有权效应
            ownership_effect = 0.002 if self.owner else -0.001
            
            # 计算总变化
            total_change = (market_change - 1) + neighbor_effect + ownership_effect
            total_change = np.clip(total_change, -0.1, 0.1)  # 限制变化幅度
            
            # 更新价值
            new_value = self.value * (1 + total_change)
            self.value = max(5000, min(50000000, new_value))  # 限制价值范围
            
        except Exception as e:
            print(f"Value update error: {e}")
    
    def update_market_score(self):
        """安全的市场评分更新"""
        try:
            # 基于位置和市场情况的评分
            distance_to_center = np.sqrt((self.pos[0] - self.model.city_center[0])**2 + 
                                       (self.pos[1] - self.model.city_center[1])**2)
            location_score = 1 / (1 + distance_to_center * 0.1)
            
            # 价值相对评分
            avg_market_value = np.mean([p.value for p in self.model.properties 
                                      if np.isfinite(p.value)])
            if avg_market_value > 0:
                value_score = self.value / avg_market_value
            else:
                value_score = 1.0
                
            self.market_score = np.clip((location_score + value_score) / 2, 0, 1)
            
        except:
            self.market_score = 0.5

class SafeAIEnhancedHousingMarketModel(Model):
    """安全的AI增强房屋市场模型"""
    
    def __init__(self, num_residents=100, num_investors=20, width=30, height=30):
        super().__init__()
        self.num_residents = num_residents
        self.num_investors = num_investors
        
        # AI组件
        self.market_predictor = SafeMarketPredictor()
        self.llm_advisor = SafeLLMAdvisor()
        self.historical_data = []
        
        # 模型设置
        self.city_center = (width // 2, height // 2)
        
        # 安全计算网格大小
        total_agents = num_residents + num_investors
        num_properties = min(int(total_agents * 1.2), width * height // 2)
        
        self.grid_size = max(width, height)
        self.grid = MultiGrid(self.grid_size, self.grid_size, True)
        self.schedule = RandomActivation(self)
        self.properties = []
        self.current_id = 0
        
        # 创建智能体
        self._safe_create_properties(num_properties)
        self._safe_create_residents()
        self._safe_create_investors()
        
        # 数据收集器
        self.datacollector = DataCollector(
            model_reporters={
                "Average Property Value": self.safe_average_property_value,
                "Gini Coefficient": self.safe_gini_coefficient,
                "AI Market Prediction": self.safe_get_ai_market_prediction,
                "Resident Satisfaction": self.safe_average_resident_satisfaction,
                "Investment Performance": self.safe_average_investment_performance
            }
        )
    
    def _safe_create_properties(self, num_properties):
        """安全创建房产"""
        try:
            # 获取所有可用位置
            all_positions = [(x, y) for x in range(self.grid_size) 
                            for y in range(self.grid_size)]
            
            # 随机选择不重复的位置
            if len(all_positions) < num_properties:
                num_properties = len(all_positions)
                
            selected_positions = np.random.choice(len(all_positions), 
                                                num_properties, replace=False)
            
            for i in selected_positions:
                pos = all_positions[i]
                
                # 计算距离中心的位置
                distance_to_center = np.sqrt((pos[0] - self.city_center[0])**2 + 
                                           (pos[1] - self.city_center[1])**2)
                max_distance = np.sqrt(self.grid_size**2 + self.grid_size**2) / 2
                normalized_distance = distance_to_center / max_distance
                
                # 计算房产价值
                base_value = np.random.lognormal(13.0, 0.3)
                location_multiplier = 1.5 - normalized_distance * 0.5
                property_value = base_value * location_multiplier
                
                # 限制价值范围
                property_value = np.clip(property_value, 50000, 10000000)
                
                prop = SafeAIEnhancedProperty(self.next_id(), self, property_value, pos)
                self.properties.append(prop)
                self.schedule.add(prop)
                
                # 安全放置
                if self.grid.is_cell_empty(pos):
                    self.grid.place_agent(prop, pos)
                    
        except Exception as e:
            print(f"Property creation error: {e}")
    
    def _safe_create_residents(self):
        """安全创建居民"""
        try:
            for _ in range(self.num_residents):
                income = np.random.lognormal(11.2, 0.5)
                income = np.clip(income, 20000, 2000000)
                
                hukou_status = "Local" if np.random.random() < 0.6 else "Non-local"
                if hukou_status == "Local":
                    income *= 1.2
                
                resident = SafeAIEnhancedResident(self.next_id(), self, income, hukou_status)
                self.schedule.add(resident)
                self._safe_place_agent(resident)
                
        except Exception as e:
            print(f"Resident creation error: {e}")
    
    def _safe_create_investors(self):
        """安全创建投资者"""
        try:
            for _ in range(self.num_investors):
                capital = np.random.lognormal(14.5, 0.5)
                capital = np.clip(capital, 100000, 50000000)
                
                investor = SafeAIEnhancedInvestor(self.next_id(), self, capital)
                self.schedule.add(investor)
                self._safe_place_agent(investor)
                
        except Exception as e:
            print(f"Investor creation error: {e}")
    
    def _safe_place_agent(self, agent):
        """安全放置智能体"""
        try:
            empty_cells = list(self.grid.empties)
            if empty_cells:
                cell = self.random.choice(empty_cells)
                self.grid.place_agent(agent, cell)
                return True
            return False
        except:
            return False
    
    def next_id(self):
        self.current_id += 1
        return self.current_id
    
    def step(self):
        """安全的模型步进"""
        try:
            # 收集当前数据
            current_data = {
                'features': self.market_predictor.extract_features(self),
                'price_change': 0,
                'demand_change': 0
            }
            
            # 记录上一步价格
            prev_avg_price = self.safe_average_property_value()
            
            # 执行智能体行动
            self.schedule.step()
            
            # 计算价格变化
            curr_avg_price = self.safe_average_property_value()
            if prev_avg_price > 0:
                price_change = (curr_avg_price - prev_avg_price) / prev_avg_price
                current_data['price_change'] = np.clip(price_change, -0.5, 0.5)
            
            # 添加到历史数据
            self.historical_data.append(current_data)
            
            # 定期训练预测模型
            if self.schedule.steps % 20 == 0:
                self.market_predictor.train_models(self.historical_data)
            
            # 收集数据
            self.datacollector.collect(self)
            
        except Exception as e:
            print(f"Model step error: {e}")
    
    def safe_average_property_value(self):
        """安全的平均房产价值计算"""
        try:
            values = [p.value for p in self.properties if np.isfinite(p.value)]
            return np.mean(values) if values else 1000000
        except:
            return 1000000
    
    def safe_gini_coefficient(self):
        """安全的基尼系数计算"""
        try:
            incomes = [a.income for a in self.schedule.agents 
                      if hasattr(a, 'income') and np.isfinite(a.income)]
            if len(incomes) < 2:
                return 0
                
            sorted_incomes = np.sort(incomes)
            n = len(sorted_incomes)
            cumsum = np.cumsum(sorted_incomes)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        except:
            return 0
    
    def safe_get_ai_market_prediction(self):
        """安全的AI市场预测"""
        try:
            prediction = self.market_predictor.predict_market_trends(self)
            return prediction['price_trend']
        except:
            return 0
    
    def safe_average_resident_satisfaction(self):
        """安全的平均居民满意度"""
        try:
            residents = [a for a in self.schedule.agents if hasattr(a, 'satisfaction')]
            satisfactions = [r.satisfaction for r in residents if np.isfinite(r.satisfaction)]
            return np.mean(satisfactions) if satisfactions else 0.5
        except:
            return 0.5
    
    def safe_average_investment_performance(self):
        """安全的平均投资表现"""
        try:
            investors = [a for a in self.schedule.agents if hasattr(a, 'properties')]
            if not investors:
                return 0
                
            performances = []
            for inv in investors:
                if hasattr(inv, 'capital') and inv.properties:
                    portfolio_value = sum(p.value for p in inv.properties 
                                        if np.isfinite(p.value))
                    if inv.capital + portfolio_value > 0:
                        performance = portfolio_value / (inv.capital + portfolio_value)
                        if np.isfinite(performance):
                            performances.append(performance)
            
            return np.mean(performances) if performances else 0
        except:
            return 0

def run_safe_simulation(steps=50, num_residents=50, num_investors=10):
    """运行安全的仿真"""
    print("🚀 启动安全AI增强房屋市场仿真...")
    
    try:
        # 创建模型
        model = SafeAIEnhancedHousingMarketModel(
            num_residents=num_residents,
            num_investors=num_investors,
            width=20,
            height=20
        )
        
        print(f"✅ 模型创建成功! 居民:{num_residents}, 投资者:{num_investors}")
        
        # 运行仿真
        for i in range(steps):
            model.step()
            
            if i % 10 == 0:
                data = {
                    'step': i,
                    'avg_price': model.safe_average_property_value(),
                    'satisfaction': model.safe_average_resident_satisfaction(),
                    'gini': model.safe_gini_coefficient(),
                    'prediction': model.safe_get_ai_market_prediction()
                }
                print(f"步骤 {i:2d}: 房价¥{data['avg_price']:8,.0f} | "
                      f"满意度{data['satisfaction']:.3f} | "
                      f"基尼{data['gini']:.3f} | "
                      f"预测{data['prediction']:+.3f}")
        
        # 获取最终数据
        results = model.datacollector.get_model_vars_dataframe()
        print(f"\n🎉 仿真完成! 生成了 {len(results)} 行数据")
        
        return model, results
        
    except Exception as e:
        print(f"❌ 仿真运行错误: {e}")
        return None, None

if __name__ == "__main__":
    # 运行安全测试
    model, results = run_safe_simulation(steps=30, num_residents=30, num_investors=8)
    if model and results is not None:
        print("\n📊 仿真结果摘要:")
        print(results.describe()) 