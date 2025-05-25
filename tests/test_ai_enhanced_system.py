"""
AI增强住房市场仿真系统测试脚本
用于验证各个AI组件的功能是否正常
"""

import numpy as np
import matplotlib.pyplot as plt
from ai_enhanced_housing_model import (
    AIEnhancedHousingMarketModel,
    ReinforcementLearningAgent, 
    MarketPredictor,
    LLMAdvisor,
    run_ai_enhanced_simulation
)

def test_reinforcement_learning_agent():
    """测试强化学习智能体"""
    print("🧠 测试强化学习智能体...")
    
    # 创建RL智能体
    rl_agent = ReinforcementLearningAgent(state_dim=10, action_dim=4)
    
    # 测试决策制定
    state = np.random.rand(10)
    available_actions = [0, 1, 2, 3]
    
    action = rl_agent.make_decision(state, available_actions)
    print(f"   ✅ 决策制定测试通过 - 选择动作: {action}")
    
    # 测试经验学习
    experience = (state, action, 0.1, np.random.rand(10), False)
    rl_agent.update_model(experience)
    print(f"   ✅ 经验学习测试通过 - 经验池大小: {len(rl_agent.memory)}")
    
    return True

def test_market_predictor():
    """测试市场预测器"""
    print("📈 测试市场预测器...")
    
    # 创建预测器
    predictor = MarketPredictor()
    
    # 创建模拟模型进行测试
    model = AIEnhancedHousingMarketModel(100, 20, 10, 10)
    
    # 测试特征提取
    features = predictor.extract_features(model)
    print(f"   ✅ 特征提取测试通过 - 特征维度: {len(features)}")
    
    # 测试预测（未训练状态）
    prediction = predictor.predict_market_trends(model)
    print(f"   ✅ 市场预测测试通过 - 预测结果: {prediction}")
    
    return True

def test_llm_advisor():
    """测试LLM顾问"""
    print("🤖 测试LLM顾问...")
    
    advisor = LLMAdvisor()
    
    # 模拟市场数据
    market_data = {
        'avg_price': 1500000,
        'avg_income': 120000,
        'vacancy_rate': 0.08,
        'gini_coefficient': 0.45,
        'hukou_restriction_rate': 0.3
    }
    
    # 测试市场分析
    analysis = advisor.analyze_market_situation(market_data)
    print(f"   ✅ 市场分析测试通过")
    print(f"      市场分析: {analysis['market_analysis']}")
    print(f"      投资建议: {analysis['investment_advice']}")
    
    return True

def test_ai_enhanced_model():
    """测试完整的AI增强模型"""
    print("🏠 测试完整AI增强模型...")
    
    # 创建小规模模型进行快速测试
    model = AIEnhancedHousingMarketModel(
        num_residents=50,
        num_investors=10, 
        width=10,
        height=10
    )
    
    print(f"   ✅ 模型创建成功")
    print(f"      居民数量: {len([a for a in model.schedule.agents if hasattr(a, 'income')])}")
    print(f"      投资者数量: {len([a for a in model.schedule.agents if hasattr(a, 'capital')])}")
    print(f"      房产数量: {len(model.properties)}")
    
    # 运行几步测试
    for i in range(5):
        model.step()
        
    print(f"   ✅ 模型运行测试通过 - 运行了 {model.schedule.steps} 步")
    
    # 测试AI分析功能
    analysis = model.get_comprehensive_analysis()
    print(f"   ✅ AI综合分析测试通过")
    print(f"      当前平均房价: ¥{analysis['market_data']['avg_price']:,.0f}")
    print(f"      AI价格趋势预测: {analysis['ai_prediction']['price_trend']:.3f}")
    
    return True

def test_simulation_run():
    """测试完整仿真运行"""
    print("🚀 测试完整仿真运行...")
    
    try:
        # 运行小规模仿真
        model, results = run_ai_enhanced_simulation(steps=10, save_results=False)
        
        print(f"   ✅ 仿真运行测试通过")
        print(f"      最终步数: {model.schedule.steps}")
        print(f"      数据点数量: {len(results)}")
        
        # 检查关键指标
        final_price = results['Average Property Value'].iloc[-1]
        ai_prediction = results['AI Market Prediction'].iloc[-1]
        resident_satisfaction = results['Resident Satisfaction'].iloc[-1]
        
        print(f"      最终房价: ¥{final_price:,.0f}")
        print(f"      AI最终预测: {ai_prediction:.3f}")
        print(f"      居民满意度: {resident_satisfaction:.3f}")
        
        return True
    except Exception as e:
        print(f"   ❌ 仿真运行测试失败: {e}")
        return False

def visualize_test_results():
    """可视化测试结果"""
    print("📊 生成测试可视化...")
    
    try:
        # 运行短期仿真获取数据
        model, results = run_ai_enhanced_simulation(steps=20, save_results=False)
        
        # 创建测试图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('AI增强住房市场仿真系统测试结果', fontsize=16)
        
        # 房价趋势
        axes[0,0].plot(results['Average Property Value'], 'b-', linewidth=2)
        axes[0,0].set_title('房价趋势')
        axes[0,0].set_ylabel('房价 (¥)')
        axes[0,0].grid(True)
        
        # AI预测 vs 实际
        axes[0,1].plot(results['Average Property Value'], 'b-', label='实际房价', alpha=0.7)
        axes[0,1].plot(results['AI Market Prediction'], 'r--', label='AI预测', alpha=0.7)
        axes[0,1].set_title('AI预测准确性')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # 居民满意度
        axes[1,0].plot(results['Resident Satisfaction'], 'g-', linewidth=2)
        axes[1,0].set_title('居民满意度')
        axes[1,0].set_ylabel('满意度')
        axes[1,0].set_xlabel('时间步')
        axes[1,0].grid(True)
        
        # 投资表现
        axes[1,1].plot(results['Investment Performance'], 'm-', linewidth=2)
        axes[1,1].set_title('AI投资表现')
        axes[1,1].set_ylabel('投资回报率')
        axes[1,1].set_xlabel('时间步')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('ai_enhanced_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ 测试可视化完成 - 图表已保存为 ai_enhanced_test_results.png")
        return True
        
    except Exception as e:
        print(f"   ❌ 可视化测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("🎯 开始AI增强住房市场仿真系统全面测试...\n")
    
    tests = [
        ("强化学习智能体", test_reinforcement_learning_agent),
        ("市场预测器", test_market_predictor), 
        ("LLM顾问", test_llm_advisor),
        ("AI增强模型", test_ai_enhanced_model),
        ("完整仿真运行", test_simulation_run),
        ("结果可视化", visualize_test_results)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                failed += 1
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} 测试异常: {e}")
    
    print(f"\n{'='*50}")
    print(f"🏁 测试完成!")
    print(f"✅ 通过: {passed}/{len(tests)}")
    print(f"❌ 失败: {failed}/{len(tests)}")
    
    if failed == 0:
        print("🎉 所有测试均通过！AI增强系统运行正常。")
        print("\n📋 建议下一步:")
        print("1. 运行完整仿真: run_ai_enhanced_simulation(steps=120)")
        print("2. 调整AI参数以优化性能")
        print("3. 集成真实数据源")
    else:
        print("⚠️  部分测试失败，请检查依赖项和配置。")
    
    return passed, failed

if __name__ == "__main__":
    # 运行所有测试
    passed, failed = run_all_tests()
    
    # 输出详细的系统信息
    print(f"\n📋 系统信息:")
    try:
        import torch
        print(f"   PyTorch版本: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
    except ImportError:
        print("   PyTorch: 未安装")
    
    try:
        import sklearn
        print(f"   scikit-learn版本: {sklearn.__version__}")
    except ImportError:
        print("   scikit-learn: 未安装")
    
    try:
        import mesa
        print(f"   Mesa版本: {mesa.__version__}")
    except ImportError:
        print("   Mesa: 未安装") 