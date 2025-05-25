#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏠 AI增强房屋市场仿真系统 - 完整启动脚本
专为解决仿真运行失败问题而设计
"""

import sys
import os

def check_dependencies():
    """检查依赖包"""
    print("🔍 检查系统依赖...")
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'), 
        ('matplotlib', 'Matplotlib'),
        ('mesa', 'Mesa'),
        ('torch', 'PyTorch'),
        ('sklearn', 'Scikit-learn')
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}: 已安装")
        except ImportError:
            print(f"❌ {name}: 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少依赖包: {', '.join(missing_packages)}")
        print("💡 请运行: pip install -r requirements_ai_enhanced.txt")
        return False
    
    print("✅ 所有依赖包检查通过！")
    return True

def run_demo_simulation():
    """运行演示仿真"""
    try:
        print("\n🚀 启动AI增强房屋市场仿真演示...")
        
        # 导入修复版本模型
        import sys; sys.path.append("../src"); from ai_enhanced_housing_model_fixed import run_safe_simulation
        
        # 运行小规模演示
        print("📊 运行小规模市场仿真...")
        model, results = run_safe_simulation(
            steps=20,
            num_residents=25, 
            num_investors=8
        )
        
        if model and results is not None:
            print("\n📈 演示仿真结果:")
            print(f"   初始房价: ¥{results['Average Property Value'].iloc[0]:,.0f}")
            print(f"   最终房价: ¥{results['Average Property Value'].iloc[-1]:,.0f}")
            print(f"   价格变化: {((results['Average Property Value'].iloc[-1] / results['Average Property Value'].iloc[0]) - 1) * 100:+.2f}%")
            print(f"   最终满意度: {results['Resident Satisfaction'].iloc[-1]:.3f}")
            print(f"   基尼系数: {results['Gini Coefficient'].iloc[-1]:.3f}")
            print(f"   AI预测: {results['AI Market Prediction'].iloc[-1]:+.3f}")
            
            return True
        else:
            print("❌ 演示仿真失败")
            return False
            
    except Exception as e:
        print(f"❌ 仿真运行错误: {e}")
        print(f"错误类型: {type(e).__name__}")
        return False

def run_custom_simulation():
    """运行自定义仿真"""
    try:
        print("\n🎛️ 自定义仿真参数:")
        
        # 获取用户输入
        try:
            num_residents = int(input("居民数量 (默认50): ") or "50")
            num_investors = int(input("投资者数量 (默认12): ") or "12") 
            steps = int(input("仿真步数 (默认30): ") or "30")
        except ValueError:
            print("⚠️ 输入无效，使用默认参数")
            num_residents, num_investors, steps = 50, 12, 30
        
        print(f"\n🏗️ 创建市场: {num_residents}居民, {num_investors}投资者, {steps}步")
        
        import sys; sys.path.append("../src"); from ai_enhanced_housing_model_fixed import run_safe_simulation
        
        model, results = run_safe_simulation(
            steps=steps,
            num_residents=num_residents,
            num_investors=num_investors
        )
        
        if model and results is not None:
            print("\n📊 自定义仿真完成!")
            
            # 保存结果
            results.to_csv('simulation_results.csv', index=False)
            print("💾 结果已保存到: simulation_results.csv")
            
            return True
        else:
            print("❌ 自定义仿真失败")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️ 用户取消操作")
        return False
    except Exception as e:
        print(f"❌ 自定义仿真错误: {e}")
        return False

def visualize_results():
    """可视化结果"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        if not os.path.exists('simulation_results.csv'):
            print("❌ 未找到仿真结果文件")
            return False
        
        print("📈 生成可视化图表...")
        results = pd.read_csv('simulation_results.csv')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('🏠 AI增强房屋市场仿真结果', fontsize=14, fontweight='bold')
        
        # 房价趋势
        axes[0,0].plot(results['Average Property Value'], 'b-', linewidth=2)
        axes[0,0].set_title('房价趋势')
        axes[0,0].set_ylabel('房价 (¥)')
        axes[0,0].grid(True, alpha=0.3)
        
        # 满意度
        axes[0,1].plot(results['Resident Satisfaction'], 'g-', linewidth=2)
        axes[0,1].set_title('居民满意度')
        axes[0,1].set_ylabel('满意度')
        axes[0,1].set_ylim(0, 1)
        axes[0,1].grid(True, alpha=0.3)
        
        # AI预测
        axes[1,0].plot(results['AI Market Prediction'], 'r-', linewidth=2)
        axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1,0].set_title('AI市场预测')
        axes[1,0].set_ylabel('预测趋势')
        axes[1,0].grid(True, alpha=0.3)
        
        # 基尼系数
        axes[1,1].plot(results['Gini Coefficient'], 'orange', linewidth=2)
        axes[1,1].set_title('收入不平等(基尼系数)')
        axes[1,1].set_ylabel('基尼系数')
        axes[1,1].set_ylim(0, 1)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('simulation_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("💾 图表已保存到: simulation_visualization.png")
        return True
        
    except Exception as e:
        print(f"❌ 可视化错误: {e}")
        return False

def main():
    """主程序"""
    print("🏠 AI增强房屋市场仿真系统")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("\n💡 解决依赖问题后重新运行此脚本")
        return
    
    while True:
        print("\n📋 请选择操作:")
        print("1. 🎬 运行演示仿真")
        print("2. 🎛️ 自定义仿真")
        print("3. 📈 可视化结果")
        print("4. 🚪 退出")
        
        try:
            choice = input("\n请输入选择 (1-4): ").strip()
            
            if choice == '1':
                run_demo_simulation()
            elif choice == '2':
                run_custom_simulation()
            elif choice == '3':
                visualize_results()
            elif choice == '4':
                print("👋 感谢使用AI增强房屋市场仿真系统!")
                break
            else:
                print("⚠️ 无效选择，请输入 1-4")
                
        except KeyboardInterrupt:
            print("\n👋 用户退出程序")
            break
        except Exception as e:
            print(f"❌ 程序错误: {e}")

if __name__ == "__main__":
    main() 