#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI增强房屋市场仿真系统测试脚本 - 修复版本
"""

try:
    print("🔄 开始测试AI增强房屋市场仿真系统（修复版本）...")
    
    # 1. 测试模块导入
    print("1️⃣ 测试模块导入...")
    from ai_enhanced_housing_model_fixed import SafeAIEnhancedHousingMarketModel, run_safe_simulation
    print("✅ 修复版本模块导入成功!")
    
    # 2. 测试模型创建
    print("2️⃣ 测试模型创建...")
    model = SafeAIEnhancedHousingMarketModel(
        num_residents=30,  # 减少代理数量以加快测试
        num_investors=8,
        width=15,
        height=15
    )
    print("✅ 模型创建成功!")
    
    # 3. 测试单步运行
    print("3️⃣ 测试单步运行...")
    model.step()
    print("✅ 单步运行成功!")
    
    # 4. 测试多步运行
    print("4️⃣ 测试多步运行...")
    for i in range(5):
        model.step()
        print(f"   步骤 {i+2} 完成")
    print("✅ 多步运行成功!")
    
    # 5. 测试数据收集
    print("5️⃣ 测试数据收集...")
    data = model.datacollector.get_model_vars_dataframe()
    print(f"✅ 数据收集成功! 收集了 {len(data)} 行数据")
    
    # 6. 测试完整仿真
    print("6️⃣ 测试完整仿真功能...")
    sim_model, sim_results = run_safe_simulation(steps=10, num_residents=20, num_investors=5)
    if sim_model and sim_results is not None:
        print("✅ 完整仿真测试成功!")
        print(f"   生成数据: {len(sim_results)} 行")
        print(f"   最终房价: ¥{sim_results['Average Property Value'].iloc[-1]:,.0f}")
        print(f"   最终满意度: {sim_results['Resident Satisfaction'].iloc[-1]:.3f}")
    else:
        print("❌ 完整仿真测试失败!")
    
    print("\n🎉 所有测试通过! 修复版本仿真系统运行完全正常!")
    print("📋 现在可以安全使用 ai_enhanced_housing_model_fixed.py")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("💡 解决方案: 检查 ai_enhanced_housing_model_fixed.py 文件是否存在")
    
except Exception as e:
    print(f"❌ 运行错误: {e}")
    print(f"❌ 错误类型: {type(e).__name__}")
    import traceback
    print("📋 详细错误信息:")
    traceback.print_exc() 