#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI增强房屋市场仿真系统测试脚本
"""

try:
    print("🔄 开始测试AI增强房屋市场仿真系统...")
    
    # 1. 测试模块导入
    print("1️⃣ 测试模块导入...")
    from ai_enhanced_housing_model import AIEnhancedHousingMarketModel
    print("✅ 模块导入成功!")
    
    # 2. 测试模型创建
    print("2️⃣ 测试模型创建...")
    model = AIEnhancedHousingMarketModel(
        num_residents=50,  # 减少代理数量以加快测试
        num_investors=10,
        width=20,
        height=20
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
    
    print("\n🎉 所有测试通过! 仿真系统运行正常!")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("💡 解决方案: 检查依赖包是否已安装")
    
except Exception as e:
    print(f"❌ 运行错误: {e}")
    print(f"❌ 错误类型: {type(e).__name__}")
    import traceback
    print("📋 详细错误信息:")
    traceback.print_exc() 