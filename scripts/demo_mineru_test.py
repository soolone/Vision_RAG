#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MinerU Parser Test Demo

演示如何使用 test_mineru.py 脚本测试 MinerU 解析器的各种功能
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            print("✅ 执行成功")
            if result.stdout:
                print("输出:")
                print(result.stdout)
        else:
            print("❌ 执行失败")
            if result.stderr:
                print("错误信息:")
                print(result.stderr)
    except Exception as e:
        print(f"❌ 执行异常: {e}")

def main():
    """主演示函数"""
    print("🚀 MinerU 解析器测试演示")
    print("本演示将展示如何使用 test_mineru.py 脚本测试不同的文档解析功能")
    
    # 切换到项目目录
    project_dir = Path(__file__).parent.parent
    
    # 演示不同的测试选项
    demos = [
        {
            "cmd": ["python", "scripts/test_mineru.py", "--help"],
            "desc": "查看测试脚本帮助信息"
        },
        {
            "cmd": ["python", "scripts/test_mineru.py", "--test", "pdf"],
            "desc": "仅测试PDF文档解析功能"
        },
        {
            "cmd": ["python", "scripts/test_mineru.py", "--test", "image"],
            "desc": "仅测试图片文档解析功能"
        },
        {
            "cmd": ["python", "scripts/test_mineru.py", "--test", "office"],
            "desc": "仅测试Office文档解析功能"
        },
        {
            "cmd": ["python", "scripts/test_mineru.py", "--test", "all"],
            "desc": "运行所有测试"
        }
    ]
    
    print("\n📋 可用的测试选项:")
    for i, demo in enumerate(demos, 1):
        print(f"{i}. {demo['desc']}")
        print(f"   命令: {' '.join(demo['cmd'])}")
    
    print("\n💡 使用建议:")
    print("1. 首先运行 --help 查看所有可用选项")
    print("2. 使用单独的测试选项(pdf, image, office)来测试特定功能")
    print("3. 使用 --test all 运行完整测试套件")
    print("4. 检查 test_output/mineru_test/ 目录查看解析结果")
    print("5. 查看 test_report.json 获取详细的测试报告")
    
    print("\n📁 测试数据位置:")
    data_dir = project_dir / "data" / "doc_data"
    if data_dir.exists():
        print(f"   {data_dir}")
        files = list(data_dir.glob("*"))
        for file in files:
            if file.is_file():
                size = file.stat().st_size
                if size > 1024*1024:
                    size_str = f"{size/(1024*1024):.1f}MB"
                elif size > 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                print(f"   - {file.name} ({size_str})")
    else:
        print(f"   ❌ 测试数据目录不存在: {data_dir}")
    
    print("\n📊 输出目录:")
    output_dir = project_dir / "test_output" / "mineru_test"
    print(f"   {output_dir}")
    if output_dir.exists():
        print("   (目录已存在，包含之前的测试结果)")
    else:
        print("   (目录将在首次运行测试时创建)")
    
    # 询问用户是否要运行演示
    print("\n❓ 是否要运行帮助命令演示? (y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice in ['y', 'yes', '是']:
            run_command(
                ["python", str(project_dir / "scripts" / "test_mineru.py"), "--help"],
                "显示测试脚本帮助信息"
            )
    except KeyboardInterrupt:
        print("\n\n👋 演示结束")
    
    print("\n✨ 演示完成!")
    print("\n🔗 相关文档:")
    print("   - MinerU CLI 工具: https://opendatalab.github.io/MinerU/zh/usage/cli_tools/#_2")
    print("   - 测试脚本: scripts/test_mineru.py")
    print("   - 解析器实现: agent/mineru_parser.py")

if __name__ == "__main__":
    main()