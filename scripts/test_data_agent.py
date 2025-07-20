#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试DataAgent类的功能
参考test_mineru.py的测试内容，对doc_data目录中的各种文档格式进行全面测试
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.data_agent import DataAgent, parse_single_file, DocumentParseResult

class DataAgentTester:
    """DataAgent测试类"""
    
    def __init__(self, data_dir: str = None, output_dir: str = None):
        """
        初始化测试器
        
        Args:
            data_dir: 测试数据目录路径
            output_dir: 输出目录路径
        """
        self.project_root = Path(__file__).parent.parent
        self.data_dir = Path(data_dir) if data_dir else self.project_root / "data" / "doc_data"
        self.output_dir = Path(output_dir) if output_dir else self.project_root / "test_output" / "first_output"
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 测试结果记录
        self.test_results = []
        
    def log_test_result(self, test_name: str, success: bool, details: str = "", 
                       execution_time: float = 0.0, output_files: List[str] = None):
        """记录测试结果"""
        result = {
            "test_name": test_name,
            "success": success,
            "details": details,
            "execution_time": execution_time,
            "output_files": output_files or [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"\n{status} {test_name}")
        if details:
            print(f"   详情: {details}")
        if execution_time > 0:
            print(f"   耗时: {execution_time:.2f}秒")
        if output_files:
            print(f"   输出文件: {', '.join(output_files[:5])}{'...' if len(output_files) > 5 else ''}")
            
    def test_data_agent_basic(self):
        """测试DataAgent基本功能"""
        print("\n=== DataAgent 基本功能测试 ===")
        
        # 创建DataAgent实例
        agent = DataAgent(
            output_dir="test_output",
            cls_dir="first_output",
            lang="ch"
        )
        
        print(f"输出根目录: {agent.output_root}")
        print(f"分类目录: {agent.cls_dir}")
        print(f"LibreOffice可用: {agent.libreoffice_available}")
        print(f"支持的文件扩展名: {agent.get_supported_extensions()}")
        
        # 测试文件类型检测
        test_files = [
            "test.pdf",
            "image.jpg", 
            "document.docx",
            "presentation.pptx",
            "spreadsheet.xlsx",
            "unknown.txt"
        ]
        
        print("\n=== 文件类型检测测试 ===")
        for file_path in test_files:
            file_type = agent.detect_file_type(file_path)
            is_supported = agent.is_supported_file(file_path)
            print(f"{file_path:20} -> {file_type:8} (支持: {is_supported})")
        
        # 测试输出目录生成
        print("\n=== 输出目录路径测试 ===")
        for file_path in test_files[:3]:
            output_dir = agent.get_output_dir(file_path)
            print(f"{file_path:20} -> {output_dir}")
            
        self.log_test_result("DataAgent基本功能", True, "文件类型检测和目录生成正常")

    def test_comprehensive_document_parsing(self):
        """全面测试各种文档格式解析"""
        print("\n=== 全面文档格式解析测试 ===")
        
        # 创建DataAgent实例，指定输出到test_output/first_output
        agent = DataAgent(
            output_dir="test_output",
            cls_dir="first_output",
            lang="ch",
            enable_formula=True,
            enable_table=True
        )
        
        print(f"📁 测试数据目录: {self.data_dir}")
        print(f"📁 输出目录: {agent.output_root / agent.cls_dir}")
        print(f"LibreOffice可用: {agent.libreoffice_available}")
        
        # 检查测试数据目录
        if not self.data_dir.exists():
            self.log_test_result("文档解析测试", False, f"测试数据目录不存在: {self.data_dir}")
            return
            
        # 定义测试文件和对应的类型
        test_cases = [
            {"file": "OminiSVG.pdf", "type": "PDF文档"},
            {"file": "car.jpg", "type": "图片文档"},
            {"file": "demo.docx", "type": "Word文档", "requires_libreoffice": True},
            {"file": "pcs.pptx", "type": "PowerPoint文档", "requires_libreoffice": True},
            {"file": "resource.xlsx", "type": "Excel文档", "requires_libreoffice": True}
        ]
        
        for test_case in test_cases:
            self._run_single_document_test(agent, test_case)
            
        # 显示解析历史摘要
        summary = agent.get_parse_summary()
        print("\n=== 解析历史摘要 ===")
        print(f"总文件数: {summary['total_files']}")
        print(f"成功文件数: {summary['success_files']}")
        print(f"失败文件数: {summary['failed_files']}")
        print(f"成功率: {summary['success_rate']:.1f}%")
        
        if summary['type_statistics']:
            print("\n按类型统计:")
            for file_type, stats in summary['type_statistics'].items():
                print(f"  {file_type}: {stats['success']}/{stats['total']} 成功")
                
        # 保存解析报告
        report_path = agent.save_parse_report()
        print(f"\n📄 详细报告已保存到: {report_path}")
        
        return agent
        
    def _run_single_document_test(self, agent: DataAgent, test_case: Dict[str, Any]):
        """运行单个文档测试"""
        file_path = self.data_dir / test_case["file"]
        
        # 检查文件是否存在
        if not file_path.exists():
            self.log_test_result(
                f"{test_case['type']}解析测试",
                False,
                f"测试文件不存在: {test_case['file']}"
            )
            return
            
        # 检查是否需要LibreOffice
        if test_case.get("requires_libreoffice", False) and not agent.libreoffice_available:
            self.log_test_result(
                f"{test_case['type']}解析测试",
                False,
                "需要LibreOffice但不可用"
            )
            return
            
        try:
            print(f"\n   📄 开始解析{test_case['type']}: {test_case['file']}")
            
            # 使用DataAgent解析文档
            result = agent.parse_document(file_path, method="auto")
            
            if result.success:
                md_length = len(result.md_content)
                data_count = len(result.content_list)
                
                details = f"内容长度: {md_length} 字符, 结构化数据: {data_count} 项"
                
                # 获取输出文件列表
                output_files = []
                output_dir = Path(result.output_dir)
                if output_dir.exists():
                    output_files = [f.name for f in output_dir.rglob("*") if f.is_file()]
                    
                self.log_test_result(
                    f"{test_case['type']}解析测试",
                    True,
                    details,
                    result.execution_time,
                    output_files
                )
                
                # 显示部分解析内容预览
                if result.md_content and len(result.md_content) > 0:
                    preview = result.md_content[:150].replace('\n', ' ').replace('\r', ' ')
                    print(f"   📝 内容预览: {preview}...")
                elif result.content_list and len(result.content_list) > 0:
                    print(f"   📊 解析到 {len(result.content_list)} 个数据项")
                else:
                    print(f"   ⚠️  未获取到解析内容")
            else:
                self.log_test_result(
                    f"{test_case['type']}解析测试",
                    False,
                    f"解析失败: {result.error_message}",
                    result.execution_time
                )
                
        except Exception as e:
            self.log_test_result(
                f"{test_case['type']}解析测试",
                False,
                f"异常: {str(e)}"
            )
            
    def test_batch_parsing(self):
        """测试批量解析功能"""
        print("\n=== 批量解析测试 ===")
        
        # 创建DataAgent实例
        agent = DataAgent(
            output_dir="test_output",
            cls_dir="first_output",
            lang="ch"
        )
        
        if not self.data_dir.exists():
            self.log_test_result("批量解析测试", False, "测试数据目录不存在")
            return
            
        # 获取所有支持的文件
        supported_files = []
        for file_path in self.data_dir.iterdir():
            if file_path.is_file() and agent.is_supported_file(file_path):
                supported_files.append(file_path)
                
        if not supported_files:
            self.log_test_result("批量解析测试", False, "未找到支持的文件")
            return
            
        print(f"找到 {len(supported_files)} 个支持的文件")
        
        try:
            start_time = time.time()
            
            # 使用parse_directory方法批量解析
            results = agent.parse_directory(self.data_dir, recursive=False)
            
            execution_time = time.time() - start_time
            
            success_count = sum(1 for r in results if r.success)
            details = f"批量解析 {len(results)} 个文件，成功 {success_count} 个"
            
            self.log_test_result(
                "批量解析测试",
                success_count > 0,
                details,
                execution_time
            )
            
        except Exception as e:
            self.log_test_result("批量解析测试", False, f"异常: {str(e)}")
            
    def test_convenience_functions(self):
        """测试便捷函数"""
        print("\n=== 便捷函数测试 ===")
        
        if not self.data_dir.exists():
            self.log_test_result("便捷函数测试", False, "测试数据目录不存在")
            return
            
        # 查找第一个PDF文件进行测试
        for file_path in self.data_dir.glob("*.pdf"):
            print(f"使用便捷函数解析: {file_path.name}")
            
            try:
                start_time = time.time()
                
                result = parse_single_file(
                    file_path,
                    output_dir="test_output",
                    cls_dir="first_output"
                )
                
                execution_time = time.time() - start_time
                
                if result.success:
                    details = f"便捷函数解析成功，内容长度: {len(result.md_content)} 字符"
                    self.log_test_result("便捷函数测试", True, details, execution_time)
                else:
                    self.log_test_result("便捷函数测试", False, f"解析失败: {result.error_message}")
                    
            except Exception as e:
                self.log_test_result("便捷函数测试", False, f"异常: {str(e)}")
                
            break
        else:
            self.log_test_result("便捷函数测试", False, "未找到PDF文件进行测试")
            
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始DataAgent全面测试")
        print(f"📁 测试数据目录: {self.data_dir}")
        print(f"📁 输出目录: {self.output_dir}")
        
        # 检查测试数据目录
        if not self.data_dir.exists():
            print(f"❌ 测试数据目录不存在: {self.data_dir}")
            return
            
        # 运行各项测试
        self.test_data_agent_basic()
        agent = self.test_comprehensive_document_parsing()
        self.test_batch_parsing()
        self.test_convenience_functions()
        
        # 生成测试报告
        self.generate_report()
        
        return agent
        
    def generate_report(self):
        """生成测试报告"""
        print("\n" + "="*60)
        print("📊 测试报告")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"\n总测试数: {total_tests}")
        print(f"通过: {passed_tests} ✅")
        print(f"失败: {failed_tests} ❌")
        print(f"成功率: {(passed_tests/total_tests*100):.1f}%")
        
        if failed_tests > 0:
            print("\n❌ 失败的测试:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"   - {result['test_name']}: {result['details']}")
                    
        # 保存详细报告到JSON文件
        report_file = self.output_dir / "test_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "success_rate": passed_tests/total_tests*100
                },
                "test_results": self.test_results
            }, f, ensure_ascii=False, indent=2)
            
        print(f"\n📄 详细报告已保存到: {report_file}")
        
        # 显示输出目录结构
        print(f"\n📁 输出目录结构:")
        output_root = self.project_root / "test_output" / "first_output"
        if output_root.exists():
            for item in sorted(output_root.rglob("*")):
                if item.is_file():
                    rel_path = item.relative_to(output_root)
                    size = item.stat().st_size
                    if size > 1024*1024:
                        size_str = f"{size/(1024*1024):.1f}MB"
                    elif size > 1024:
                        size_str = f"{size/1024:.1f}KB"
                    else:
                        size_str = f"{size}B"
                    print(f"   📄 {rel_path} ({size_str})")

def main():
    """主函数"""
    # 创建测试器实例
    tester = DataAgentTester(
        data_dir="data/doc_data",
        output_dir="test_output/first_output"
    )
    
    # 运行所有测试
    agent = tester.run_all_tests()
    
    return agent

if __name__ == "__main__":
    main()