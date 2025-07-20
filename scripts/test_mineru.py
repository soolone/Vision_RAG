#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MinerU Parser Test Script

全面测试 MinerU 文档解析器的核心功能，包括：
- PDF 文档解析 (OminiSVG.pdf)
- 图片文档解析 (car.jpg)
- Word 文档解析 (demo.docx)
- PowerPoint 文档解析 (pcs.pptx)
- Excel 文档解析 (resource.xlsx)
- 不同解析方法和参数的测试
- 全面的文档格式兼容性测试
"""

import sys
import os
import subprocess
from pathlib import Path
import json
import time
from typing import Dict, List, Any, Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.mineru_parser import MineruParser


class MineruTester:
    """MinerU 解析器测试类"""
    
    def __init__(self, data_dir: str = None, output_dir: str = None):
        """
        初始化测试器
        
        Args:
            data_dir: 测试数据目录路径
            output_dir: 输出目录路径
        """
        self.project_root = Path(__file__).parent.parent
        self.data_dir = Path(data_dir) if data_dir else self.project_root / "data" / "doc_data"
        self.output_dir = Path(output_dir) if output_dir else self.project_root / "test_output" / "mineru_test"
        
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
            print(f"   输出文件: {', '.join(output_files)}")
            
    def test_mineru_installation(self):
        """测试MinerU是否正确安装"""
        print("\n=== 测试MinerU安装状态 ===")
        
        try:
            # 测试mineru命令是否可用
            result = subprocess.run(
                ["mineru", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8",
                errors="ignore"
            )
            
            if result.returncode == 0:
                self.log_test_result("MinerU命令可用性", True, "mineru命令正常工作")
                print(f"   MinerU帮助信息预览: {result.stdout[:200]}...")
            else:
                self.log_test_result("MinerU命令可用性", False, f"命令返回错误: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.log_test_result("MinerU命令可用性", False, "命令执行超时")
        except FileNotFoundError:
            self.log_test_result("MinerU命令可用性", False, "mineru命令未找到，请检查安装")
        except Exception as e:
            self.log_test_result("MinerU命令可用性", False, f"异常: {str(e)}")
    
    def test_pdf_parsing(self):
        """测试PDF文档解析"""
        print("\n=== 测试PDF文档解析 ===")
        
        pdf_file = self.data_dir / "OminiSVG.pdf"
        if not pdf_file.exists():
            self.log_test_result("PDF解析测试", False, f"测试文件不存在: {pdf_file}")
            return
            
        # 测试1: 基本PDF解析（只解析第一页以节省时间）
        try:
            start_time = time.time()
            output_dir = self.output_dir / "pdf_basic"
            
            print(f"   开始解析PDF文件: {pdf_file.name} (仅第一页)")
            content_list, md_content = MineruParser.parse_pdf(
                pdf_path=pdf_file,
                output_dir=output_dir,
                method="auto",
                lang="ch",  # 中文优化
                backend="pipeline",
                start_page=0,
                end_page=0,  # 只解析第一页
                formula=False,  # 禁用公式解析以加快速度
                table=False   # 禁用表格解析以加快速度
            )
            
            execution_time = time.time() - start_time
            
            # 检查结果
            success = bool(md_content or content_list)
            details = f"解析内容长度: {len(md_content)} 字符, 结构化数据: {len(content_list)} 项"
            
            output_files = []
            if output_dir.exists():
                output_files = [f.name for f in output_dir.rglob("*") if f.is_file()]
                
            self.log_test_result("PDF基本解析(第一页)", success, details, execution_time, output_files)
            
            # 显示部分解析内容
            if md_content:
                preview = md_content[:200].replace('\n', ' ')
                print(f"   内容预览: {preview}...")
            
        except Exception as e:
            self.log_test_result("PDF基本解析(第一页)", False, f"异常: {str(e)}")
            
    def test_image_parsing(self):
        """测试图片文档解析"""
        print("\n=== 测试图片文档解析 ===")
        
        image_file = self.data_dir / "car.jpg"
        if not image_file.exists():
            self.log_test_result("图片解析测试", False, f"测试文件不存在: {image_file}")
            return
            
        try:
            start_time = time.time()
            output_dir = self.output_dir / "image_basic"
            
            print(f"   开始解析图片文件: {image_file.name}")
            content_list, md_content = MineruParser.parse_image(
                image_path=image_file,
                output_dir=output_dir,
                lang="ch",
                backend="pipeline"
            )
            
            execution_time = time.time() - start_time
            
            success = bool(md_content or content_list)
            details = f"图片解析内容长度: {len(md_content)} 字符, 结构化数据: {len(content_list)} 项"
            
            output_files = []
            if output_dir.exists():
                output_files = [f.name for f in output_dir.rglob("*") if f.is_file()]
                
            self.log_test_result("PNG图片解析", success, details, execution_time, output_files)
            
            # 显示部分解析内容
            if md_content:
                preview = md_content[:200].replace('\n', ' ')
                print(f"   内容预览: {preview}...")
            
        except Exception as e:
            self.log_test_result("PNG图片解析", False, f"异常: {str(e)}")
            
    def test_comprehensive_document_parsing(self):
        """全面测试各种文档格式解析"""
        print("\n=== 全面文档格式解析测试 ===")
        
        # 首先检查LibreOffice是否可用（Office文档需要）
        libreoffice_available = self._check_libreoffice()
        
        # 定义测试文件和对应的解析方法
        test_cases = [
            # PDF测试
            {
                "file": "OminiSVG.pdf",
                "type": "PDF",
                "method": "parse_pdf",
                "params": {
                    "method": "auto",
                    "lang": "ch",
                    "backend": "pipeline",
                    "start_page": 0,
                    "end_page": 2,  # 解析前3页
                    "formula": True,
                    "table": True
                }
            },
            # 图片测试
            {
                "file": "car.jpg",
                "type": "图片",
                "method": "parse_image",
                "params": {
                    "lang": "ch",
                    "backend": "pipeline"
                }
            },
            # Word文档测试
            {
                "file": "demo.docx",
                "type": "Word文档",
                "method": "parse_office_doc",
                "params": {
                    "lang": "ch",
                    "backend": "pipeline"
                },
                "requires_libreoffice": True
            },
            # PowerPoint测试
            {
                "file": "pcs.pptx",
                "type": "PowerPoint文档",
                "method": "parse_office_doc",
                "params": {
                    "lang": "ch",
                    "backend": "pipeline"
                },
                "requires_libreoffice": True
            },
            # Excel测试
            {
                "file": "resource.xlsx",
                "type": "Excel文档",
                "method": "parse_office_doc",
                "params": {
                    "lang": "ch",
                    "backend": "pipeline"
                },
                "requires_libreoffice": True
            }
        ]
        
        for test_case in test_cases:
            self._run_single_document_test(test_case, libreoffice_available)
            
    def _check_libreoffice(self):
        """检查LibreOffice是否可用"""
        try:
            result = subprocess.run(
                ["libreoffice", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                encoding="utf-8",
                errors="ignore"
            )
            if result.returncode == 0:
                print(f"   ✅ LibreOffice可用: {result.stdout.strip()[:50]}...")
                return True
            else:
                print("   ❌ LibreOffice不可用")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("   ❌ LibreOffice未安装或不可用")
            return False
            
    def _run_single_document_test(self, test_case, libreoffice_available):
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
        if test_case.get("requires_libreoffice", False) and not libreoffice_available:
            self.log_test_result(
                f"{test_case['type']}解析测试",
                False,
                "需要LibreOffice但不可用"
            )
            return
            
        try:
            start_time = time.time()
            output_dir = self.output_dir / f"{test_case['type'].lower().replace('文档', '').replace('档', '')}_test"
            
            print(f"\n   📄 开始解析{test_case['type']}: {test_case['file']}")
            
            # 根据方法类型调用相应的解析函数
            if test_case["method"] == "parse_pdf":
                content_list, md_content = MineruParser.parse_pdf(
                    pdf_path=file_path,
                    output_dir=output_dir,
                    **test_case["params"]
                )
            elif test_case["method"] == "parse_image":
                content_list, md_content = MineruParser.parse_image(
                    image_path=file_path,
                    output_dir=output_dir,
                    **test_case["params"]
                )
            elif test_case["method"] == "parse_office_doc":
                content_list, md_content = MineruParser.parse_office_doc(
                    doc_path=file_path,
                    output_dir=output_dir,
                    **test_case["params"]
                )
            else:
                raise ValueError(f"未知的解析方法: {test_case['method']}")
                
            execution_time = time.time() - start_time
            
            # 检查解析结果
            success = bool(md_content or content_list)
            md_length = len(md_content) if md_content else 0
            data_count = len(content_list) if content_list else 0
            
            details = f"内容长度: {md_length} 字符, 结构化数据: {data_count} 项"
            
            # 获取输出文件列表
            output_files = []
            if output_dir.exists():
                output_files = [f.name for f in output_dir.rglob("*") if f.is_file()]
                
            self.log_test_result(
                f"{test_case['type']}解析测试",
                success,
                details,
                execution_time,
                output_files
            )
            
            # 显示部分解析内容预览
            if md_content and len(md_content) > 0:
                preview = md_content[:150].replace('\n', ' ').replace('\r', ' ')
                print(f"   📝 内容预览: {preview}...")
            elif content_list and len(content_list) > 0:
                print(f"   📊 解析到 {len(content_list)} 个数据项")
            else:
                print(f"   ⚠️  未获取到解析内容")
                
        except Exception as e:
            self.log_test_result(
                f"{test_case['type']}解析测试",
                False,
                f"异常: {str(e)}"
            )
            
    def test_parsing_methods(self):
        """测试不同解析方法"""
        print("\n=== 测试不同解析方法 ===")
        
        pdf_file = self.data_dir / "OminiSVG.pdf"
        if not pdf_file.exists():
            self.log_test_result("解析方法测试", False, "PDF测试文件不存在")
            return
            
        methods = ["auto", "txt", "ocr"]
        
        for method in methods:
            try:
                start_time = time.time()
                output_dir = self.output_dir / f"method_{method}"
                
                content_list, md_content = MineruParser.parse_pdf(
                    pdf_path=pdf_file,
                    output_dir=output_dir,
                    method=method,
                    lang="ch",
                    backend="pipeline",
                    start_page=0,
                    end_page=1,  # 只解析前2页以节省时间
                    formula=True,
                    table=True
                )
                
                execution_time = time.time() - start_time
                
                success = bool(md_content or content_list)
                details = f"方法{method}: 内容长度{len(md_content)}字符, 数据{len(content_list)}项"
                
                self.log_test_result(f"解析方法-{method}", success, details, execution_time)
                
            except Exception as e:
                self.log_test_result(f"解析方法-{method}", False, f"异常: {str(e)}")
                
    def test_parsing_options(self):
        """测试解析选项"""
        print("\n=== 测试解析选项 ===")
        
        pdf_file = self.data_dir / "OminiSVG.pdf"
        if not pdf_file.exists():
            self.log_test_result("解析选项测试", False, "PDF测试文件不存在")
            return
            
        # 测试禁用公式解析
        try:
            start_time = time.time()
            output_dir = self.output_dir / "no_formula"
            
            content_list, md_content = MineruParser.parse_pdf(
                pdf_path=pdf_file,
                output_dir=output_dir,
                method="auto",
                lang="ch",
                backend="pipeline",
                start_page=0,
                end_page=1,
                formula=False,  # 禁用公式解析
                table=True
            )
            
            execution_time = time.time() - start_time
            
            success = bool(md_content or content_list)
            details = f"禁用公式: 内容长度{len(md_content)}字符, 数据{len(content_list)}项"
            
            self.log_test_result("禁用公式解析", success, details, execution_time)
            
        except Exception as e:
            self.log_test_result("禁用公式解析", False, f"异常: {str(e)}")
            
        # 测试禁用表格解析
        try:
            start_time = time.time()
            output_dir = self.output_dir / "no_table"
            
            content_list, md_content = MineruParser.parse_pdf(
                pdf_path=pdf_file,
                output_dir=output_dir,
                method="auto",
                lang="ch",
                backend="pipeline",
                start_page=0,
                end_page=1,
                formula=True,
                table=False  # 禁用表格解析
            )
            
            execution_time = time.time() - start_time
            
            success = bool(md_content or content_list)
            details = f"禁用表格: 内容长度{len(md_content)}字符, 数据{len(content_list)}项"
            
            self.log_test_result("禁用表格解析", success, details, execution_time)
            
        except Exception as e:
            self.log_test_result("禁用表格解析", False, f"异常: {str(e)}")
            
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始MinerU解析器全面测试")
        print(f"📁 测试数据目录: {self.data_dir}")
        print(f"📁 输出目录: {self.output_dir}")
        
        # 检查测试数据目录
        if not self.data_dir.exists():
            print(f"❌ 测试数据目录不存在: {self.data_dir}")
            return
            
        # 首先测试MinerU安装状态
        self.test_mineru_installation()
        
        # 运行全面文档格式测试
        self.test_comprehensive_document_parsing()
        
        # 运行解析方法和选项测试
        self.test_parsing_methods()
        self.test_parsing_options()
        
        # 生成测试报告
        self.generate_report()
        
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
        for item in sorted(self.output_dir.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(self.output_dir)
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
    import argparse
    
    parser = argparse.ArgumentParser(description="MinerU解析器测试脚本")
    parser.add_argument("--data-dir", help="测试数据目录路径")
    parser.add_argument("--output-dir", help="输出目录路径")
    parser.add_argument("--test", choices=["pdf", "image", "office", "comprehensive", "methods", "options", "all"], 
                       default="all", help="指定要运行的测试类型")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = MineruTester(data_dir=args.data_dir, output_dir=args.output_dir)
    
    # 运行指定的测试
    if args.test == "pdf":
        tester.test_pdf_parsing()
    elif args.test == "image":
        tester.test_image_parsing()
    elif args.test == "office":
        # 为了向后兼容，保留原有的office测试，但使用新的全面测试方法
        tester.test_comprehensive_document_parsing()
    elif args.test == "comprehensive":
        tester.test_comprehensive_document_parsing()
    elif args.test == "methods":
        tester.test_parsing_methods()
    elif args.test == "options":
        tester.test_parsing_options()
    else:
        tester.run_all_tests()
    
    # 如果不是运行所有测试，也生成报告
    if args.test != "all":
        tester.generate_report()


if __name__ == "__main__":
    main()