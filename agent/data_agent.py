#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Agent for Vision RAG System

模块化的文档解析代理，用于RAG系统的知识库文件解析。
支持自动文件类型识别和统一的解析管理功能。

支持的文件格式：
- PDF文档 (.pdf)
- 图片文件 (.jpg, .jpeg, .png, .bmp, .tiff, .webp)
- Word文档 (.docx, .doc)
- PowerPoint文档 (.pptx, .ppt)
- Excel文档 (.xlsx, .xls)
"""

import sys
import os
import subprocess
import logging
from pathlib import Path
import json
import time
import base64
import asyncio
from typing import Dict, List, Any, Tuple, Optional, Union
from PIL import Image
import io

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.mineru_parser import MineruParser
from utils.apis import Qwen25VL72BInstruct
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class DocumentParseResult:
    """文档解析结果类"""
    
    def __init__(self, success: bool, file_path: str, file_type: str, 
                 content_list: List[Dict] = None, md_content: str = "",
                 output_dir: str = "", execution_time: float = 0.0,
                 error_message: str = ""):
        self.success = success
        self.file_path = file_path
        self.file_type = file_type
        self.content_list = content_list or []
        self.md_content = md_content
        self.output_dir = output_dir
        self.execution_time = execution_time
        self.error_message = error_message
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "success": self.success,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "content_length": len(self.md_content),
            "data_count": len(self.content_list),
            "output_dir": self.output_dir,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "timestamp": self.timestamp
        }
        
    def __str__(self) -> str:
        status = "✅ 成功" if self.success else "❌ 失败"
        return f"{status} - {self.file_type}文档: {Path(self.file_path).name}"


class DataAgent:
    """RAG系统文档解析代理"""
    
    # 支持的文件类型映射
    FILE_TYPE_MAPPING = {
        # PDF文档
        '.pdf': 'pdf',
        
        # 图片文件
        '.jpg': 'image',
        '.jpeg': 'image', 
        '.png': 'image',
        '.bmp': 'image',
        '.tiff': 'image',
        '.tif': 'image',
        '.webp': 'image',
        '.gif': 'image',
        
        # Word文档
        '.docx': 'office',
        '.doc': 'office',
        
        # PowerPoint文档
        '.pptx': 'office',
        '.ppt': 'office',
        
        # Excel文档
        '.xlsx': 'office',
        '.xls': 'office'
    }
    
    def __init__(self, output_dir: str = "output_mineru", cls_dir: str = "default",
                 lang: str = "ch", backend: str = "pipeline", 
                 enable_formula: bool = True, enable_table: bool = True,
                 auto_caption: bool = True, log_level: str = "INFO"):
        """
        初始化数据代理
        
        Args:
            output_dir: 输出根目录名称
            cls_dir: 知识库分类目录名称
            lang: 解析语言 ("ch" 中文, "en" 英文)
            backend: 解析后端 ("pipeline")
            enable_formula: 是否启用公式解析
            enable_table: 是否启用表格解析
            auto_caption: 是否自动生成图片描述 (默认开启)
            log_level: 日志级别
        """
        self.project_root = Path(__file__).parent.parent
        self.output_root = self.project_root / output_dir
        self.cls_dir = cls_dir
        self.lang = lang
        self.backend = backend
        self.enable_formula = enable_formula
        self.enable_table = enable_table
        self.auto_caption = auto_caption
        
        # 确保输出目录存在
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging(log_level)
        
        # 检查依赖
        self.libreoffice_available = self._check_libreoffice()
        
        # 初始化图片描述模型
        self.caption_llm = None
        if self.auto_caption:
            self._initialize_caption_model()
        
        # 解析历史记录
        self.parse_history: List[DocumentParseResult] = []
        
    def _setup_logging(self, log_level: str):
        """设置日志配置"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _check_libreoffice(self) -> bool:
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
                self.logger.info(f"LibreOffice可用: {result.stdout.strip()[:50]}...")
                return True
            else:
                self.logger.warning("LibreOffice不可用")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.warning("LibreOffice未安装或不可用")
            return False
            
    def _initialize_caption_model(self):
        """初始化图片描述模型"""
        try:
            model_config = Qwen25VL72BInstruct()
            self.caption_llm = ChatOpenAI(
                openai_api_base=model_config.api_base,
                openai_api_key=model_config.api_key,
                model_name=model_config.model,
                streaming=False,
                temperature=0.1,
                max_tokens=512,
                extra_body={
                    "vl_high_resolution_images": "True",
                    "top_k": 1,
                }
            )
            self.logger.info("图片描述模型初始化成功")
        except Exception as e:
            self.logger.warning(f"图片描述模型初始化失败: {e}")
            self.auto_caption = False
            self.caption_llm = None
            
    def detect_file_type(self, file_path: Union[str, Path]) -> str:
        """
        根据文件扩展名自动检测文件类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件类型 ('pdf', 'image', 'office', 'unknown')
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        file_type = self.FILE_TYPE_MAPPING.get(suffix, 'unknown')
        self.logger.debug(f"文件 {file_path.name} 检测为类型: {file_type}")
        
        return file_type
        
    def get_output_dir(self, file_path: Union[str, Path]) -> Path:
        """
        获取文件的输出目录路径
        
        Args:
            file_path: 文件路径
            
        Returns:
            输出目录路径: output_dir/cls_dir/file_name_dir
        """
        file_path = Path(file_path)
        file_name_dir = file_path.stem  # 不包含扩展名的文件名
        
        output_dir = self.output_root / self.cls_dir / file_name_dir
        return output_dir
        
    def _image_to_data_url(self, image_path: Union[str, Path]) -> str:
        """将图像路径转换为data URL格式"""
        try:
            with Image.open(image_path) as img:
                # 转换为RGB格式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 调整图像大小以避免过大
                max_size = (1024, 1024)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 转换为base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return f"data:image/jpeg;base64,{img_data}"
        except Exception as e:
            self.logger.error(f"转换图像 {image_path} 时出错: {e}")
            return ""
            
    async def _generate_image_caption(self, image_path: Union[str, Path]) -> str:
        """为图片生成描述"""
        if not self.caption_llm:
            return ""
            
        try:
            data_url = self._image_to_data_url(image_path)
            if not data_url:
                return ""
                
            system_message = SystemMessage(content="你是一个专业的图像分析助手。请仔细观察图像并提供准确、详细的中文描述。描述应该包括图像的主要内容、对象、场景、颜色、布局等关键信息。请保持描述简洁明了，不超过200字。")
            
            human_message = HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {"url": data_url}
                },
                {
                    "type": "text",
                    "text": "请详细描述这张图片的内容。"
                }
            ])
            
            response = await self.caption_llm.ainvoke([system_message, human_message])
            caption = response.content.strip()
            
            self.logger.debug(f"为图片 {Path(image_path).name} 生成描述: {caption[:50]}...")
            return caption
            
        except Exception as e:
            self.logger.error(f"生成图片描述失败 {image_path}: {e}")
            return ""
            
    async def _process_content_list_captions(self, content_list: List[Dict], output_dir: Path):
        """处理content_list中的图片描述"""
        if not self.auto_caption or not self.caption_llm:
            return
            
        # 查找需要生成描述的图片
        images_to_caption = []
        for item in content_list:
            if item.get('type') == 'image':
                # 检查是否已有描述（可能是字符串或数组）
                image_caption = item.get('image_caption')
                has_caption = False
                if isinstance(image_caption, str) and image_caption.strip():
                    has_caption = True
                elif isinstance(image_caption, list) and len(image_caption) > 0:
                    has_caption = True
                    
                if not has_caption:
                    img_path = item.get('img_path')
                    if img_path:
                        # 构建完整的图片路径
                        if not Path(img_path).is_absolute():
                            # 查找图片文件
                            possible_paths = list(output_dir.rglob(Path(img_path).name))
                            if possible_paths:
                                img_path = possible_paths[0]
                            else:
                                img_path = output_dir / img_path
                        if Path(img_path).exists():
                            images_to_caption.append((item, img_path))
                        
        if not images_to_caption:
            return
            
        self.logger.info(f"开始为 {len(images_to_caption)} 张图片生成描述")
        
        # 批量生成描述
        for item, img_path in images_to_caption:
            caption = await self._generate_image_caption(img_path)
            if caption:
                # 如果原来是数组格式，保持数组格式；否则使用字符串
                if isinstance(item.get('image_caption'), list):
                    item['image_caption'] = [caption]
                else:
                    item['image_caption'] = caption
                self.logger.debug(f"已为图片 {Path(img_path).name} 添加描述")
                
        # 更新content_list.json文件
        content_list_files = list(output_dir.rglob("*_content_list.json"))
        if content_list_files:
            content_list_path = content_list_files[0]
            try:
                with open(content_list_path, 'w', encoding='utf-8') as f:
                    json.dump(content_list, f, ensure_ascii=False, indent=2)
                self.logger.info(f"已更新 {content_list_path.name} 文件")
            except Exception as e:
                self.logger.error(f"更新content_list文件失败: {e}")
        else:
            self.logger.warning("未找到content_list.json文件")
        
    def parse_document(self, file_path: Union[str, Path], 
                      method: str = "auto", **kwargs) -> DocumentParseResult:
        """
        解析单个文档文件
        
        Args:
            file_path: 文件路径
            method: 解析方法 ("auto", "txt", "ocr")
            **kwargs: 其他解析参数
            
        Returns:
            DocumentParseResult: 解析结果
        """
        file_path = Path(file_path)
        
        # 检查文件是否存在
        if not file_path.exists():
            error_msg = f"文件不存在: {file_path}"
            self.logger.error(error_msg)
            result = DocumentParseResult(
                success=False,
                file_path=str(file_path),
                file_type="unknown",
                error_message=error_msg
            )
            self.parse_history.append(result)
            return result
            
        # 检测文件类型
        file_type = self.detect_file_type(file_path)
        
        if file_type == 'unknown':
            error_msg = f"不支持的文件类型: {file_path.suffix}"
            self.logger.error(error_msg)
            result = DocumentParseResult(
                success=False,
                file_path=str(file_path),
                file_type=file_type,
                error_message=error_msg
            )
            self.parse_history.append(result)
            return result
            
        # 检查Office文档是否需要LibreOffice
        if file_type == 'office' and not self.libreoffice_available:
            error_msg = "Office文档解析需要LibreOffice但不可用"
            self.logger.error(error_msg)
            result = DocumentParseResult(
                success=False,
                file_path=str(file_path),
                file_type=file_type,
                error_message=error_msg
            )
            self.parse_history.append(result)
            return result
            
        # 获取输出目录
        output_dir = self.get_output_dir(file_path)
        
        # 合并解析参数
        parse_params = {
            "lang": self.lang,
            "backend": self.backend,
            **kwargs
        }
        
        # 根据文件类型调用相应的解析方法
        try:
            start_time = time.time()
            self.logger.info(f"开始解析{file_type}文档: {file_path.name}")
            
            if file_type == 'pdf':
                # PDF解析参数
                pdf_params = {
                    "method": method,
                    "formula": self.enable_formula,
                    "table": self.enable_table,
                    **parse_params
                }
                content_list, md_content = MineruParser.parse_pdf(
                    pdf_path=file_path,
                    output_dir=output_dir,
                    **pdf_params
                )
                
            elif file_type == 'image':
                content_list, md_content = MineruParser.parse_image(
                    image_path=file_path,
                    output_dir=output_dir,
                    **parse_params
                )
                
            elif file_type == 'office':
                content_list, md_content = MineruParser.parse_office_doc(
                    doc_path=file_path,
                    output_dir=output_dir,
                    **parse_params
                )
                
            else:
                raise ValueError(f"未知的文件类型: {file_type}")
                
            execution_time = time.time() - start_time
            
            # 处理图片描述
            if self.auto_caption and content_list:
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._process_content_list_captions(content_list, output_dir))
                    finally:
                        loop.close()
                except Exception as e:
                    self.logger.warning(f"处理图片描述时出错: {e}")
            
            # 创建成功结果
            result = DocumentParseResult(
                success=True,
                file_path=str(file_path),
                file_type=file_type,
                content_list=content_list,
                md_content=md_content,
                output_dir=str(output_dir),
                execution_time=execution_time
            )
            
            self.logger.info(
                f"解析完成: {file_path.name} - "
                f"内容长度: {len(md_content)} 字符, "
                f"结构化数据: {len(content_list)} 项, "
                f"耗时: {execution_time:.2f}秒"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"解析异常: {str(e)}"
            self.logger.error(f"解析失败 {file_path.name}: {error_msg}")
            
            result = DocumentParseResult(
                success=False,
                file_path=str(file_path),
                file_type=file_type,
                output_dir=str(output_dir),
                execution_time=execution_time,
                error_message=error_msg
            )
            
        # 记录到历史
        self.parse_history.append(result)
        return result
        
    def parse_documents(self, file_paths: List[Union[str, Path]], 
                       method: str = "auto", **kwargs) -> List[DocumentParseResult]:
        """
        批量解析多个文档文件
        
        Args:
            file_paths: 文件路径列表
            method: 解析方法
            **kwargs: 其他解析参数
            
        Returns:
            List[DocumentParseResult]: 解析结果列表
        """
        results = []
        
        self.logger.info(f"开始批量解析 {len(file_paths)} 个文件")
        
        for i, file_path in enumerate(file_paths, 1):
            self.logger.info(f"处理文件 {i}/{len(file_paths)}: {Path(file_path).name}")
            result = self.parse_document(file_path, method, **kwargs)
            results.append(result)
            
        # 统计结果
        success_count = sum(1 for r in results if r.success)
        self.logger.info(
            f"批量解析完成: {success_count}/{len(file_paths)} 个文件成功解析"
        )
        
        return results
        
    def parse_directory(self, directory: Union[str, Path], 
                       recursive: bool = True, method: str = "auto", 
                       **kwargs) -> List[DocumentParseResult]:
        """
        解析目录中的所有支持的文档文件
        
        Args:
            directory: 目录路径
            recursive: 是否递归搜索子目录
            method: 解析方法
            **kwargs: 其他解析参数
            
        Returns:
            List[DocumentParseResult]: 解析结果列表
        """
        directory = Path(directory)
        
        if not directory.exists() or not directory.is_dir():
            self.logger.error(f"目录不存在或不是有效目录: {directory}")
            return []
            
        # 搜索支持的文件
        supported_extensions = set(self.FILE_TYPE_MAPPING.keys())
        file_paths = []
        
        if recursive:
            for ext in supported_extensions:
                file_paths.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in supported_extensions:
                file_paths.extend(directory.glob(f"*{ext}"))
                
        self.logger.info(
            f"在目录 {directory} 中找到 {len(file_paths)} 个支持的文件"
        )
        
        return self.parse_documents(file_paths, method, **kwargs)
        
    def get_parse_summary(self) -> Dict[str, Any]:
        """
        获取解析历史统计摘要
        
        Returns:
            Dict: 统计摘要
        """
        total_count = len(self.parse_history)
        success_count = sum(1 for r in self.parse_history if r.success)
        
        # 按文件类型统计
        type_stats = {}
        for result in self.parse_history:
            file_type = result.file_type
            if file_type not in type_stats:
                type_stats[file_type] = {"total": 0, "success": 0}
            type_stats[file_type]["total"] += 1
            if result.success:
                type_stats[file_type]["success"] += 1
                
        return {
            "total_files": total_count,
            "success_files": success_count,
            "failed_files": total_count - success_count,
            "success_rate": (success_count / total_count * 100) if total_count > 0 else 0,
            "type_statistics": type_stats,
            "output_root": str(self.output_root),
            "cls_dir": self.cls_dir
        }
        
    def save_parse_report(self, report_path: Optional[Union[str, Path]] = None) -> Path:
        """
        保存解析报告到JSON文件
        
        Args:
            report_path: 报告文件路径，如果为None则使用默认路径
            
        Returns:
            Path: 报告文件路径
        """
        if report_path is None:
            report_path = self.output_root / self.cls_dir / "parse_report.json"
        else:
            report_path = Path(report_path)
            
        # 确保目录存在
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 生成报告数据
        report_data = {
            "summary": self.get_parse_summary(),
            "parse_results": [result.to_dict() for result in self.parse_history],
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存到文件
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"解析报告已保存到: {report_path}")
        return report_path
        
    def clear_history(self):
        """清空解析历史记录"""
        self.parse_history.clear()
        self.logger.info("解析历史记录已清空")
        
    def get_supported_extensions(self) -> List[str]:
        """获取支持的文件扩展名列表"""
        return list(self.FILE_TYPE_MAPPING.keys())
        
    def is_supported_file(self, file_path: Union[str, Path]) -> bool:
        """
        检查文件是否为支持的格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否支持
        """
        return self.detect_file_type(file_path) != 'unknown'


# 便捷函数
def parse_single_file(file_path: Union[str, Path], 
                     output_dir: str = "output_mineru",
                     cls_dir: str = "default",
                     **kwargs) -> DocumentParseResult:
    """
    便捷函数：解析单个文件
    
    Args:
        file_path: 文件路径
        output_dir: 输出根目录
        cls_dir: 分类目录
        **kwargs: 其他参数
        
    Returns:
        DocumentParseResult: 解析结果
    """
    agent = DataAgent(output_dir=output_dir, cls_dir=cls_dir, **kwargs)
    return agent.parse_document(file_path)


def parse_directory(directory: Union[str, Path],
                   output_dir: str = "output_mineru", 
                   cls_dir: str = "default",
                   recursive: bool = True,
                   **kwargs) -> List[DocumentParseResult]:
    """
    便捷函数：解析目录中的所有文件
    
    Args:
        directory: 目录路径
        output_dir: 输出根目录
        cls_dir: 分类目录
        recursive: 是否递归
        **kwargs: 其他参数
        
    Returns:
        List[DocumentParseResult]: 解析结果列表
    """
    agent = DataAgent(output_dir=output_dir, cls_dir=cls_dir, **kwargs)
    return agent.parse_directory(directory, recursive=recursive, **kwargs)


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description="文档解析代理")
    parser.add_argument("input_path", help="输入文件或目录路径")
    parser.add_argument("--output-dir", default="output_mineru", help="输出根目录")
    parser.add_argument("--cls-dir", default="default", help="分类目录")
    parser.add_argument("--method", default="auto", choices=["auto", "txt", "ocr"], help="解析方法")
    parser.add_argument("--recursive", action="store_true", help="递归处理目录")
    parser.add_argument("--lang", default="ch", choices=["ch", "en"], help="语言")
    
    args = parser.parse_args()
    
    # 创建代理
    agent = DataAgent(
        output_dir=args.output_dir,
        cls_dir=args.cls_dir,
        lang=args.lang
    )
    
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        # 解析单个文件
        result = agent.parse_document(input_path, method=args.method)
        print(result)
    elif input_path.is_dir():
        # 解析目录
        results = agent.parse_directory(input_path, recursive=args.recursive, method=args.method)
        
        # 显示摘要
        summary = agent.get_parse_summary()
        print(f"\n解析完成: {summary['success_files']}/{summary['total_files']} 个文件成功")
        print(f"成功率: {summary['success_rate']:.1f}%")
        
        # 保存报告
        report_path = agent.save_parse_report()
        print(f"详细报告: {report_path}")
    else:
        print(f"错误: 输入路径不存在 - {input_path}")