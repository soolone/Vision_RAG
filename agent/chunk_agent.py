#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态RAG系统的文档切片代理

该模块实现了对MinerU解析结果的智能切片功能，支持文本、图片、表格等多种数据类型的切片策略。
切片结果将用于后续的向量化、存储与检索。

主要功能：
1. 解析MinerU输出的结构化数据
2. 根据不同数据类型应用相应的切片策略
3. 保持上下文连贯性和语义完整性
4. 支持多种切片模式（固定大小、语义边界、文档结构等）

Author: Vision_RAG Team
Date: 2024-12-19
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """切片类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    MIXED = "mixed"  # 混合类型切片


class ChunkStrategy(Enum):
    """切片策略枚举"""
    FIXED_SIZE = "fixed_size"  # 固定大小切片
    SEMANTIC = "semantic"  # 语义边界切片
    SENTENCE = "sentence"  # 句子边界切片
    PARAGRAPH = "paragraph"  # 段落边界切片
    DOCUMENT_STRUCTURE = "document_structure"  # 文档结构切片
    HYBRID = "hybrid"  # 混合策略


@dataclass
class ChunkConfig:
    """切片配置类"""
    # 基本配置
    max_chunk_size: int = 1000  # 最大切片大小（字符数）
    min_chunk_size: int = 100   # 最小切片大小（字符数）
    overlap_size: int = 100     # 重叠大小（字符数）
    
    # 策略配置
    text_strategy: ChunkStrategy = ChunkStrategy.SEMANTIC
    table_strategy: ChunkStrategy = ChunkStrategy.DOCUMENT_STRUCTURE
    image_strategy: ChunkStrategy = ChunkStrategy.DOCUMENT_STRUCTURE
    
    # 文本切片特定配置
    preserve_sentences: bool = True  # 保持句子完整性
    preserve_paragraphs: bool = True  # 保持段落完整性
    text_level_aware: bool = True    # 考虑文本层级（标题、正文等）
    
    # 表格切片特定配置
    table_as_single_chunk: bool = True  # 表格作为单个切片
    include_table_caption: bool = True  # 包含表格标题
    include_table_context: bool = True  # 包含表格上下文
    
    # 图片切片特定配置
    image_as_single_chunk: bool = True  # 图片作为单个切片
    include_image_caption: bool = True  # 包含图片标题
    include_image_context: bool = True  # 包含图片上下文
    
    # 元数据配置
    include_page_info: bool = True      # 包含页面信息
    include_position_info: bool = True  # 包含位置信息
    include_source_info: bool = True    # 包含源文件信息


@dataclass
class Chunk:
    """切片数据类"""
    chunk_id: str
    chunk_type: ChunkType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 位置信息
    page_idx: Optional[int] = None
    chunk_idx: int = 0
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    
    # 关联信息
    source_file: Optional[str] = None
    parent_document: Optional[str] = None
    related_chunks: List[str] = field(default_factory=list)
    
    # 特殊内容路径（图片、表格等）
    content_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'chunk_id': self.chunk_id,
            'chunk_type': self.chunk_type.value,
            'content': self.content,
            'metadata': self.metadata,
            'page_idx': self.page_idx,
            'chunk_idx': self.chunk_idx,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'source_file': self.source_file,
            'parent_document': self.parent_document,
            'related_chunks': self.related_chunks,
            'content_path': self.content_path
        }


class ChunkAgent:
    """文档切片代理类
    
    负责对MinerU解析的文档结果进行智能切片，支持多种数据类型和切片策略。
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """初始化切片代理
        
        Args:
            config: 切片配置，如果为None则使用默认配置
        """
        self.config = config or ChunkConfig()
        self.chunks: List[Chunk] = []
        self.chunk_counter = 0
        
        # 句子分割正则表达式 <mcreference link="https://www.multimodal.dev/post/how-to-chunk-documents-for-rag" index="1">1</mcreference>
        self.sentence_pattern = re.compile(r'[.!?。！？]+\s*')
        # 段落分割正则表达式
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        
        logger.info(f"ChunkAgent initialized with config: {self.config}")
    
    def chunk_document(self, 
                      content_list_path: str, 
                      output_dir: str,
                      document_name: Optional[str] = None) -> List[Chunk]:
        """对文档进行切片处理
        
        Args:
            content_list_path: MinerU输出的content_list.json文件路径
            output_dir: 输出目录路径
            document_name: 文档名称，用于生成chunk_id
            
        Returns:
            切片结果列表
        """
        try:
            # 读取MinerU解析结果
            with open(content_list_path, 'r', encoding='utf-8') as f:
                content_list = json.load(f)
            
            if document_name is None:
                document_name = Path(content_list_path).parent.name
            
            logger.info(f"开始处理文档: {document_name}, 共{len(content_list)}个元素")
            
            # 重置切片计数器
            self.chunk_counter = 0
            self.chunks = []
            
            # 根据数据类型分组处理 <mcreference link="https://unstructured.io/blog/chunking-for-rag-best-practices" index="2">2</mcreference>
            grouped_content = self._group_content_by_type(content_list)
            
            # 处理每个内容组
            for group in grouped_content:
                chunks = self._process_content_group(group, document_name, output_dir)
                self.chunks.extend(chunks)
            
            # 后处理：添加关联信息和优化切片
            self._post_process_chunks()
            
            logger.info(f"文档切片完成，共生成{len(self.chunks)}个切片")
            return self.chunks
            
        except Exception as e:
            logger.error(f"文档切片处理失败: {str(e)}")
            raise
    
    def _group_content_by_type(self, content_list: List[Dict]) -> List[List[Dict]]:
        """根据内容类型对数据进行分组
        
        将连续的同类型内容分组，以便应用相应的切片策略。
        
        Args:
            content_list: MinerU解析的内容列表
            
        Returns:
            分组后的内容列表
        """
        if not content_list:
            return []
        
        groups = []
        current_group = [content_list[0]]
        current_type = content_list[0].get('type')
        
        for item in content_list[1:]:
            item_type = item.get('type')
            
            # 如果类型相同且都是文本，继续添加到当前组
            if (item_type == current_type and item_type == 'text' and 
                len(current_group) < 10):  # 限制文本组大小，避免过大
                current_group.append(item)
            else:
                # 开始新组
                groups.append(current_group)
                current_group = [item]
                current_type = item_type
        
        # 添加最后一组
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _process_content_group(self, 
                              group: List[Dict], 
                              document_name: str,
                              output_dir: str) -> List[Chunk]:
        """处理内容组
        
        Args:
            group: 内容组
            document_name: 文档名称
            output_dir: 输出目录
            
        Returns:
            切片列表
        """
        if not group:
            return []
        
        group_type = group[0].get('type')
        
        if group_type == 'text':
            return self._chunk_text_group(group, document_name)
        elif group_type == 'image':
            return self._chunk_image_group(group, document_name, output_dir)
        elif group_type == 'table':
            return self._chunk_table_group(group, document_name, output_dir)
        elif group_type == 'equation':
            return self._chunk_equation_group(group, document_name, output_dir)
        else:
            logger.warning(f"未知内容类型: {group_type}")
            return []
    
    def _chunk_text_group(self, text_group: List[Dict], document_name: str) -> List[Chunk]:
        """对文本组进行切片 <mcreference link="https://medium.com/@anuragmishra_27746/five-levels-of-chunking-strategies-in-rag-notes-from-gregs-video-7b735895694d" index="3">3</mcreference>
        
        Args:
            text_group: 文本内容组
            document_name: 文档名称
            
        Returns:
            文本切片列表
        """
        chunks = []
        
        if self.config.text_strategy == ChunkStrategy.DOCUMENT_STRUCTURE:
            # 基于文档结构的切片：考虑文本层级
            chunks = self._chunk_by_document_structure(text_group, document_name)
        elif self.config.text_strategy == ChunkStrategy.SEMANTIC:
            # 语义切片：基于段落和句子边界 <mcreference link="https://www.sagacify.com/news/a-guide-to-chunking-strategies-for-retrieval-augmented-generation-rag" index="4">4</mcreference>
            chunks = self._chunk_by_semantic_boundaries(text_group, document_name)
        elif self.config.text_strategy == ChunkStrategy.FIXED_SIZE:
            # 固定大小切片
            chunks = self._chunk_by_fixed_size(text_group, document_name)
        else:
            # 默认使用语义切片
            chunks = self._chunk_by_semantic_boundaries(text_group, document_name)
        
        return chunks
    
    def _chunk_by_document_structure(self, text_group: List[Dict], document_name: str) -> List[Chunk]:
        """基于文档结构进行切片
        
        考虑文本层级（标题、正文等），保持文档结构的完整性。
        """
        chunks = []
        current_chunk_content = []
        current_chunk_size = 0
        current_page = None
        
        for item in text_group:
            text = item.get('text', '').strip()
            if not text:
                continue
            
            text_level = item.get('text_level', 0)
            page_idx = item.get('page_idx')
            
            # 如果是标题（text_level > 0）且当前切片不为空，结束当前切片
            if (text_level > 0 and current_chunk_content and 
                current_chunk_size > self.config.min_chunk_size):
                
                chunk = self._create_text_chunk(
                    current_chunk_content, document_name, current_page
                )
                chunks.append(chunk)
                current_chunk_content = []
                current_chunk_size = 0
            
            # 添加到当前切片
            current_chunk_content.append(item)
            current_chunk_size += len(text)
            current_page = page_idx
            
            # 如果切片大小超过限制，结束当前切片
            if current_chunk_size >= self.config.max_chunk_size:
                chunk = self._create_text_chunk(
                    current_chunk_content, document_name, current_page
                )
                chunks.append(chunk)
                current_chunk_content = []
                current_chunk_size = 0
        
        # 处理最后一个切片
        if current_chunk_content:
            chunk = self._create_text_chunk(
                current_chunk_content, document_name, current_page
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_semantic_boundaries(self, text_group: List[Dict], document_name: str) -> List[Chunk]:
        """基于语义边界进行切片 <mcreference link="https://www.analyticsvidhya.com/blog/2024/10/chunking-techniques-to-build-exceptional-rag-systems/" index="5">5</mcreference>
        
        在句子和段落边界处切分，保持语义完整性。
        """
        chunks = []
        
        # 合并所有文本
        full_text = ""
        text_items = []
        for item in text_group:
            text = item.get('text', '').strip()
            if text:
                full_text += text + " "
                text_items.append(item)
        
        if not full_text.strip():
            return chunks
        
        # 按段落分割
        paragraphs = self.paragraph_pattern.split(full_text)
        
        current_chunk = ""
        current_items = []
        current_page = text_items[0].get('page_idx') if text_items else None
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 如果添加当前段落会超过大小限制
            if (len(current_chunk) + len(paragraph) > self.config.max_chunk_size and 
                len(current_chunk) > self.config.min_chunk_size):
                
                # 创建当前切片
                if current_chunk.strip():
                    chunk = Chunk(
                    chunk_id=f"{document_name}_chunk_{self.chunk_counter}",
                    chunk_type=ChunkType.TEXT,
                    content=current_chunk.strip(),
                    page_idx=current_page,
                    chunk_idx=self.chunk_counter,
                    source_file=document_name,
                    parent_document=document_name,
                    metadata={
                        'chunk_strategy': 'semantic_boundaries',
                        'text_items_count': len(current_items),
                        'created_at': datetime.now().isoformat()
                    }
                )
                    chunks.append(chunk)
                    self.chunk_counter += 1
                
                # 开始新切片
                current_chunk = paragraph
                current_items = []
            else:
                # 添加到当前切片
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # 处理最后一个切片
        if current_chunk.strip():
            chunk = Chunk(
            chunk_id=f"{document_name}_chunk_{self.chunk_counter}",
            chunk_type=ChunkType.TEXT,
            content=current_chunk.strip(),
            page_idx=current_page,
            chunk_idx=self.chunk_counter,
            source_file=document_name,
            parent_document=document_name,
            metadata={
                'chunk_strategy': 'semantic_boundaries',
                'text_items_count': len(current_items),
                'created_at': datetime.now().isoformat()
            }
        )
            chunks.append(chunk)
            self.chunk_counter += 1
        
        return chunks
    
    def _chunk_by_fixed_size(self, text_group: List[Dict], document_name: str) -> List[Chunk]:
        """基于固定大小进行切片
        
        按照固定字符数进行切分，支持重叠。
        """
        chunks = []
        
        # 合并所有文本
        full_text = ""
        for item in text_group:
            text = item.get('text', '').strip()
            if text:
                full_text += text + " "
        
        if not full_text.strip():
            return chunks
        
        # 固定大小切片
        start = 0
        while start < len(full_text):
            end = start + self.config.max_chunk_size
            chunk_text = full_text[start:end]
            
            # 如果不是最后一个切片，尝试在句子边界处结束
            if end < len(full_text) and self.config.preserve_sentences:
                # 寻找最后一个句子结束符
                last_sentence_end = -1
                for match in self.sentence_pattern.finditer(chunk_text):
                    last_sentence_end = match.end()
                
                if last_sentence_end > self.config.min_chunk_size:
                    chunk_text = chunk_text[:last_sentence_end]
                    end = start + last_sentence_end
            
            if chunk_text.strip():
                chunk = Chunk(
                    chunk_id=f"{document_name}_chunk_{self.chunk_counter}",
                    chunk_type=ChunkType.TEXT,
                    content=chunk_text.strip(),
                    chunk_idx=self.chunk_counter,
                    start_pos=start,
                    end_pos=end,
                    source_file=document_name,
                    parent_document=document_name,
                    metadata={
                        'chunk_strategy': 'fixed_size',
                        'created_at': datetime.now().isoformat()
                    }
                )
                chunks.append(chunk)
                self.chunk_counter += 1
            
            # 移动到下一个位置，考虑重叠
            start = end - self.config.overlap_size
            if start <= 0:
                start = end
        
        return chunks
    
    def _create_text_chunk(self, text_items: List[Dict], document_name: str, page_idx: Optional[int]) -> Chunk:
        """创建文本切片"""
        content = "\n".join([item.get('text', '').strip() for item in text_items if item.get('text', '').strip()])
        
        # 提取元数据
        metadata = {
            'chunk_strategy': 'document_structure',
            'text_items_count': len(text_items),
            'has_title': any(item.get('text_level', 0) > 0 for item in text_items),
            'created_at': datetime.now().isoformat()
        }
        
        # 添加文本层级信息
        text_levels = [item.get('text_level', 0) for item in text_items]
        if text_levels:
            metadata['text_levels'] = text_levels
            metadata['max_text_level'] = max(text_levels)
        
        chunk = Chunk(
            chunk_id=f"{document_name}_chunk_{self.chunk_counter}",
            chunk_type=ChunkType.TEXT,
            content=content,
            page_idx=page_idx,
            chunk_idx=self.chunk_counter,
            source_file=document_name,
            parent_document=document_name,
            metadata=metadata
        )
        
        self.chunk_counter += 1
        return chunk
    
    def _chunk_image_group(self, image_group: List[Dict], document_name: str, output_dir: str) -> List[Chunk]:
        """对图片组进行切片
        
        图片通常作为单独的切片处理，包含图片路径和相关描述信息。
        对于图片和表格类切片，需要执行双入库操作：
        1. 以图像作为embedding输入，metadata保存图像描述和Base64参数
        2. 以image_caption作为embedding输入，metadata保存图像描述和Base64参数
        """
        chunks = []
        
        for item in image_group:
            img_path = item.get('img_path')
            if not img_path:
                continue
            
            # 构建完整的图片路径
            full_img_path = os.path.join(output_dir, img_path)
            
            # 获取图片标题，确保是字符串类型
            caption_raw = item.get('image_caption', '')
            if isinstance(caption_raw, list):
                caption = '; '.join(caption_raw) if caption_raw else ''
            else:
                caption = caption_raw or ''
            
            # 构建内容描述
            content_parts = []
            
            # 添加图片标题
            if self.config.include_image_caption and caption:
                content_parts.append(f"图片标题: {caption}")
            
            # 添加图片描述
            content_parts.append(f"图片路径: {img_path}")
            
            # 注意：当前 content_list.json 中的图片条目不包含 text 字段
            # OCR 文本实际存储在 model.json 文件中，但当前实现未读取该文件
            # 如果需要 OCR 文本，需要额外读取 model.json 文件
            
            content = "\n".join(content_parts)
            
            # 第一个切片：以图像作为embedding输入
            image_chunk = Chunk(
                chunk_id=f"{document_name}_img_{self.chunk_counter}",
                chunk_type=ChunkType.IMAGE,
                content=content,
                content_path=full_img_path,
                page_idx=item.get('page_idx'),
                chunk_idx=self.chunk_counter,
                source_file=document_name,
                parent_document=document_name,
                metadata={
                    'chunk_strategy': 'image_visual',
                    'image_path': img_path,
                    'has_caption': bool(caption),
                    'has_ocr_text': False,
                    'embedding_type': 'visual',  # 标记为视觉embedding
                    'image_caption': caption,
                    'created_at': datetime.now().isoformat()
                }
            )
            
            chunks.append(image_chunk)
            self.chunk_counter += 1
            
            # 第二个切片：以image_caption作为embedding输入（如果有标题）
            if caption:
                caption_chunk = Chunk(
                    chunk_id=f"{document_name}_img_caption_{self.chunk_counter}",
                    chunk_type=ChunkType.IMAGE,
                    content=caption,  # 直接使用标题作为内容
                    content_path=full_img_path,
                    page_idx=item.get('page_idx'),
                    chunk_idx=self.chunk_counter,
                    source_file=document_name,
                    parent_document=document_name,
                    metadata={
                        'chunk_strategy': 'image_caption',
                        'image_path': img_path,
                        'has_caption': True,
                        'has_ocr_text': False,
                        'embedding_type': 'text',  # 标记为文本embedding
                        'image_caption': caption,
                        'related_visual_chunk': f"{document_name}_img_{self.chunk_counter-1}",  # 关联的视觉切片ID
                        'created_at': datetime.now().isoformat()
                    }
                )
                
                chunks.append(caption_chunk)
                self.chunk_counter += 1
        
        return chunks
    
    def _chunk_table_group(self, table_group: List[Dict], document_name: str, output_dir: str) -> List[Chunk]:
        """对表格组进行切片
        
        表格通常作为单独的切片处理，包含表格的HTML内容和相关描述信息。
        对于表格类切片，需要执行双入库操作：
        1. 以表格图像作为embedding输入，metadata保存表格描述和Base64参数
        2. 以表格标题和内容作为embedding输入，metadata保存表格描述和Base64参数
        """
        chunks = []
        
        for item in table_group:
            table_body = item.get('table_body')
            if not table_body:
                continue
            
            # 获取表格相关信息
            caption = item.get('table_caption', [])
            footnote = item.get('table_footnote', [])
            img_path = item.get('img_path')
            
            # 构建内容描述
            content_parts = []
            
            # 添加表格标题
            if self.config.include_table_caption and caption:
                content_parts.append(f"表格标题: {'; '.join(caption)}")
            
            # 添加表格内容
            content_parts.append("表格内容:")
            content_parts.append(table_body)
            
            # 添加表格脚注
            if footnote:
                content_parts.append(f"表格脚注: {'; '.join(footnote)}")
            
            content = "\n".join(content_parts)
            
            # 构建表格图片路径（如果存在）
            full_img_path = None
            if img_path:
                full_img_path = os.path.join(output_dir, img_path)
            
            # 第一个切片：以表格图像作为embedding输入（如果有图片）
            if img_path:
                table_visual_chunk = Chunk(
                    chunk_id=f"{document_name}_table_{self.chunk_counter}",
                    chunk_type=ChunkType.TABLE,
                    content=content,
                    content_path=full_img_path,
                    page_idx=item.get('page_idx'),
                    chunk_idx=self.chunk_counter,
                    source_file=document_name,
                    parent_document=document_name,
                    metadata={
                        'chunk_strategy': 'table_visual',
                        'table_image_path': img_path,
                        'has_caption': bool(caption),
                        'has_footnote': bool(footnote),
                        'table_html_length': len(table_body),
                        'embedding_type': 'visual',  # 标记为视觉embedding
                        'table_caption': '; '.join(caption) if caption else '',
                        'table_content': table_body,
                        'created_at': datetime.now().isoformat()
                    }
                )
                
                chunks.append(table_visual_chunk)
                self.chunk_counter += 1
            
            # 第二个切片：以表格文本内容作为embedding输入
            table_text_chunk = Chunk(
                chunk_id=f"{document_name}_table_text_{self.chunk_counter}",
                chunk_type=ChunkType.TABLE,
                content=content,
                content_path=full_img_path,
                page_idx=item.get('page_idx'),
                chunk_idx=self.chunk_counter,
                source_file=document_name,
                parent_document=document_name,
                metadata={
                    'chunk_strategy': 'table_text',
                    'table_image_path': img_path,
                    'has_caption': bool(caption),
                    'has_footnote': bool(footnote),
                    'table_html_length': len(table_body),
                    'embedding_type': 'text',  # 标记为文本embedding
                    'table_caption': '; '.join(caption) if caption else '',
                    'table_content': table_body,
                    'related_visual_chunk': f"{document_name}_table_{self.chunk_counter-1}" if img_path else None,  # 关联的视觉切片ID
                    'created_at': datetime.now().isoformat()
                }
            )
            
            chunks.append(table_text_chunk)
            self.chunk_counter += 1
        
        return chunks
    
    def _chunk_equation_group(self, equation_group: List[Dict], document_name: str, output_dir: str) -> List[Chunk]:
        """对数学公式组进行切片
        
        数学公式通常作为单独的切片处理，包含公式的LaTeX内容和相关信息。
        """
        chunks = []
        
        for item in equation_group:
            latex_text = item.get('text', '')
            if not latex_text:
                continue
            
            # 构建内容描述
            content_parts = []
            
            # 添加公式类型说明
            content_parts.append("数学公式:")
            content_parts.append(latex_text)
            
            content = "\n".join(content_parts)
            
            # 构建公式图片路径（如果存在）
            img_path = item.get('img_path')
            full_img_path = None
            if img_path:
                full_img_path = os.path.join(output_dir, img_path)
            
            chunk = Chunk(
                chunk_id=f"{document_name}_equation_{self.chunk_counter}",
                chunk_type=ChunkType.EQUATION,
                content=content,
                content_path=full_img_path,
                page_idx=item.get('page_idx'),
                chunk_idx=self.chunk_counter,
                source_file=document_name,
                parent_document=document_name,
                metadata={
                    'chunk_strategy': 'equation_single',
                    'equation_image_path': img_path,
                    'text_format': item.get('text_format', 'latex'),
                    'has_latex_text': bool(latex_text),
                    'has_equation_image': bool(img_path),
                    'latex_length': len(latex_text),
                    'created_at': datetime.now().isoformat()
                }
            )
            
            chunks.append(chunk)
            self.chunk_counter += 1
        
        return chunks
    
    def _post_process_chunks(self):
        """后处理切片
        
        添加切片间的关联信息，优化切片质量。
        """
        if not self.chunks:
            return
        
        # 添加相邻切片的关联信息
        for i, chunk in enumerate(self.chunks):
            related_chunks = []
            
            # 添加前一个切片
            if i > 0:
                related_chunks.append(self.chunks[i-1].chunk_id)
            
            # 添加后一个切片
            if i < len(self.chunks) - 1:
                related_chunks.append(self.chunks[i+1].chunk_id)
            
            chunk.related_chunks = related_chunks
            
            # 添加全局元数据
            chunk.metadata.update({
                'total_chunks': len(self.chunks),
                'chunk_position': i + 1
            })
    
    def save_chunks(self, output_path: str, format: str = 'json') -> None:
        """保存切片结果
        
        Args:
            output_path: 输出文件路径
            format: 输出格式，支持 'json', 'jsonl'
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump([chunk.to_dict() for chunk in self.chunks], f, 
                             ensure_ascii=False, indent=2)
            elif format == 'jsonl':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for chunk in self.chunks:
                        f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')
            else:
                raise ValueError(f"不支持的输出格式: {format}")
            
            logger.info(f"切片结果已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存切片结果失败: {str(e)}")
            raise
    
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """获取切片统计信息
        
        Returns:
            包含切片统计信息的字典
        """
        if not self.chunks:
            return {}
        
        # 按类型统计
        type_counts = {}
        for chunk in self.chunks:
            chunk_type = chunk.chunk_type.value
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        # 内容长度统计
        content_lengths = [len(chunk.content) for chunk in self.chunks]
        
        # 页面分布统计
        page_distribution = {}
        for chunk in self.chunks:
            if chunk.page_idx is not None:
                page_distribution[chunk.page_idx] = page_distribution.get(chunk.page_idx, 0) + 1
        
        return {
            'total_chunks': len(self.chunks),
            'chunk_types': type_counts,
            'content_length_stats': {
                'min': min(content_lengths) if content_lengths else 0,
                'max': max(content_lengths) if content_lengths else 0,
                'avg': sum(content_lengths) / len(content_lengths) if content_lengths else 0
            },
            'page_distribution': page_distribution,
            'config_used': {
                'max_chunk_size': self.config.max_chunk_size,
                'min_chunk_size': self.config.min_chunk_size,
                'overlap_size': self.config.overlap_size,
                'text_strategy': self.config.text_strategy.value
            }
        }


def main():
    """主函数，用于测试切片功能"""
    # 配置切片参数
    config = ChunkConfig(
        max_chunk_size=1000,
        min_chunk_size=100,
        overlap_size=50,
        text_strategy=ChunkStrategy.SEMANTIC
    )
    
    # 创建切片代理
    chunk_agent = ChunkAgent(config)
    
    # 测试数据路径
    test_output_dir = "/home/rt/proj_lab/Vision_RAG/test_output/first_output"
    
    # 测试不同类型的文档
    test_documents = [
        "OminiSVG/OminiSVG/auto/OminiSVG_content_list.json",
        "demo/demo/auto/demo_content_list.json",
        "resource/resource/auto/resource_content_list.json"
    ]
    
    for doc_path in test_documents:
        full_path = os.path.join(test_output_dir, doc_path)
        if os.path.exists(full_path):
            print(f"\n处理文档: {doc_path}")
            
            try:
                # 执行切片
                chunks = chunk_agent.chunk_document(
                    content_list_path=full_path,
                    output_dir=os.path.dirname(full_path),
                    document_name=os.path.basename(doc_path).replace('_content_list.json', '')
                )
                
                # 输出统计信息
                stats = chunk_agent.get_chunk_statistics()
                print(f"切片统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
                
                # 保存切片结果
                output_file = full_path.replace('_content_list.json', '_chunks.json')
                chunk_agent.save_chunks(output_file)
                
            except Exception as e:
                print(f"处理文档失败: {str(e)}")
        else:
            print(f"文档不存在: {full_path}")


if __name__ == "__main__":
    main()