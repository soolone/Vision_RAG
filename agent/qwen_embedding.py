#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通义千问多模态嵌入工具类

简单的embedding工具，支持文本和图片输入，返回模型调用结果。
"""

import sys
import os
import base64
from pathlib import Path
from typing import Optional, Union

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dashscope
from utils.apis import AliBailian_API


class TongyiEmbedding:
    """通义千问嵌入工具类"""
    
    def __init__(self, api_key: Optional[str] = None):
        """初始化
        
        Args:
            api_key: API密钥，不提供则使用默认配置
        """
        if api_key:
            dashscope.api_key = api_key
        else:
            api_config = AliBailian_API()
            dashscope.api_key = api_config.api_key
    
    def _image_to_base64(self, image_path: str) -> str:
        """图片转Base64"""
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        
        # 获取图片格式
        image_format = Path(image_path).suffix.lower().lstrip('.')
        if image_format == 'jpg':
            image_format = 'jpeg'
            
        return f"data:image/{image_format};base64,{base64_image}"
    
    def get_embedding(self, input_data: Union[str, dict]):
        """获取embedding
        
        Args:
            input_data: 输入数据，可以是:
                - str: 文本内容或图片路径
                - dict: 指定输入类型，如 {'text': '文本'} 或 {'image': '图片路径'}
            
        Returns:
            模型调用结果
        """
        # 处理输入数据
        if isinstance(input_data, str):
            # 判断是文本还是图片路径
            if self._is_image_path(input_data):
                # 图片路径
                image_data = self._image_to_base64(input_data)
                model_input = [{'image': image_data}]
            else:
                # 文本内容
                model_input = [{'text': input_data}]
        elif isinstance(input_data, dict):
            if 'text' in input_data:
                model_input = [{'text': input_data['text']}]
            elif 'image' in input_data:
                if input_data['image'].startswith(('http://', 'https://')):
                    # URL形式的图片
                    model_input = [{'image': input_data['image']}]
                else:
                    # 本地图片路径
                    image_data = self._image_to_base64(input_data['image'])
                    model_input = [{'image': image_data}]
            else:
                raise ValueError("字典输入必须包含 'text' 或 'image' 键")
        else:
            raise ValueError("输入数据必须是字符串或字典")
        
        # 调用模型
        resp = dashscope.MultiModalEmbedding.call(
            model="multimodal-embedding-v1",
            input=model_input
        )
        
        return resp
    
    def get_text_embedding(self, text: str):
        """获取文本embedding
        
        Args:
            text: 文本内容
            
        Returns:
            模型调用结果
        """
        return self.get_embedding({'text': text})
    
    def get_image_embedding(self, image_input: str):
        """获取图片embedding
        
        Args:
            image_input: 图片路径或URL
            
        Returns:
            模型调用结果
        """
        return self.get_embedding({'image': image_input})
    
    def _is_image_path(self, text: str) -> bool:
        """判断字符串是否为图片路径
        
        Args:
            text: 输入字符串
            
        Returns:
            是否为图片路径
        """
        # 检查是否为文件路径且存在
        if os.path.exists(text):
            # 检查文件扩展名
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
            return Path(text).suffix.lower() in image_extensions
        return False
    
if __name__ == "__main__":
    # 简单测试
    embedding = TongyiEmbedding()
    
    # 测试文本输入
    print("=== 测试文本输入 ===")
    text_result = embedding.get_embedding("这是一个测试文本")
    print(f"文本embedding结果: {text_result.status_code}")
    
    # 测试图片输入（如果文件存在）
    image_path = "pdf_pages/OminiSVG/page_1.png"
    if os.path.exists(image_path):
        print("\n=== 测试图片输入 ===")
        image_result = embedding.get_embedding(image_path)
        print(f"图片embedding结果: {image_result.status_code}")
    
    # 测试字典格式输入
    print("\n=== 测试字典格式输入 ===")
    dict_result = embedding.get_embedding({'text': '字典格式的文本输入'})
    print(f"字典文本embedding结果: {dict_result.status_code}")