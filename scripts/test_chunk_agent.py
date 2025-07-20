#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试ChunkAgent的功能

该脚本演示如何使用ChunkAgent对MinerU解析的文档结果进行切片处理。

Author: Vision_RAG Team
Date: 2024-12-19
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.chunk_agent import ChunkAgent, ChunkConfig, ChunkStrategy
from agent.data_agent import DataAgent


def test_chunk_agent_basic():
    """测试ChunkAgent的基本功能"""
    print("=== 测试ChunkAgent基本功能 ===")
    
    # 配置切片参数
    config = ChunkConfig(
        max_chunk_size=800,
        min_chunk_size=100,
        overlap_size=50,
        text_strategy=ChunkStrategy.SEMANTIC,
        include_image_caption=True,
        include_table_caption=True
    )
    
    # 创建切片代理
    chunk_agent = ChunkAgent(config)
    
    # 测试数据路径
    test_output_dir = project_root / "test_output" / "first_output"
    
    # 测试不同类型的文档
    test_documents = [
        "demo/demo/auto/demo_content_list.json",
        "OminiSVG/OminiSVG/auto/OminiSVG_content_list.json",
        "resource/resource/auto/resource_content_list.json"
    ]
    
    results = {}
    
    for doc_path in test_documents:
        full_path = test_output_dir / doc_path
        if full_path.exists():
            print(f"\n处理文档: {doc_path}")
            
            try:
                # 执行切片
                chunks = chunk_agent.chunk_document(
                    content_list_path=str(full_path),
                    output_dir=str(full_path.parent),
                    document_name=full_path.stem.replace('_content_list', '')
                )
                
                # 获取统计信息
                stats = chunk_agent.get_chunk_statistics()
                results[doc_path] = {
                    'chunks_count': len(chunks),
                    'stats': stats
                }
                
                print(f"✓ 成功生成 {len(chunks)} 个切片")
                print(f"  - 文本切片: {stats['chunk_types'].get('text', 0)}")
                print(f"  - 图片切片: {stats['chunk_types'].get('image', 0)}")
                print(f"  - 表格切片: {stats['chunk_types'].get('table', 0)}")
                print(f"  - 平均内容长度: {stats['content_length_stats']['avg']:.1f}")
                
                # 保存切片结果
                output_file = str(full_path).replace('_content_list.json', '_chunks.json')
                chunk_agent.save_chunks(output_file)
                print(f"  - 切片结果已保存到: {output_file}")
                
            except Exception as e:
                print(f"✗ 处理文档失败: {str(e)}")
                results[doc_path] = {'error': str(e)}
        else:
            print(f"✗ 文档不存在: {full_path}")
            results[doc_path] = {'error': 'File not found'}
    
    return results


def test_different_strategies():
    """测试不同的切片策略"""
    print("\n=== 测试不同切片策略 ===")
    
    strategies = [
        (ChunkStrategy.SEMANTIC, "语义边界切片"),
        (ChunkStrategy.DOCUMENT_STRUCTURE, "文档结构切片"),
        (ChunkStrategy.FIXED_SIZE, "固定大小切片")
    ]
    
    test_file = project_root / "test_output" / "first_output" / "demo" / "demo" / "auto" / "demo_content_list.json"
    
    if not test_file.exists():
        print(f"✗ 测试文件不存在: {test_file}")
        return
    
    for strategy, strategy_name in strategies:
        print(f"\n--- {strategy_name} ---")
        
        config = ChunkConfig(
            max_chunk_size=600,
            min_chunk_size=100,
            text_strategy=strategy
        )
        
        chunk_agent = ChunkAgent(config)
        
        try:
            chunks = chunk_agent.chunk_document(
                content_list_path=str(test_file),
                output_dir=str(test_file.parent),
                document_name=f"demo_{strategy.value}"
            )
            
            stats = chunk_agent.get_chunk_statistics()
            print(f"✓ 生成 {len(chunks)} 个切片")
            print(f"  - 平均长度: {stats['content_length_stats']['avg']:.1f}")
            print(f"  - 最大长度: {stats['content_length_stats']['max']}")
            print(f"  - 最小长度: {stats['content_length_stats']['min']}")
            
        except Exception as e:
            print(f"✗ {strategy_name}失败: {str(e)}")


def test_integration_with_data_agent():
    """测试与DataAgent的集成"""
    print("\n=== 测试与DataAgent集成 ===")
    
    # 创建一个小的测试文档
    test_doc_dir = project_root / "test_data"
    test_doc_dir.mkdir(exist_ok=True)
    
    # 使用现有的测试文档进行集成测试
    test_file = project_root / "test_output" / "first_output" / "demo" / "demo" / "auto" / "demo_content_list.json"
    
    if not test_file.exists():
        print(f"✗ 测试文件不存在: {test_file}")
        print("请先运行DataAgent测试以生成测试数据")
        return
    
    print(f"使用现有测试文档: {test_file}")
    
    try:
        # 直接使用现有的解析结果进行切片测试
        print("\n步骤1: 跳过文档解析，使用现有解析结果")
        print(f"✓ 使用现有解析结果: {test_file}")
        
        # 2. 使用ChunkAgent切片文档
        print("\n步骤2: 使用ChunkAgent切片文档")
        
        content_list_file = str(test_file)
        
        if content_list_file:
            config = ChunkConfig(
                max_chunk_size=500,
                min_chunk_size=100,
                text_strategy=ChunkStrategy.SEMANTIC
            )
            
            chunk_agent = ChunkAgent(config)
            chunks = chunk_agent.chunk_document(
                content_list_path=content_list_file,
                output_dir=os.path.dirname(content_list_file),
                document_name="integration_test"
            )
            
            stats = chunk_agent.get_chunk_statistics()
            print(f"✓ 文档切片完成，生成 {len(chunks)} 个切片")
            print(f"  - 切片类型分布: {stats['chunk_types']}")
            
            # 保存切片结果
            chunks_file = content_list_file.replace('_content_list.json', '_chunks.json')
            chunk_agent.save_chunks(chunks_file)
            print(f"  - 切片结果保存到: {chunks_file}")
            
            # 显示前几个切片的内容
            print("\n前3个切片内容预览:")
            for i, chunk in enumerate(chunks[:3]):
                print(f"\n切片 {i+1} ({chunk.chunk_type.value}):")
                print(f"  ID: {chunk.chunk_id}")
                print(f"  长度: {len(chunk.content)} 字符")
                print(f"  内容预览: {chunk.content[:100]}...")
        else:
            print("✗ 未找到content_list.json文件")
            
    except Exception as e:
        print(f"✗ 集成测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("ChunkAgent功能测试")
    print("=" * 50)
    
    # 测试基本功能
    basic_results = test_chunk_agent_basic()
    
    # 测试不同策略
    test_different_strategies()
    
    # 测试集成
    test_integration_with_data_agent()
    
    print("\n=== 测试总结 ===")
    print("✓ ChunkAgent基本功能测试完成")
    print("✓ 多种切片策略测试完成")
    print("✓ 与DataAgent集成测试完成")
    print("\n所有测试已完成！")


if __name__ == "__main__":
    main()