import os
import io
import base64
import PIL
from PIL import Image
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import all agent tools
try:
    from agent.data_agent import DataAgent
    from agent.chunk_agent import ChunkAgent, ChunkConfig
    from agent.chromadb_agent import ChromaDBManager
    from agent.qwen_embedding import TongyiEmbedding
    from agent.query_agent import QueryExpansionAgent
    from agent.answer_agent import AnswerAgent
except ImportError as e:
    st.error(f"导入错误: {e}")
    DataAgent = None
    ChunkAgent = None
    ChunkConfig = None
    ChromaDBManager = None
    TongyiEmbedding = None
    QueryExpansionAgent = None
    AnswerAgent = None

# --- Helper functions ---

def process_document_with_agents(file_path: str, config) -> List:
    """使用工具类处理文档"""
    try:
        # 初始化数据代理
        data_agent = DataAgent(
            output_dir=config.output_dir,
            cls_dir=config.cls_dir,
            lang=config.lang,
            enable_formula=config.enable_formula,
            enable_table=config.enable_table,
            auto_caption=config.auto_caption
        )
        
        # 解析文档
        parse_result = data_agent.parse_document(file_path)
        
        if not parse_result.success:
            st.error(f"文档解析失败: {parse_result.error_message}")
            return []
        
        # 初始化切片代理
        chunk_config = ChunkConfig(
            max_chunk_size=config.max_chunk_size,
            min_chunk_size=config.min_chunk_size,
            overlap_size=config.overlap_size,
            preserve_sentences=config.preserve_sentences,
            preserve_paragraphs=config.preserve_paragraphs
        )
        chunk_agent = ChunkAgent(chunk_config)
        
        # 查找content_list文件
        content_list_files = list(Path(parse_result.output_dir).rglob("*_content_list.json"))
        if not content_list_files:
            st.error("未找到content_list文件")
            return []
        
        # 对文档进行切片
        # 使用content_list文件的父目录作为output_dir，这样可以正确找到images目录
        actual_output_dir = str(content_list_files[0].parent)
        chunks = chunk_agent.chunk_document(
            str(content_list_files[0]),
            actual_output_dir,
            Path(file_path).stem
        )
        
        return chunks
        
    except Exception as e:
        st.error(f"处理文档时出错: {e}")
        return []

def display_knowledge_base_management():
    """显示知识库管理界面"""
    st.header("📚 知识库管理")
    
    # 获取数据库管理器
    if 'db_manager' not in st.session_state or st.session_state.db_manager is None:
        st.warning("数据库管理器未初始化，请先在侧边栏配置数据库。")
        return
    
    db_manager = st.session_state.db_manager
    
    # 获取数据库统计信息和所有文档
    try:
        collection_info = db_manager.get_collection_info()
        total_chunks = collection_info.get('count', 0)
        
        if total_chunks == 0:
            st.info("知识库为空，请先上传并解析文档。")
            return
        
        # 查询所有文档
        all_docs = db_manager.collection.get(
            include=['metadatas', 'documents']
        )
        
        if not all_docs['metadatas']:
            st.info("知识库中没有文档。")
            return
        
        # 计算文档数量
        unique_docs = set()
        for metadata in all_docs['metadatas']:
            doc_name = metadata.get('parent_document') or metadata.get('source_file', 'unknown')
            if doc_name != 'unknown':
                unique_docs.add(doc_name)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("文档数量", len(unique_docs))
        with col2:
            st.metric("总切片数量", total_chunks)
            
    except Exception as e:
        st.error(f"获取数据库信息失败: {e}")
        return
    
    # 按文档分组
    documents_dict = {}
    for i, metadata in enumerate(all_docs['metadatas']):
        # 优先使用parent_document，如果为空则使用source_file
        doc_name = metadata.get('parent_document') or metadata.get('source_file', 'unknown')
        if doc_name not in documents_dict:
            documents_dict[doc_name] = []
        
        chunk_data = {
            'chunk_id': metadata.get('chunk_id', f'chunk_{i}'),
            'chunk_type': metadata.get('chunk_type', 'text'),
            'page_idx': metadata.get('page_idx') or metadata.get('page_number', 0),
            'chunk_idx': metadata.get('chunk_idx', 0),
            'content': all_docs['documents'][i],
            'metadata': metadata
        }
        documents_dict[doc_name].append(chunk_data)
    
    # 显示文档列表
    st.subheader("📄 已解析文档")
    
    for doc_name, chunks in documents_dict.items():
            # 按页面和切片索引排序
            chunks.sort(key=lambda x: (x['page_idx'], x['chunk_idx']))
            
            # 显示文档统计
            chunk_types = {}
            for chunk in chunks:
                chunk_type = chunk['chunk_type']
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            # 创建文档标题行，包含删除按钮
            col1, col2 = st.columns([4, 1])
            with col1:
                doc_expander = st.expander(f"📖 {doc_name} ({len(chunks)} 个切片)", expanded=False)
            with col2:
                if st.button(f"🗑️ 删除", key=f"delete_{doc_name}", help=f"删除文档 {doc_name} 的所有切片"):
                    st.session_state[f"confirm_delete_{doc_name}"] = True
                    st.rerun()
            
            # 如果处于确认删除状态，显示确认按钮
            if st.session_state.get(f"confirm_delete_{doc_name}", False):
                st.warning(f"⚠️ 确认删除文档 '{doc_name}' 的所有 {len(chunks)} 个切片？此操作不可恢复！")
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("✅ 确认删除", key=f"confirm_yes_{doc_name}"):
                        try:
                            deleted_count = db_manager.delete_documents_by_source(doc_name)
                            if deleted_count > 0:
                                st.success(f"成功删除文档 '{doc_name}' 的 {deleted_count} 个切片")
                                # 清除确认状态
                                del st.session_state[f"confirm_delete_{doc_name}"]
                                st.rerun()
                            else:
                                st.error(f"删除文档 '{doc_name}' 失败：未找到相关切片")
                        except Exception as e:
                            st.error(f"删除文档时出错: {e}")
                with col2:
                    if st.button("❌ 取消", key=f"confirm_no_{doc_name}"):
                        del st.session_state[f"confirm_delete_{doc_name}"]
                        st.rerun()
                continue
            
            # 文档标题和基本信息
            with doc_expander:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**切片类型统计:**")
                    for chunk_type, count in chunk_types.items():
                        st.write(f"- {chunk_type}: {count}")
                
                with col2:
                    st.write("**页面范围:**")
                    pages = set(chunk['page_idx'] for chunk in chunks if chunk['page_idx'] is not None)
                    if pages:
                        if len(pages) == 1:
                            st.write(f"第 {list(pages)[0]} 页")
                        else:
                            st.write(f"第 {min(pages)} - {max(pages)} 页")
                    else:
                        st.write("页面信息不可用")
                
                st.divider()
                
                # 显示切片详情
                st.write("**切片详情:**")
                
                # 对切片进行去重处理，合并双入库的切片
                deduplicated_chunks = deduplicate_chunks_for_display(chunks)
                
                for chunk_info in deduplicated_chunks:
                    chunk_type = chunk_info['chunk_type']
                    page_idx = chunk_info['page_idx']
                    chunk_idx_display = chunk_info['chunk_idx_display']
                    primary_chunk = chunk_info['primary_chunk']
                    secondary_chunk = chunk_info.get('secondary_chunk')
                    
                    # 每个切片使用独立的expander
                    with st.expander(f"切片 {chunk_idx_display} (页面 {page_idx}, 类型: {chunk_type})", expanded=False):
                        chunk = primary_chunk  # 使用主切片进行显示
                        if chunk_type == 'image':
                            # 使用列布局，图片在左侧，描述在右侧
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                # 显示图片
                                try:
                                    image_displayed = False
                                    
                                    # 首先检查是否有original_content (base64数据)
                                    if 'original_content' in chunk['metadata']:
                                        try:
                                            import base64
                                            import io
                                            from PIL import Image
                                            
                                            original_content = chunk['metadata']['original_content']
                                            if original_content:
                                                # 解码Base64图片
                                                image_data = base64.b64decode(original_content)
                                                image = Image.open(io.BytesIO(image_data))
                                                st.image(image, caption=f"图片切片 {chunk_idx_display}", use_container_width=True)
                                                image_displayed = True
                                        except Exception as b64_error:
                                            st.warning(f"解码base64图片失败: {b64_error}")
                                    
                                    # 如果base64显示失败，尝试从本地路径加载
                                    if not image_displayed and 'image_path' in chunk['metadata']:
                                        image_path = chunk['metadata']['image_path']
                                        # 如果是相对路径，尝试构建绝对路径
                                        if not os.path.isabs(image_path) and chunk.get('content_path'):
                                            # 使用content_path作为绝对路径
                                            full_image_path = chunk['content_path']
                                        else:
                                            full_image_path = image_path
                                        
                                        if os.path.exists(full_image_path):
                                            st.image(full_image_path, caption=f"图片切片 {chunk_idx_display}", use_container_width=True)
                                            image_displayed = True
                                        else:
                                            st.warning(f"图片文件不存在: {full_image_path}")
                                    
                                    # 如果都失败了，显示错误信息
                                    if not image_displayed:
                                        st.warning("无法显示图片：缺少图片数据")
                                        
                                except Exception as img_error:
                                    st.error(f"显示图片时出错: {img_error}")
                            
                            with col2:
                                # 显示图片的文本描述（如果有的话）
                                content = chunk.get('content', '')
                                if content and content.strip():
                                    st.text_area(
                                        "图片描述",
                                        content,
                                        height=200,
                                        disabled=False
                                    )
                        
                        elif chunk_type == 'table':
                            # 使用列布局，表格图片在左侧，文本内容在右侧
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                # 如果有表格图片，显示图片
                                if 'table_image_path' in chunk['metadata'] or 'image_path' in chunk['metadata']:
                                    # 优先使用table_image_path，如果没有则使用image_path
                                    image_path = chunk['metadata'].get('table_image_path') or chunk['metadata'].get('image_path')
                                    # 如果是相对路径，尝试构建绝对路径
                                    if not os.path.isabs(image_path) and chunk.get('content_path'):
                                        # 使用content_path作为绝对路径
                                        full_image_path = chunk['content_path']
                                    else:
                                        full_image_path = image_path
                                    
                                    if os.path.exists(full_image_path):
                                        st.image(full_image_path, caption=f"表格图片", use_container_width=True)
                                    else:
                                        st.warning(f"表格图片文件不存在: {full_image_path}")
                            
                            with col2:
                                # 显示表格文本内容
                                content = chunk.get('content', '')
                                if content and content.strip():
                                    st.text_area(
                                        "表格文本内容",
                                        content,
                                        height=200,
                                        disabled=False
                                    )
                        
                        else:
                            # 显示文本内容
                            content = chunk['content']
                            if content:
                                # 限制显示长度
                                display_content = content[:500] + "..." if len(content) > 500 else content
                                st.text_area(
                                    f"文本内容",
                                    display_content,
                                    height=100,
                                    disabled=False
                                )
                        
                        # 显示元数据
                        with st.expander(f"元数据", expanded=False):
                            metadata_display = {k: v for k, v in chunk['metadata'].items() 
                                              if k not in ['original_content']}  # 排除base64数据
                            st.json(metadata_display)
    
    # 数据库操作
    st.subheader("🛠️ 数据库操作")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 刷新知识库", type="secondary"):
            st.rerun()
    
    with col2:
        if st.button("🗑️ 清空知识库", type="secondary"):
            if st.session_state.get('confirm_clear_kb', False):
                try:
                    db_manager.clear_collection()
                    st.success("知识库已清空")
                    st.session_state.confirm_clear_kb = False
                    st.rerun()
                except Exception as e:
                    st.error(f"清空知识库失败: {e}")
            else:
                st.session_state.confirm_clear_kb = True
                st.warning("请再次点击确认清空知识库")
    
    if st.session_state.get('confirm_clear_kb', False):
        st.warning("⚠️ 注意：清空操作不可恢复！")

def deduplicate_chunks_for_display(chunks: List[Dict]) -> List[Dict]:
    """对知识库管理页面的切片进行去重处理，合并双入库的切片
    
    Args:
        chunks: 原始切片列表
        
    Returns:
        去重后的切片信息列表，每个元素包含:
        - chunk_type: 切片类型
        - page_idx: 页面索引
        - chunk_idx_display: 显示用的切片索引（如"5-6"表示双入库）
        - primary_chunk: 主切片（用于显示）
        - secondary_chunk: 次切片（如果存在）
    """
    if not chunks:
        return []
    
    deduplicated = []
    processed_chunks = set()
    
    for chunk in chunks:
        chunk_id = chunk.get('chunk_id', '') or chunk.get('id', '')
        if chunk_id in processed_chunks:
            continue
            
        chunk_type = chunk['chunk_type']
        metadata = chunk.get('metadata', {})
        
        # 检查是否是双入库的切片（仅对image和table类型进行双入库合并）
        if chunk_type in ['image', 'table']:
            embedding_type = metadata.get('embedding_type', '')
            
            if embedding_type == 'visual':
                # 这是视觉embedding切片，寻找对应的文本embedding切片
                text_chunk = None
                
                # 使用更稳定的标识符来查找配对切片
                source_file = metadata.get('source_file', '')
                page_idx = chunk['page_idx']
                chunk_strategy = metadata.get('chunk_strategy', '')
                
                # 构建基础标识符
                if chunk_type == 'image':
                    image_path = metadata.get('image_path', '')
                    base_identifier = f"{chunk_type}_{source_file}_{page_idx}_{image_path}"
                else:  # table
                    table_image_path = metadata.get('table_image_path', '')
                    base_identifier = f"{chunk_type}_{source_file}_{page_idx}_{table_image_path}"
                
                # 查找对应的文本embedding切片
                for other_chunk in chunks:
                    other_metadata = other_chunk.get('metadata', {})
                    if (other_chunk['chunk_type'] == chunk_type and
                        other_chunk['page_idx'] == page_idx and
                        other_metadata.get('embedding_type') == 'text' and
                        other_metadata.get('source_file') == source_file):
                        
                        # 检查是否是同一个图像/表格的文本版本
                        if chunk_type == 'image':
                            other_image_path = other_metadata.get('image_path', '')
                            if other_image_path == image_path:
                                text_chunk = other_chunk
                                break
                        else:  # table
                            other_table_image_path = other_metadata.get('table_image_path', '')
                            if other_table_image_path == table_image_path:
                                text_chunk = other_chunk
                                break
                
                if text_chunk:
                    # 找到了配对的切片，合并显示
                    chunk_info = {
                        'chunk_type': chunk_type,
                        'page_idx': chunk['page_idx'],
                        'chunk_idx_display': f"{chunk['chunk_idx']}-{text_chunk['chunk_idx']}",
                        'primary_chunk': chunk,  # 使用视觉切片作为主切片
                        'secondary_chunk': text_chunk
                    }
                    processed_chunks.add(chunk_id)
                    processed_chunks.add(text_chunk.get('chunk_id', '') or text_chunk.get('id', ''))
                else:
                    # 没有找到配对切片，单独显示
                    chunk_info = {
                        'chunk_type': chunk_type,
                        'page_idx': chunk['page_idx'],
                        'chunk_idx_display': str(chunk['chunk_idx']),
                        'primary_chunk': chunk
                    }
                    processed_chunks.add(chunk_id)
                    
            elif embedding_type == 'text':
                # 这是文本embedding切片，检查是否已经被处理过
                if chunk_id not in processed_chunks:
                    # 单独的文本切片（没有对应的视觉切片）
                    chunk_info = {
                        'chunk_type': chunk_type,
                        'page_idx': chunk['page_idx'],
                        'chunk_idx_display': str(chunk['chunk_idx']),
                        'primary_chunk': chunk
                    }
                    processed_chunks.add(chunk_id)
                else:
                    continue
            else:
                # 没有embedding_type标记的切片，单独显示
                chunk_info = {
                    'chunk_type': chunk_type,
                    'page_idx': chunk['page_idx'],
                    'chunk_idx_display': str(chunk['chunk_idx']),
                    'primary_chunk': chunk
                }
                processed_chunks.add(chunk_id)
        else:
            # 非image/table类型的切片（如text类型），直接显示，不进行双入库合并
            chunk_info = {
                'chunk_type': chunk_type,
                'page_idx': chunk['page_idx'],
                'chunk_idx_display': str(chunk['chunk_idx']),
                'primary_chunk': chunk
            }
            processed_chunks.add(chunk_id)
        
        deduplicated.append(chunk_info)
    
    return deduplicated

def deduplicate_search_results(results: List[Dict]) -> List[Dict]:
    """对搜索结果进行去重，避免同一图像或表格的重复使用
    特别处理双入库的image和table类切片，合并相似度信息
    
    Args:
        results: 原始搜索结果列表
        
    Returns:
        去重后的搜索结果列表，双入库的切片会包含两种相似度信息
    """
    if not results:
        return []
    
    # 用于存储已处理的切片
    processed_chunks = {}
    deduplicated_results = []
    
    for result in results:
        metadata = result.get('metadata', {})
        chunk_type = metadata.get('chunk_type', 'unknown')
        
        # 对于image和table类型，检查是否是双入库的切片
        if chunk_type in ['image', 'table']:
            # 构建切片的唯一标识符，基于内容而不是chunk_idx
            source_file = metadata.get('source_file', '')
            page_idx = metadata.get('page_idx', '')
            
            # 对于双入库的切片，使用更稳定的标识符
            # 基于chunk_strategy来识别同源切片
            chunk_strategy = metadata.get('chunk_strategy', '')
            
            # 创建基础标识符，去除embedding相关的后缀
            if chunk_strategy.endswith('_visual'):
                base_strategy = chunk_strategy[:-7]  # 移除'_visual'
            elif chunk_strategy.endswith('_text') or chunk_strategy.endswith('_caption'):
                base_strategy = chunk_strategy.rsplit('_', 1)[0]  # 移除最后一个下划线后的部分
            else:
                base_strategy = chunk_strategy
            
            # 对于image类型，还需要考虑image_path
            if chunk_type == 'image':
                image_path = metadata.get('image_path', '')
                base_key = f"{chunk_type}_{source_file}_{page_idx}_{image_path}_{base_strategy}"
            else:  # table类型
                table_image_path = metadata.get('table_image_path', '')
                base_key = f"{chunk_type}_{source_file}_{page_idx}_{table_image_path}_{base_strategy}"
            
            # 检查是否已经处理过这个切片
            if base_key in processed_chunks:
                # 已经存在，合并相似度信息
                existing_result = processed_chunks[base_key]
                embedding_type = metadata.get('embedding_type', 'unknown')
                
                # 添加第二种相似度信息
                if embedding_type == 'visual':
                    existing_result['visual_similarity'] = 1 - result['distance']
                    existing_result['visual_distance'] = result['distance']
                elif embedding_type == 'text':
                    existing_result['text_similarity'] = 1 - result['distance']
                    existing_result['text_distance'] = result['distance']
                
                # 标记为双入库切片
                existing_result['is_dual_indexed'] = True
            else:
                # 第一次遇到这个切片
                embedding_type = metadata.get('embedding_type', 'unknown')
                
                # 复制结果并添加相似度信息
                new_result = result.copy()
                new_result['is_dual_indexed'] = False
                
                if embedding_type == 'visual':
                    new_result['visual_similarity'] = 1 - result['distance']
                    new_result['visual_distance'] = result['distance']
                elif embedding_type == 'text':
                    new_result['text_similarity'] = 1 - result['distance']
                    new_result['text_distance'] = result['distance']
                
                # 存储到已处理列表
                processed_chunks[base_key] = new_result
                deduplicated_results.append(new_result)
        else:
            # 非image/table类型，使用原有的去重逻辑
            source_key = None
            
            if metadata.get('source_type') == 'image':
                # 图像去重：基于原始文件名和页码
                filename = metadata.get('original_filename') or metadata.get('pdf_filename', '')
                page_num = metadata.get('page_number', '')
                source_key = f"image_{filename}_{page_num}"
            elif metadata.get('source_type') == 'table':
                # 表格去重：基于原始文件名、页码和表格索引
                filename = metadata.get('original_filename') or metadata.get('pdf_filename', '')
                page_num = metadata.get('page_number', '')
                table_idx = metadata.get('table_index', '')
                source_key = f"table_{filename}_{page_num}_{table_idx}"
            else:
                # 其他类型：基于文件名和内容哈希
                filename = metadata.get('original_filename') or metadata.get('pdf_filename', '')
                content_hash = hash(result.get('document', '')[:100])  # 使用内容前100字符的哈希
                source_key = f"other_{filename}_{content_hash}"
            
            # 检查是否已经见过这个源
            if source_key and source_key not in processed_chunks:
                processed_chunks[source_key] = result
                deduplicated_results.append(result)
    
    return deduplicated_results

def search_similar_images(query: str, db_manager: ChromaDBManager, top_k: int = 3, similarity_threshold: Optional[float] = None, query_type: str = 'text', image_path: Optional[str] = None) -> Optional[List[Dict]]:
    """Search for similar images based on text or image query.
    
    Args:
        query: Search query text or image description.
        db_manager: ChromaDBManager instance.
        top_k: Number of top results to return.
        similarity_threshold: Similarity threshold (0-1), results below this threshold will be filtered out.
        query_type: Type of query ('text' or 'image').
        image_path: Path to image file when query_type is 'image'.
        
    Returns:
        List of search results with metadata, or None if failed.
    """
    try:
        # 根据查询类型选择合适的搜索方法
        if query_type == 'image' and image_path:
            # 使用图像进行搜索
            results = db_manager.search_documents(
                query, 
                n_results=top_k * 2,  # 获取更多结果用于去重
                similarity_threshold=similarity_threshold,
                embedding_type='visual',
                image_path=image_path
            )
        else:
            # 使用文本进行搜索
            results = db_manager.search_documents(
                query, 
                n_results=top_k * 2,  # 获取更多结果用于去重
                similarity_threshold=similarity_threshold,
                embedding_type='text'
            )
        
        if not results or not results['documents']:
            return None
            
        # Format results
        formatted_results = []
        for i in range(len(results['documents'])):
            result_data = {
                'document': results['documents'][i],
                'distance': results['distances'][i],
                'metadata': results['metadatas'][i],
                'id': results['ids'][i]
            }
            formatted_results.append(result_data)
        
        # 对结果进行去重
        deduplicated_results = deduplicate_search_results(formatted_results)
        
        # 返回top_k个去重后的结果
        return deduplicated_results[:top_k]
        
    except Exception as e:
        st.error(f"搜索过程中出错：{e}")
        return None

def display_chunk_content(result: Dict, index: int):
    """根据chunk类型展示对应的内容（文本、图像、表格等）
    
    Args:
        result: 单个搜索结果字典
        index: 结果索引
    """
    metadata = result['metadata']
    source_type = metadata.get('chunk_type', 'unknown')
    similarity = 1 - result['distance']
    filename = metadata.get('source_file', '未知')
    
    # 根据类型设置图标
    type_icons = {
        'image': '🖼️',
        'table': '📊', 
        'text': '📝'
    }
    icon = type_icons.get(source_type, '❓')
    
    # 检查是否是双入库切片并准备相似度信息
    similarity_info = ""
    if result.get('is_dual_indexed', False):
        # 双入库切片，显示两种相似度
        visual_sim = result.get('visual_similarity')
        text_sim = result.get('text_similarity')
        
        if visual_sim is not None and text_sim is not None:
            similarity_info = f"图像-文本: {visual_sim:.3f}, 文本-文本: {text_sim:.3f}"
        elif visual_sim is not None:
            similarity_info = f"图像-文本: {visual_sim:.3f}"
        elif text_sim is not None:
            similarity_info = f"文本-文本: {text_sim:.3f}"
        else:
            similarity_info = f"{similarity:.3f}"
    else:
        # 单一入库切片，显示单一相似度
        embedding_type = metadata.get('embedding_type', 'unknown')
        if embedding_type == 'visual':
            similarity_info = f"图像-文本: {similarity:.3f}"
        elif embedding_type == 'text':
            similarity_info = f"文本-文本: {similarity:.3f}"
        else:
            similarity_info = f"{similarity:.3f}"
    
    # 创建可折叠的参考资料展示
    with st.expander(f"{icon} 参考资料 {index + 1} - {filename} (相似度: {similarity_info})", expanded=False):
        # 显示基本信息
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**来源:** {filename}")
            if metadata.get('page_number'):
                st.markdown(f"**页码:** {metadata['page_number']}")
        with col2:
            st.markdown(f"**类型:** {icon} {source_type}")
            if result.get('is_dual_indexed', False):
                st.markdown(f"**双入库切片**")
                if result.get('visual_similarity') is not None:
                    st.markdown(f"**图像-文本相似度:** {result['visual_similarity']:.3f}")
                if result.get('text_similarity') is not None:
                    st.markdown(f"**文本-文本相似度:** {result['text_similarity']:.3f}")
            else:
                embedding_type = metadata.get('embedding_type', 'unknown')
                if embedding_type == 'visual':
                    st.markdown(f"**图像-文本相似度:** {similarity:.3f}")
                elif embedding_type == 'text':
                    st.markdown(f"**文本-文本相似度:** {similarity:.3f}")
                else:
                    st.markdown(f"**相似度:** {similarity:.3f}")
        
        # 根据chunk类型显示不同内容
        if source_type == 'image':
            # 使用三列布局显示图像、描述和其他信息
            col_img, col_desc, col_info = st.columns([1, 2, 1])
            
            with col_img:
                # 显示图像
                if metadata.get('has_original_content', False):
                    try:
                        original_content = metadata.get('original_content')
                        if original_content:
                            import base64
                            import io
                            image_data = base64.b64decode(original_content)
                            image = Image.open(io.BytesIO(image_data))
                            st.image(image, caption="图像内容", width=200)
                    except Exception as e:
                        st.error(f"显示图像失败: {e}")
            
            with col_desc:
                # 显示图像描述
                if result['document']:
                    st.markdown("**图像描述:**")
                    st.write(result['document'])
            
            with col_info:
                # 显示额外信息（如果需要）
                if metadata.get('chunk_strategy'):
                    st.markdown(f"**切片策略:** {metadata['chunk_strategy']}")
        
        elif source_type == 'table':
            # 使用三列布局显示表格图像、内容和其他信息
            col_img, col_desc, col_info = st.columns([1, 2, 1])
            
            with col_img:
                # 显示表格图像（如果有）
                if metadata.get('has_original_content', False):
                    try:
                        original_content = metadata.get('original_content')
                        if original_content:
                            import base64
                            import io
                            image_data = base64.b64decode(original_content)
                            image = Image.open(io.BytesIO(image_data))
                            st.image(image, caption="表格图像", width=200)
                    except Exception as e:
                        st.error(f"显示表格图像失败: {e}")
            
            with col_desc:
                # 显示表格HTML内容
                if result['document']:
                    st.markdown("**表格内容:**")
                    # 尝试渲染HTML表格
                    if '<table' in result['document'].lower():
                        st.markdown(result['document'], unsafe_allow_html=True)
                    else:
                        st.write(result['document'])
            
            with col_info:
                # 显示表格索引和其他信息
                if metadata.get('table_index') is not None:
                    st.markdown(f"**表格索引:** {metadata['table_index']}")
                if metadata.get('chunk_strategy'):
                    st.markdown(f"**切片策略:** {metadata['chunk_strategy']}")
        
        elif source_type == 'text':
            # 显示文本内容
            if result['document']:
                st.markdown("**文本内容:**")
                st.write(result['document'])
        
        else:
            # 显示通用内容
            if result['document']:
                st.markdown("**内容:**")
                st.write(result['document'])
            
            # 如果有图像内容，也显示
            if metadata.get('has_original_content', False):
                try:
                    original_content = metadata.get('original_content')
                    if original_content:
                        import base64
                        import io
                        image_data = base64.b64decode(original_content)
                        image = Image.open(io.BytesIO(image_data))
                        st.image(image, caption="相关图像", width=400)
                except Exception as e:
                    st.error(f"显示图像失败: {e}")
        
        # 显示详细元数据（可折叠）
        with st.expander("🔍 查看详细元数据", expanded=False):
            for key, value in metadata.items():
                if key not in ['original_content']:  # 不显示Base64内容
                    st.write(f"**{key}:** {value}")

def display_search_results(results: List[Dict]):
    """Display search results with images and metadata in a grid layout.
    
    Args:
        results: List of search result dictionaries.
    """
    if not results:
        st.info("没有找到相关结果")
        return
        
    st.write(f"找到 {len(results)} 个相关结果：")
    
    # Display results in a grid layout (3 columns per row)
    cols_per_row = 3
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            if i + j < len(results):
                result = results[i + j]
                metadata = result['metadata']
                
                with cols[j]:
                    # Display image from ChromaDB
                    image_displayed = False
                    
                    # 从ChromaDB获取原始内容
                    if metadata.get('has_original_content', False):
                        try:
                            # 从当前结果的元数据中获取原始内容
                            original_content = metadata.get('original_content')
                            if original_content:
                                import base64
                                import io
                                # 解码Base64图片
                                image_data = base64.b64decode(original_content)
                                image = Image.open(io.BytesIO(image_data))
                                st.image(
                                    image, 
                                    caption=f"相似度: {1-result['distance']:.3f}",
                                    width=300
                                )
                                image_displayed = True
                        except Exception as e:
                            st.error(f"从数据库显示图片失败: {e}")
                    
                    # 如果从数据库获取失败，显示占位符
                    if not image_displayed:
                        st.info("图片不可用")
                    
                    # Display basic info in a compact format
                    with st.expander(f"详细信息 #{i+j+1}", expanded=False):
                        st.write(f"**来源:** {metadata.get('source_file', '未知')}")
                        st.write(f"**内容:** {result['document']}")
                        st.write(f"**相似度分数:** {1-result['distance']:.4f}")
                        st.write(f"**距离分数:** {result['distance']:.4f}")
                        
                        if metadata:
                            st.write("**元数据:**")
                            for key, value in metadata.items():
                                if key not in ['original_content']:  # Don't show Base64 content
                                    st.write(f"- {key}: {value}")

def display_duplicate_confirmation(duplicate_results: Dict[str, Any]) -> bool:
    """显示重复检测结果和覆盖确认界面
    
    Args:
        duplicate_results: 重复检测结果
        
    Returns:
        True if user confirms overwrite, False otherwise
    """
    duplicates = duplicate_results.get('duplicates', [])
    new_items = duplicate_results.get('new_items', [])
    
    if not duplicates and not new_items:
        st.info("没有检测到任何内容")
        return False
    
    st.warning(f"检测到 {len(duplicates)} 个重复项和 {len(new_items)} 个新项目")
    
    if duplicates:
        st.subheader("🔄 重复项目")
        st.write("以下项目已存在于数据库中：")
        
        for i, dup in enumerate(duplicates):
            with st.expander(f"重复项 {i+1}: {dup['content'][:50]}...", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**新上传内容:**")
                    st.write(f"内容: {dup['content']}")
                    if dup['metadata']:
                        st.write("元数据:")
                        for key, value in dup['metadata'].items():
                            if key not in ['original_content']:
                                st.write(f"- {key}: {value}")
                
                with col2:
                    st.write("**数据库中已存在:**")
                    if 'existing_doc' in dup and dup['existing_doc']:
                        existing = dup['existing_doc']
                        st.write(f"内容: {existing['document']}")
                        st.write(f"ID: {existing['id']}")
                        if existing['metadata']:
                            st.write("元数据:")
                            for key, value in existing['metadata'].items():
                                if key not in ['original_content']:
                                    st.write(f"- {key}: {value}")
                    else:
                        st.write("无法获取已存在文档的详细信息")
                        st.write(f"重复项ID: {dup.get('existing_id', '未知')}")
    
    if new_items:
        st.subheader("✨ 新项目")
        st.write(f"以下 {len(new_items)} 个项目将被添加到数据库：")
        
        for i, item in enumerate(new_items):
            with st.expander(f"新项目 {i+1}: {item['content'][:50]}...", expanded=False):
                st.write(f"内容: {item['content']}")
                if item['metadata']:
                    st.write("元数据:")
                    for key, value in item['metadata'].items():
                        if key not in ['original_content']:
                            st.write(f"- {key}: {value}")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 覆盖重复项并添加新项目", type="primary"):
            return True
    
    with col2:
        if st.button("➕ 仅添加新项目"):
            return "new_only"
    
    with col3:
        if st.button("❌ 取消上传"):
            st.session_state.show_duplicate_confirmation = False
            st.session_state.duplicate_check_results = None
            st.session_state.pending_uploads = None
            st.rerun()
    
    return False

def initialize_session_state():
    """初始化Streamlit会话状态变量"""
    if 'config' not in st.session_state:
        st.session_state.config = VisionRAGConfig()
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = None
    if 'embedding_tool' not in st.session_state:
        st.session_state.embedding_tool = None
    if 'answer_agent' not in st.session_state:
        st.session_state.answer_agent = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'num_results' not in st.session_state:
        st.session_state.num_results = st.session_state.config.default_num_results
    if 'similarity_threshold' not in st.session_state:
        st.session_state.similarity_threshold = st.session_state.config.default_similarity_threshold
    if 'use_similarity_threshold' not in st.session_state:
        st.session_state.use_similarity_threshold = False
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'pending_uploads' not in st.session_state:
        st.session_state.pending_uploads = None
    if 'duplicate_check_results' not in st.session_state:
        st.session_state.duplicate_check_results = None
    if 'show_duplicate_confirmation' not in st.session_state:
        st.session_state.show_duplicate_confirmation = False

def setup_sidebar():
    """设置侧边栏UI并返回设置"""
    with st.sidebar:
        st.header("🔧 设置")
        
        # Configuration management
        st.subheader("配置管理")
        config = st.session_state.config
        
        # Database settings
        st.subheader("数据库设置")
        db_path = st.text_input(
            "数据库路径", 
            value=config.db_path,
            help="向量数据库存储路径，建议使用持久化目录"
        )
        collection_name = st.text_input(
            "集合名称", 
            value=config.collection_name,
            help="数据库集合名称，不同项目建议使用不同名称"
        )
        
        # Update config
        config.db_path = db_path
        config.collection_name = collection_name
        
        st.markdown("---")
        
        # Search settings
        st.subheader("搜索设置")
        num_results = st.slider(
            "返回结果数量",
            min_value=1,
            max_value=config.max_num_results,
            value=st.session_state.num_results,
            help="设置检索返回的相关图片数量"
        )
        st.session_state.num_results = num_results
        
        # Similarity threshold settings
        use_similarity_threshold = st.checkbox(
            "启用相似度阈值过滤",
            value=st.session_state.use_similarity_threshold,
            help="启用后，只返回相似度高于设定阈值的结果"
        )
        st.session_state.use_similarity_threshold = use_similarity_threshold
        
        if use_similarity_threshold:
            similarity_threshold = st.slider(
                "相似度阈值",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.similarity_threshold,
                step=0.05,
                help="设置相似度阈值（0-1），只返回相似度高于此值的结果"
            )
            st.session_state.similarity_threshold = similarity_threshold
        
        # Auto expand settings
        st.subheader("查询扩写设置")
        enable_auto_expand = st.checkbox(
            "启用自动扩写",
            value=config.enable_auto_expand,
            help="当查询少于指定字数时，自动扩写查询以提高检索效果"
        )
        config.enable_auto_expand = enable_auto_expand
        
        if enable_auto_expand:
            auto_expand_min_length = st.number_input(
                "自动扩写最低字数",
                min_value=10,
                max_value=200,
                value=config.auto_expand_min_length,
                step=10,
                help="当查询字数少于此值时，将触发自动扩写"
            )
            config.auto_expand_min_length = auto_expand_min_length
        
        # Advanced options
        if st.checkbox("显示高级选项", value=config.show_advanced_options):
            config.show_advanced_options = True
            
            st.subheader("文档解析配置")
            config.output_dir = st.text_input(
                "输出目录", 
                value=config.output_dir,
                help="文档解析结果的输出目录"
            )
            config.cls_dir = st.text_input(
                "分类目录", 
                value=config.cls_dir,
                help="知识库分类目录名称"
            )
            config.lang = st.selectbox(
                "解析语言",
                options=["ch", "en"],
                index=0 if config.lang == "ch" else 1,
                help="文档解析语言"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                config.enable_formula = st.checkbox(
                    "启用公式解析", 
                    value=config.enable_formula,
                    help="适用于学术文档"
                )
            with col2:
                config.enable_table = st.checkbox(
                    "启用表格解析", 
                    value=config.enable_table,
                    help="适用于包含表格的文档"
                )
            with col3:
                config.auto_caption = st.checkbox(
                    "自动图片描述", 
                    value=config.auto_caption,
                    help="提高检索准确性"
                )
            
            st.subheader("切片配置")
            config.max_chunk_size = st.number_input(
                "最大切片大小",
                min_value=100,
                max_value=5000,
                value=config.max_chunk_size,
                step=100,
                help="单个文本切片的最大字符数"
            )
            config.min_chunk_size = st.number_input(
                "最小切片大小",
                min_value=50,
                max_value=1000,
                value=config.min_chunk_size,
                step=50,
                help="单个文本切片的最小字符数"
            )
            config.overlap_size = st.number_input(
                "重叠大小",
                min_value=0,
                max_value=500,
                value=config.overlap_size,
                step=50,
                help="相邻切片的重叠字符数"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                config.preserve_sentences = st.checkbox(
                    "保持句子完整性", 
                    value=config.preserve_sentences
                )
            with col2:
                config.preserve_paragraphs = st.checkbox(
                    "保持段落完整性", 
                    value=config.preserve_paragraphs
                )
        else:
            config.show_advanced_options = False
        
        st.markdown("---")
        
        # Database info
        if st.session_state.db_manager:
            st.subheader("数据库信息")
            info = st.session_state.db_manager.get_collection_info()
            st.write(f"集合名称: {info.get('name', 'N/A')}")
            st.write(f"切片数量: {info.get('count', 0)}")
            
            if st.button("清空数据库", type="secondary"):
                if st.session_state.db_manager.clear_collection():
                    st.success("数据库已清空")
                    st.rerun()
                else:
                    st.error("清空数据库失败")
    
    return db_path, collection_name

def initialize_tools(db_path: str, collection_name: str):
    """初始化嵌入工具、数据库管理器和问答助手"""
    if not TongyiEmbedding or not ChromaDBManager or not AnswerAgent:
        st.error("无法导入必要的工具类，请检查依赖")
        return False
        
    try:
        # Initialize embedding tool
        if st.session_state.embedding_tool is None:
            st.session_state.embedding_tool = TongyiEmbedding()
            st.sidebar.success("通义嵌入工具初始化成功！")
        
        # Initialize database manager
        if st.session_state.db_manager is None:
            st.session_state.db_manager = ChromaDBManager(
                embedding_tool=st.session_state.embedding_tool,
                db_path=db_path,
                collection_name=collection_name
            )
            st.sidebar.success("ChromaDB管理器初始化成功！")
        
        # Initialize answer agent
        if st.session_state.answer_agent is None:
            st.session_state.answer_agent = AnswerAgent()
            st.sidebar.success("问答助手初始化成功！")
        
        return True
        
    except Exception as e:
        st.sidebar.error(f"初始化失败：{e}")
        return False

@dataclass
class VisionRAGConfig:
    """Vision RAG系统配置类"""
    
    # 数据解析配置
    output_dir: str = "data/output"  # 解析输出目录
    cls_dir: str = "default"  # 知识库分类目录
    lang: str = "ch"  # 解析语言 ("ch" 中文, "en" 英文)
    enable_formula: bool = True  # 启用公式解析
    enable_table: bool = True  # 启用表格解析
    auto_caption: bool = True  # 自动生成图片描述
    
    # 切片配置
    max_chunk_size: int = 1000  # 最大切片大小（字符数）
    min_chunk_size: int = 100   # 最小切片大小（字符数）
    overlap_size: int = 100     # 重叠大小（字符数）
    preserve_sentences: bool = True  # 保持句子完整性
    preserve_paragraphs: bool = True  # 保持段落完整性
    
    # 向量数据库配置
    db_path: str = "./chromadb_data"  # 数据库存储路径
    collection_name: str = "vision_rag_documents"  # 集合名称
    
    # 检索配置
    default_num_results: int = 5  # 默认检索结果数量
    max_num_results: int = 20  # 最大检索结果数量
    default_similarity_threshold: float = 0.7  # 默认相似度阈值
    enable_query_expansion: bool = True  # 启用查询扩展
    
    # 自动扩写配置
    enable_auto_expand: bool = False  # 启用自动扩写
    auto_expand_min_length: int = 50  # 自动扩写最低字数
    
    # UI配置
    show_advanced_options: bool = False  # 显示高级选项
    show_debug_info: bool = False  # 显示调试信息
    
    @classmethod
    def get_config_descriptions(cls) -> Dict[str, str]:
        """获取配置项说明"""
        return {
            "output_dir": "文档解析结果的输出目录，建议使用相对路径",
            "cls_dir": "知识库分类目录名称，用于组织不同类型的文档",
            "lang": "文档解析语言，支持中文(ch)和英文(en)",
            "enable_formula": "是否启用数学公式解析，适用于学术文档",
            "enable_table": "是否启用表格解析，适用于包含表格的文档",
            "auto_caption": "是否自动为图片生成描述，提高检索准确性",
            "max_chunk_size": "单个文本切片的最大字符数，影响检索粒度",
            "min_chunk_size": "单个文本切片的最小字符数，避免过小的片段",
            "overlap_size": "相邻切片的重叠字符数，保持上下文连贯性",
            "preserve_sentences": "切片时保持句子完整性，提高语义连贯性",
            "preserve_paragraphs": "切片时保持段落完整性，适合结构化文档",
            "db_path": "向量数据库存储路径，建议使用持久化目录",
            "collection_name": "数据库集合名称，不同项目建议使用不同名称",
            "default_num_results": "默认检索返回的结果数量",
            "max_num_results": "最大检索结果数量限制",
            "default_similarity_threshold": "默认相似度阈值，过滤低相关性结果",
            "enable_query_expansion": "启用查询扩展，提高检索召回率",
            "enable_auto_expand": "启用自动扩写功能，当查询少于指定字数时自动扩写",
            "auto_expand_min_length": "自动扩写的最低字数要求，少于此字数将触发扩写",
            "show_advanced_options": "在界面中显示高级配置选项",
            "show_debug_info": "显示调试信息，用于开发和调试"
        }

def main():
    """主函数"""
    st.set_page_config(
        page_title="Vision RAG with Qwen",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Vision RAG with Qwen")
    st.markdown("基于通义千问和ChromaDB的视觉检索增强生成系统")
    
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar
    db_path, collection_name = setup_sidebar()
    
    # Initialize tools
    if not initialize_tools(db_path, collection_name):
        st.stop()
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📤 上传文档", "🔍 搜索查询", "📚 知识库管理"])
    
    with tab1:
        st.header("上传图片或PDF文档")
        
        # 显示重复检测确认界面
        # 切片入库确认界面
        if st.session_state.show_duplicate_confirmation and st.session_state.duplicate_check_results:
            st.subheader("🔍 切片入库检测结果")
            
            confirmation_result = display_duplicate_confirmation(st.session_state.duplicate_check_results)
            
            if confirmation_result == True:  # 覆盖重复项并添加新项目
                # 创建进度容器
                progress_container = st.container()
                
                with progress_container:
                    # 进度条
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    
                    try:
                        duplicates = st.session_state.duplicate_check_results.get('duplicates', [])
                        new_items = st.session_state.duplicate_check_results.get('new_items', [])
                        all_items = duplicates + new_items
                        
                        status_text.text(f"准备覆盖存储 {len(all_items)} 个切片到数据库...")
                        progress_bar.progress(0.1)
                        
                        # 直接使用已解析的切片数据进行批量添加（覆盖模式）
                        if all_items:
                            with st.spinner(f"💾 正在覆盖存储 {len(all_items)} 个切片到数据库"):
                                progress_bar.progress(0.5)
                                result = st.session_state.db_manager.batch_add_with_overwrite(
                                    all_items, force_overwrite=True
                                )
                                progress_bar.progress(0.9)
                            
                            success_count = len(result.get('successful_ids', []))
                            progress_bar.progress(1.0)
                            status_text.text("覆盖存储完成！")
                            
                            st.success(f"🎉 覆盖存储完成！成功处理 {success_count} 个切片")
                            
                            if success_count > 0:
                                st.balloons()
                        else:
                            progress_bar.progress(1.0)
                            status_text.text("没有切片需要处理")
                            st.info("ℹ️ 没有切片需要处理")
                            
                    except Exception as e:
                        st.error(f"❌ 覆盖存储切片时出错：{e}")
                    finally:
                        # 清理进度显示
                        progress_bar.empty()
                        status_text.empty()
                    
                    # 清理状态
                    st.session_state.show_duplicate_confirmation = False
                    st.session_state.duplicate_check_results = None
                    st.session_state.parsed_chunks = None
                    st.rerun()
                    
            elif confirmation_result == "new_only":  # 仅添加新项目
                # 创建进度容器
                progress_container = st.container()
                
                with progress_container:
                    # 进度条
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    
                    try:
                        duplicates = st.session_state.duplicate_check_results.get('duplicates', [])
                        new_items = st.session_state.duplicate_check_results.get('new_items', [])
                        
                        status_text.text(f"准备添加 {len(new_items)} 个新切片...")
                        progress_bar.progress(0.2)
                        
                        # 使用批量添加方法处理新项目
                        if new_items:
                            with st.spinner(f"💾 正在添加 {len(new_items)} 个新切片到数据库"):
                                progress_bar.progress(0.5)
                                result = st.session_state.db_manager.batch_add_with_overwrite(
                                    new_items, force_overwrite=False
                                )
                                progress_bar.progress(0.8)
                            
                            success_count = len(result.get('successful_ids', []))
                            progress_bar.progress(1.0)
                            status_text.text("新项目添加完成！")
                            
                            st.success(f"✅ 成功添加 {success_count} 个新切片")
                            
                            if success_count > 0:
                                st.balloons()
                        else:
                            progress_bar.progress(1.0)
                            status_text.text("没有新项目需要添加")
                            st.info("ℹ️ 没有新项目需要添加")
                            
                    except Exception as e:
                        st.error(f"❌ 添加新项目时出错：{e}")
                    finally:
                        # 清理进度显示
                        progress_bar.empty()
                        status_text.empty()
                    
                    # 清理状态
                    st.session_state.show_duplicate_confirmation = False
                    st.session_state.duplicate_check_results = None
                    st.session_state.parsed_chunks = None
                    st.rerun()
        
        else:
            # 统一的文档上传界面
            st.subheader("📁 文档上传与管理")
            
            # 文件上传界面
            uploaded_files = st.file_uploader(
                "选择文件",
                type=['png', 'jpg', 'jpeg', 'pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls'],
                accept_multiple_files=True,
                help="支持图片(PNG、JPG、JPEG)、PDF文档和Office文档(Word、PowerPoint、Excel)",
                key="upload_files"
            )
            
            if uploaded_files:
                # 自动检查文档状态
                with st.spinner("正在检查文档状态..."):
                    parsed_files = []
                    unparsed_files = []
                    
                    for file in uploaded_files:
                        # 检查数据库中是否存在该文件的切片
                        try:
                            # 使用文件名查询数据库
                            query_result = st.session_state.db_manager.collection.get(
                                where={"source_file": file.name}
                            )
                            
                            if query_result and len(query_result.get('ids', [])) > 0:
                                chunk_count = len(query_result['ids'])
                                parsed_files.append({
                                    'name': file.name,
                                    'chunk_count': chunk_count,
                                    'file_obj': file
                                })
                            else:
                                unparsed_files.append({
                                    'name': file.name,
                                    'file_obj': file
                                })
                        except Exception as e:
                            st.error(f"检查文件 {file.name} 时出错：{e}")
                            unparsed_files.append({
                                'name': file.name,
                                'file_obj': file
                            })
                
                # 显示检查结果
                if parsed_files:
                    st.warning(f"⚠️ 发现 {len(parsed_files)} 个已解析的文档：")
                    for file_info in parsed_files:
                        st.info(f"📄 {file_info['name']} - 已有 {file_info['chunk_count']} 个切片")
                
                if unparsed_files:
                    st.success(f"✅ 发现 {len(unparsed_files)} 个未解析的文档：")
                    for file_info in unparsed_files:
                        st.info(f"📄 {file_info['name']} - 新文档")
                
                st.markdown("---")
                
                # 处理选项
                if parsed_files:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("✅ 跳过已解析文档", type="primary"):
                            # 检查已解析文档的切片是否都已入库
                            with st.spinner("🔍 检查文档切片入库状态..."):
                                all_chunks_stored = True
                                missing_chunks = []
                                
                                for file_info in parsed_files:
                                    file_name = file_info['name']
                                    try:
                                        # 检查该文件的切片数量
                                        query_result = st.session_state.db_manager.collection.get(
                                            where={"source_file": file_name}
                                        )
                                        stored_count = len(query_result.get('ids', []))
                                        expected_count = file_info['chunk_count']
                                        
                                        if stored_count < expected_count:
                                            all_chunks_stored = False
                                            missing_chunks.append({
                                                'file': file_name,
                                                'stored': stored_count,
                                                'expected': expected_count
                                            })
                                    except Exception as e:
                                        st.error(f"检查文件 {file_name} 时出错：{e}")
                                        all_chunks_stored = False
                                
                                if all_chunks_stored:
                                    st.success("✅ 所有文档切片已入库，可直接使用！请切换到'智能图像问答'标签页开始提问")
                                    st.balloons()
                                else:
                                    st.warning(f"⚠️ 发现 {len(missing_chunks)} 个文档的切片未完全入库")
                                    for chunk_info in missing_chunks:
                                        st.info(f"📄 {chunk_info['file']}: 已入库 {chunk_info['stored']}/{chunk_info['expected']} 个切片")
                                    
                                    if st.button("📥 补充入库缺失切片", type="secondary"):
                                        st.info("请重新解析相关文档以补充缺失的切片")
                                        st.session_state.files_to_process = [f['file_obj'] for f in parsed_files if f['name'] in [c['file'] for c in missing_chunks]]
                                        st.session_state.processing_mode = 'reparse'
                                        st.rerun()
                    
                    with col2:
                        if st.button("🔄 重新解析已存在的文档", type="secondary"):
                            st.session_state.files_to_process = [f['file_obj'] for f in parsed_files]
                            st.session_state.processing_mode = 'reparse'
                            st.rerun()
                    
                    with col3:
                        if unparsed_files and st.button("📤 解析新文档", type="secondary"):
                            st.session_state.files_to_process = [f['file_obj'] for f in unparsed_files]
                            st.session_state.processing_mode = 'new'
                            st.rerun()
                else:
                    # 如果没有已解析文档，使用原来的两列布局
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        pass  # 空列
                    
                    with col2:
                        if unparsed_files and st.button("📤 解析新文档", type="primary"):
                            st.session_state.files_to_process = [f['file_obj'] for f in unparsed_files]
                            st.session_state.processing_mode = 'new'
                            st.rerun()
                
                # 如果两种文档都有，提供全部处理选项
                if parsed_files and unparsed_files:
                    st.markdown("---")
                    if st.button("🔄 处理所有文档（重新解析已存在的 + 解析新文档）", type="primary"):
                        st.session_state.files_to_process = uploaded_files
                        st.session_state.processing_mode = 'all'
                        st.rerun()
            
            # 处理文件解析
            if st.session_state.get('files_to_process') and st.session_state.get('processing_mode'):
                files_to_process = st.session_state.files_to_process
                processing_mode = st.session_state.processing_mode
                
                # 清理状态
                st.session_state.files_to_process = None
                st.session_state.processing_mode = None
                
                # 确定处理模式
                if processing_mode == 'reparse':
                    mode_text = "重新解析"
                    is_reparse_mode = True
                elif processing_mode == 'new':
                    mode_text = "解析"
                    is_reparse_mode = False
                else:  # 'all'
                     mode_text = "处理"
                     is_reparse_mode = False  # 混合模式，在处理时单独判断
                
                # 创建进度容器
                progress_container = st.container()
                
                with progress_container:
                    # 总体进度条
                    overall_progress = st.progress(0.0)
                    status_text = st.empty()
                    
                    try:
                        total_files = len(files_to_process)
                        status_text.text(f"开始{mode_text} {total_files} 个文件...")
                        
                        all_duplicate_results = {'duplicates': [], 'new_items': []}
                        
                        for file_idx, uploaded_file in enumerate(files_to_process):
                            # 更新总体进度
                            current_progress = file_idx / total_files
                            overall_progress.progress(current_progress)
                            status_text.text(f"正在{mode_text}文件 {file_idx + 1}/{total_files}: {uploaded_file.name}")
                            
                            # 对于混合模式，需要检查当前文件是否已解析过
                            current_file_is_reparse = is_reparse_mode
                            if processing_mode == 'all':
                                # 检查当前文件是否已解析过
                                try:
                                    query_result = st.session_state.db_manager.collection.get(
                                        where={"source_file": uploaded_file.name}
                                    )
                                    current_file_is_reparse = query_result and len(query_result.get('ids', [])) > 0
                                except:
                                    current_file_is_reparse = False
                            
                            # 使用新的工具类处理文档
                            try:
                                # 步骤1: 保存临时文件
                                with st.spinner(f"📁 保存临时文件: {uploaded_file.name}"):
                                    temp_file_path = f"/tmp/{uploaded_file.name}"
                                    with open(temp_file_path, "wb") as f:
                                        f.write(uploaded_file.getvalue())
                                
                                # 步骤2: 解析文档
                                with st.spinner(f"🔍 解析文档内容: {uploaded_file.name}"):
                                    chunks = process_document_with_agents(temp_file_path, st.session_state.config)
                                
                                if chunks:
                                    # 步骤3: 处理切片和检查重复
                                    with st.spinner(f"✂️ 处理文档切片: 发现 {len(chunks)} 个切片"):
                                        for chunk in chunks:
                                            # 检查是否为重复切片（基于内容和文件名）
                                            chunk_data = {
                                                'content': chunk.content,
                                                'metadata': {
                                                    'source_file': uploaded_file.name,
                                                    'chunk_type': chunk.chunk_type.value,
                                                    'chunk_id': chunk.chunk_id,
                                                    'page_idx': chunk.page_idx,
                                                    'chunk_idx': chunk.chunk_idx,
                                                    'parent_document': chunk.parent_document,
                                                    **chunk.metadata  # 包含所有原始元数据
                                                }
                                            }
                                            
                                            # 为图像和表格类切片添加Base64数据
                                            if chunk.chunk_type.value in ['image', 'table'] and chunk.content_path:
                                                try:
                                                    import base64
                                                    with open(chunk.content_path, 'rb') as f:
                                                        image_data = base64.b64encode(f.read()).decode('utf-8')
                                                        chunk_data['original_content'] = image_data
                                                except Exception as e:
                                                    print(f"读取图片文件失败 {chunk.content_path}: {e}")
                                                    # 如果读取失败，尝试使用相对路径
                                                    try:
                                                        image_path = chunk.metadata.get('image_path') or chunk.metadata.get('table_image_path')
                                                        if image_path:
                                                            # 根据source_file动态构建base_dir，并尝试不同的处理方法
                                                            source_file = chunk.source_file or uploaded_file.name.split('.')[0]
                                                            config = st.session_state.config
                                                            
                                                            # 尝试不同的处理方法：auto, ocr
                                                            for method in ['auto', 'ocr']:
                                                                base_dir = os.path.join(config.output_dir, config.cls_dir, source_file, source_file, method)
                                                                full_path = os.path.join(base_dir, image_path)
                                                                if os.path.exists(full_path):
                                                                    with open(full_path, 'rb') as f:
                                                                        image_data = base64.b64encode(f.read()).decode('utf-8')
                                                                        chunk_data['original_content'] = image_data
                                                                    break
                                                            else:
                                                                # 如果都找不到，记录错误
                                                                print(f"在所有处理方法目录中都找不到图片文件: {image_path}")
                                                    except Exception as e2:
                                                        print(f"使用相对路径读取图片文件也失败: {e2}")
                                                        print(f"尝试的路径: {full_path if 'full_path' in locals() else '路径构建失败'}")
                                            
                                            # 如果是重新解析模式或当前文件已解析过，所有切片都视为需要覆盖的重复项
                                            if current_file_is_reparse:
                                                # 为重新解析的切片添加existing_doc字段
                                                try:
                                                    existing = st.session_state.db_manager.collection.get(
                                                        where={
                                                            "$and": [
                                                                {"source_file": uploaded_file.name},
                                                                {"chunk_id": chunk.chunk_id}
                                                            ]
                                                        }
                                                    )
                                                    if existing and len(existing.get('ids', [])) > 0:
                                                        # 构造existing_doc结构
                                                        existing_doc = {
                                                            'id': existing['ids'][0],
                                                            'document': existing['documents'][0],
                                                            'metadata': existing['metadatas'][0]
                                                        }
                                                        chunk_data['existing_doc'] = existing_doc
                                                except:
                                                    # 如果查询失败，创建一个默认的existing_doc
                                                    chunk_data['existing_doc'] = {
                                                        'id': 'unknown',
                                                        'document': '无法获取已存在文档信息',
                                                        'metadata': {}
                                                    }
                                                all_duplicate_results['duplicates'].append(chunk_data)
                                            else:
                                                # 正常模式下检查数据库中是否存在
                                                try:
                                                    existing = st.session_state.db_manager.collection.get(
                                                        where={
                                                            "$and": [
                                                                {"source_file": uploaded_file.name},
                                                                {"chunk_id": chunk.chunk_id}
                                                            ]
                                                        }
                                                    )
                                                    if existing and len(existing.get('ids', [])) > 0:
                                                        # 构造existing_doc结构
                                                        existing_doc = {
                                                            'id': existing['ids'][0],
                                                            'document': existing['documents'][0],
                                                            'metadata': existing['metadatas'][0]
                                                        }
                                                        chunk_data['existing_doc'] = existing_doc
                                                        all_duplicate_results['duplicates'].append(chunk_data)
                                                    else:
                                                        all_duplicate_results['new_items'].append(chunk_data)
                                                except:
                                                    # 如果查询失败，默认为新项目
                                                    all_duplicate_results['new_items'].append(chunk_data)
                                    
                                    file_mode = "重新解析" if current_file_is_reparse else "解析"
                                    st.success(f"✅ {uploaded_file.name} {file_mode}完成，生成 {len(chunks)} 个切片")
                                else:
                                    st.warning(f"⚠️ {uploaded_file.name} 未能生成有效切片")
                                
                                # 清理临时文件
                                if os.path.exists(temp_file_path):
                                    os.remove(temp_file_path)
                                    
                            except Exception as e:
                                st.error(f"❌ {mode_text}文件 {uploaded_file.name} 时出错：{e}")
                                continue
                        
                        # 完成处理
                        overall_progress.progress(1.0)
                        status_text.text(f"文件{mode_text}完成！")
                        
                        # 检查是否有重复项或新项目
                        has_duplicates = len(all_duplicate_results['duplicates']) > 0
                        has_new_items = len(all_duplicate_results['new_items']) > 0
                        
                        if has_duplicates or has_new_items:
                            duplicate_count = len(all_duplicate_results['duplicates'])
                            new_count = len(all_duplicate_results['new_items'])
                            
                            if processing_mode == 'reparse':
                                st.info(f"📊 重新解析结果：发现 {duplicate_count} 个切片将被覆盖")
                            elif processing_mode == 'new':
                                st.info(f"📊 解析结果：发现 {new_count} 个新切片")
                            else:  # 'all'
                                st.info(f"📊 处理结果：发现 {duplicate_count} 个重复切片，{new_count} 个新切片")
                            
                            # 显示切片入库确认界面
                            st.session_state.duplicate_check_results = all_duplicate_results
                            st.session_state.show_duplicate_confirmation = True
                            st.rerun()
                        else:
                            st.warning("⚠️ 没有检测到任何有效内容")
                            
                    except Exception as e:
                        st.error(f"❌ {mode_text}文件时出错：{e}")
                    finally:
                        # 清理进度显示
                        overall_progress.empty()
                        status_text.empty()
    
    with tab2:
        st.header("🔍 搜索查询")
        
        # Check if database has documents
        if st.session_state.db_manager:
            info = st.session_state.db_manager.get_collection_info()
            if info.get('count', 0) == 0:
                st.info("数据库中暂无文档，请先上传图片或PDF文件")
                return
        
        # 智能问答界面（合并了图像检索和问答功能）
            st.subheader("智能问答")
            st.markdown("💡 **使用说明：** 输入您的问题，系统会自动检索相关资料并基于内容生成回答")
        
            # 问答输入区域
            user_question = st.text_area(
                "请输入您的问题",
                placeholder="例如：什么是深度学习？神经网络的结构是怎样的？这个数据可视化图表说明了什么？",
                help="输入您想了解的问题，系统会检索相关资料并生成回答",
                height=120
            )
            
            if user_question and user_question.strip():
                if st.button("🔍 搜索并问答", type="primary"):
                    with st.spinner("正在检索相关资料..."):
                        try:
                            # 获取配置
                            config = st.session_state.config
                            
                            # 检查是否需要自动扩写
                            original_question = user_question
                            expanded_question = user_question
                            auto_expanded = False
                            
                            if config.enable_auto_expand and len(user_question) < config.auto_expand_min_length:
                                st.info(f"🔄 检测到查询字数少于{config.auto_expand_min_length}字，正在自动扩写...")
                                try:
                                    # 导入查询扩写代理
                                    from agent.query_agent import QueryExpansionAgent
                                    query_agent = QueryExpansionAgent()
                                    expanded_question = query_agent.auto_expand_query_sync(user_question, config.auto_expand_min_length)
                                    auto_expanded = True
                                    
                                    # 显示扩写结果
                                    with st.expander("📝 查询自动扩写结果", expanded=True):
                                        st.markdown(f"**原始问题：** {original_question}")
                                        st.markdown(f"**自动扩写：** {expanded_question}")
                                        st.info("💡 扩写后的查询将用于检索，但AI回答时会更关注原始问题")
                                        
                                except Exception as e:
                                    st.warning(f"自动扩写失败，使用原始查询：{e}")
                                    expanded_question = user_question
                            
                            # 第一步：基于问题检索相关资料
                            threshold = st.session_state.similarity_threshold if st.session_state.use_similarity_threshold else None
                            
                            search_results = search_similar_images(
                                expanded_question, 
                                st.session_state.db_manager, 
                                st.session_state.num_results,
                                threshold
                            )
                            
                            if not search_results:
                                if st.session_state.use_similarity_threshold:
                                    st.warning(f"没有找到相似度高于 {st.session_state.similarity_threshold:.2f} 的相关资料，请尝试降低阈值或使用其他关键词")
                                else:
                                    st.warning("没有找到与您的问题相关的资料，请尝试其他问题")
                                return
                            
                            # 显示检索结果
                            st.subheader(f"📚 检索到 {len(search_results)} 个相关资料")
                            if st.session_state.use_similarity_threshold:
                                st.info(f"已启用相似度阈值过滤 (≥{st.session_state.similarity_threshold:.2f})")
                            
                            # 显示每个检索结果的详细内容
                            st.markdown("### 🔍 检索结果")
                            for i, result in enumerate(search_results):
                                display_chunk_content(result, i)
                            
                            # 第二步：基于问题和检索到的资料生成回答
                            st.markdown("### 🤖 AI 分析与回答")
                            with st.spinner("AI正在分析资料并生成回答..."):
                                import asyncio
                                
                                async def get_answer():
                                    # 构建传递给答案代理的查询信息
                                    if auto_expanded:
                                        query_for_answer = f"自动扩写: {expanded_question}\n\n原始问题: {original_question}"
                                    else:
                                        query_for_answer = original_question
                                    
                                    return await st.session_state.answer_agent.answer_question(
                                        query_for_answer, 
                                        search_results
                                    )
                                
                                # 在Streamlit中运行异步函数
                                answer_result = asyncio.run(get_answer())
                                
                                if answer_result.get("error"):
                                    st.error(f"问答过程中出错：{answer_result['error']}")
                                else:
                                    # 显示AI回答
                                    st.markdown("#### 💬 AI 回答")
                                    st.write(answer_result.get("answer", "无法生成回答"))
                                    
                                    # 显示使用统计
                                    usage = answer_result.get("usage_metadata", {})
                                    if usage:
                                        with st.expander("📊 Token 使用统计", expanded=False):
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("输入Token", usage.get("input_tokens", 0))
                                            with col2:
                                                st.metric("输出Token", usage.get("output_tokens", 0))
                                            with col3:
                                                st.metric("总Token", usage.get("total_tokens", 0))
                                
                        except Exception as e:
                            st.error(f"执行问答时出错：{e}")
            
            # 添加使用提示和示例
            with st.expander("💡 使用提示和示例", expanded=False):
                st.markdown("""
                ### 💡 如何更好地使用智能问答功能：
                
                **工作流程：**
                1. 输入您的问题
                2. 系统自动检索相关图片
                3. AI基于问题和图片生成回答
                
                **提问技巧：**
                - 提出具体、明确的问题
                - 可以询问概念、原理、数据分析等
                - 支持中英文问题
                
                ### 📝 示例问题：
                
                **概念解释类：**
                - 什么是深度学习？
                - 神经网络的基本结构是什么？
                - 卷积神经网络是如何工作的？
                
                **数据分析类：**
                - 这个图表显示了什么趋势？
                - 数据可视化图中的主要信息是什么？
                - 这些统计数据说明了什么问题？
                
                **技术架构类：**
                - 这个系统架构图展示了什么？
                - 技术流程图中的关键步骤是什么？
                - 这个算法的实现原理是什么？
                """)
    
    with tab3:
        display_knowledge_base_management()
    
    # Footer
    st.markdown("---")
    with st.expander("ℹ️ 关于系统"):
        st.markdown("""
        ### Vision RAG with Qwen
        
        本系统基于以下技术构建：
        
        - **通义千问多模态嵌入**: 用于图片和文本的向量化
        - **通义千问视觉语言模型**: 用于图像理解和问答
        - **ChromaDB**: 高性能向量数据库，用于存储和检索
        - **Streamlit**: 用户界面框架
        - **PyMuPDF**: PDF文档处理
        - **LangGraph**: 工作流编排框架
        
        ### 功能特点
        
        - 支持图片和PDF文档上传
        - 自动向量化和存储
        - 基于文本查询的相似图片检索
        - 智能图像问答功能
        - 直观的搜索结果展示
        - 可配置的搜索参数和相似度阈值过滤
        - 灵活的配置管理系统
        - 高级文档解析选项
        - Token使用统计
        
        ### 使用方法
        
        1. **上传文档**: 在"上传文档"标签页中上传图片或PDF文件
        2. **图像搜索**: 在"搜索查询"→"图像搜索"中输入查询文本，检索相似图像
        3. **智能问答**: 在"搜索查询"→"智能问答"中基于检索到的图像进行提问
        4. **查看结果**: 系统会显示详细的回答和使用统计
        
        ### 新增功能：智能问答
        
        - 基于检索到的图像内容进行智能问答
        - 支持多图像综合分析
        - 准确识别图像中的文字和图表
        - 提供详细的技术解释和内容分析
        - 实时显示Token使用情况
        
        ### 配置管理
        
        - 统一的配置类管理所有参数
        - 支持高级文档解析配置
        - 灵活的切片策略设置
        - 可配置的检索参数
        - 界面显示选项控制
        """)

if __name__ == "__main__":
    main()