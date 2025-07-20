import chromadb
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.qwen_embedding import TongyiEmbedding

# 模拟文本数据 - 增加不同领域的文档以提高区分度
documents = [
    # AI/ML相关文档
    "深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的工作方式。深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展，特别是在卷积神经网络和循环神经网络的应用上。",
    "机器学习是人工智能的核心技术，通过算法让计算机从数据中自动学习模式和规律。常见的机器学习方法包括监督学习、无监督学习和强化学习，广泛应用于推荐系统、金融风控和医疗诊断等场景。",
    "自然语言处理（NLP）专注于让计算机理解和生成人类语言。现代NLP技术基于Transformer架构，如BERT、GPT等大语言模型，在文本分类、机器翻译、问答系统等任务上表现出色。",
    
    # 完全不同领域的文档
    "烹饪是一门艺术，需要掌握食材的特性和烹饪技巧。中式烹饪讲究色香味俱全，常用的烹饪方法包括炒、煮、蒸、炸、烤等。不同的食材搭配和调料使用会产生截然不同的口感和风味。",
    "园艺种植需要了解植物的生长习性和环境需求。土壤的酸碱度、光照条件、水分管理都会影响植物的健康生长。春季是播种的好时节，夏季需要注意遮阴和浇水，秋季适合收获和修剪。",
    "古典音乐有着悠久的历史和丰富的表现形式。巴洛克时期的音乐结构严谨，古典主义时期注重平衡与和谐，浪漫主义时期则更加注重情感表达。不同的乐器组合能够创造出独特的音响效果。",
    
    # 边缘相关文档
    "数据科学结合了统计学、计算机科学和领域专业知识，用于从大量数据中提取有价值的信息。数据科学家需要掌握数据清洗、特征工程、模型构建和结果解释等技能。",
    "计算机视觉让机器能够理解和解释视觉信息，包括图像分类、目标检测、语义分割等任务。现代计算机视觉大量使用卷积神经网络，在自动驾驶、医疗影像分析等领域有重要应用。",
    "云计算提供了按需访问的计算资源，包括服务器、存储、数据库和网络服务。主要的服务模式有IaaS、PaaS和SaaS，帮助企业降低IT成本并提高业务灵活性。",
    "区块链是一种分布式账本技术，通过密码学方法确保数据的不可篡改性。区块链技术在数字货币、供应链管理、数字身份认证等领域有广泛应用前景。"
]

# 测试查询 - 设计了多种类型的查询来全面测试相似度计算
test_queries = [
    # 1. 直接关键词匹配测试
    {
        "query": "深度学习模型优化技巧",
        "expected_top": "深度学习",
        "description": "直接关键词匹配 - 测试精确术语识别"
    },
    
    # 2. 语义相似性测试
    {
        "query": "如何让神经网络训练得更快更好？",
        "expected_top": "深度学习",
        "description": "语义相似性测试 - 神经网络应该关联到深度学习"
    },
    
    # 3. 同义词和近义词测试
    {
        "query": "机器学习算法的选择和应用",
        "expected_top": "机器学习",
        "description": "同义词测试 - 算法选择应该关联到机器学习"
    },
    
    # 4. 上下文理解测试
    {
        "query": "我想做一道美味的家常菜，需要注意什么？",
        "expected_top": "烹饪",
        "description": "上下文理解 - 家常菜应该关联到烹饪"
    },
    
    # 5. 领域专业术语测试
    {
        "query": "卷积神经网络和循环神经网络的区别",
        "expected_top": "深度学习",
        "description": "专业术语测试 - CNN/RNN应该关联到深度学习"
    },
    
    # 6. 生活场景描述测试
    {
        "query": "春天适合种植什么花卉？",
        "expected_top": "园艺",
        "description": "生活场景测试 - 种植花卉应该关联到园艺"
    },
    
    # 7. 艺术文化测试
    {
        "query": "巴赫和莫扎特的音乐风格有什么不同？",
        "expected_top": "古典音乐",
        "description": "艺术文化测试 - 音乐家应该关联到古典音乐"
    },
    
    # 8. 技术应用场景测试
    {
        "query": "自动驾驶汽车如何识别道路标志？",
        "expected_top": "计算机视觉",
        "description": "应用场景测试 - 图像识别应该关联到计算机视觉"
    },
    
    # 9. 跨领域干扰测试
    {
        "query": "数据分析在商业决策中的重要性",
        "expected_top": "数据科学",
        "description": "跨领域测试 - 数据分析应该更偏向数据科学"
    },
    
    # 10. 技术实现细节测试
    {
        "query": "如何处理图像中的噪声和模糊？",
        "expected_top": "计算机视觉",
        "description": "技术细节测试 - 图像处理是计算机视觉的核心"
    },
    
    # 11. 抽象概念测试
    {
        "query": "什么是监督学习和无监督学习？",
        "expected_top": "机器学习",
        "description": "抽象概念测试 - 学习类型是机器学习基础概念"
    },
    
    # 12. 实用技能测试
    {
        "query": "如何搭配食材做出营养均衡的菜品？",
        "expected_top": "烹饪",
        "description": "实用技能测试 - 食材搭配应该关联到烹饪"
    }
]

# 初始化embedding工具和ChromaDB
embedding_tool = TongyiEmbedding()
client = chromadb.PersistentClient(path="./chromadb_data")
collection = client.get_or_create_collection(name="docs")

def setup_documents():
    """初始化文档向量并存储到ChromaDB"""
    print("=== 初始化文档库 ===")
    
    # 清空现有集合
    try:
        client.delete_collection(name="docs")
        print("已清空现有文档集合")
    except:
        pass
    
    # 重新创建集合
    global collection
    collection = client.get_or_create_collection(name="docs")
    
    print(f"\n正在为 {len(documents)} 个文档生成向量...")
    success_count = 0
    
    for i, doc in enumerate(documents):
        try:
            result = embedding_tool.get_text_embedding(doc)
            if result.status_code == 200:
                embedding_vector = result.output['embeddings'][0]['embedding']
                
                # 存储到ChromaDB，添加文档摘要作为metadata
                doc_summary = doc[:50] + "..." if len(doc) > 50 else doc
                collection.add(
                    embeddings=[embedding_vector],
                    documents=[doc],
                    metadatas=[{"summary": doc_summary, "doc_id": i}],
                    ids=[f"doc_{i}"]
                )
                success_count += 1
                print(f"   ✓ 文档 {i+1}: {doc_summary}")
            else:
                print(f"   ✗ 文档 {i+1} 向量化失败: {result.message}")
        except Exception as e:
            print(f"   ✗ 文档 {i+1} 处理出错: {str(e)}")
    
    print(f"\n文档初始化完成: {success_count}/{len(documents)} 成功")
    return success_count > 0

def test_single_query(query_info):
    """测试单个查询"""
    query = query_info["query"]
    expected = query_info["expected_top"]
    description = query_info["description"]
    
    print(f"\n{'='*60}")
    print(f"测试查询: {query}")
    print(f"描述: {description}")
    print(f"期望匹配: {expected}")
    print(f"{'='*60}")
    
    try:
        # 生成查询向量
        query_result = embedding_tool.get_text_embedding(query)
        if query_result.status_code != 200:
            print(f"❌ 查询向量化失败: {query_result.message}")
            return False
        
        query_embedding = query_result.output['embeddings'][0]['embedding']
        
        # 执行检索
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5  # 返回前5个结果进行分析
        )
        
        # 分析结果
        print("\n🔍 检索结果分析:")
        documents_found = search_results['documents'][0]
        distances = search_results['distances'][0]
        
        # 计算相似度分数 (1 - distance)
        similarities = [1 - dist for dist in distances]
        
        # 检查是否符合预期
        top_doc = documents_found[0]
        is_correct = expected.lower() in top_doc.lower()
        
        print(f"\n{'✅' if is_correct else '❌'} 排名验证: {'符合预期' if is_correct else '不符合预期'}")
        
        print("\n📊 详细排名:")
        for i, (doc, similarity, distance) in enumerate(zip(documents_found, similarities, distances)):
            # 截取文档前80个字符用于显示
            doc_preview = doc[:80] + "..." if len(doc) > 80 else doc
            
            # 检查是否包含期望关键词
            contains_expected = expected.lower() in doc.lower()
            marker = "🎯" if contains_expected else "  "
            
            print(f"\n{marker} 排名 {i+1}:")
            print(f"   相似度: {similarity:.4f} (距离: {distance:.4f})")
            print(f"   内容: {doc_preview}")
            
            if contains_expected:
                print(f"   ✓ 包含期望关键词: '{expected}'")
        
        # 相似度差异分析
        if len(similarities) >= 2:
            score_gap = similarities[0] - similarities[1]
            print(f"\n📈 相似度分析:")
            print(f"   最高分: {similarities[0]:.4f}")
            print(f"   第二名: {similarities[1]:.4f}")
            print(f"   分数差距: {score_gap:.4f}")
            
            if score_gap > 0.1:
                print(f"   💡 结果置信度: 高 (分数差距明显)")
            elif score_gap > 0.05:
                print(f"   💡 结果置信度: 中等")
            else:
                print(f"   💡 结果置信度: 低 (分数接近，可能存在歧义)")
        
        return is_correct
        
    except Exception as e:
        print(f"❌ 查询处理出错: {str(e)}")
        return False

def test_vector_search():
    """完整的向量检索测试"""
    print("🚀 开始向量检索综合测试")
    
    # 1. 初始化文档
    if not setup_documents():
        print("❌ 文档初始化失败，测试终止")
        return
    
    # 2. 测试所有查询
    print(f"\n🔬 开始测试 {len(test_queries)} 个查询...")
    
    correct_count = 0
    total_count = len(test_queries)
    
    for i, query_info in enumerate(test_queries):
        print(f"\n\n📝 测试进度: {i+1}/{total_count}")
        is_correct = test_single_query(query_info)
        if is_correct:
            correct_count += 1
    
    # 3. 总结测试结果
    print(f"\n\n{'='*80}")
    print(f"🎯 测试总结")
    print(f"{'='*80}")
    print(f"总查询数: {total_count}")
    print(f"正确匹配: {correct_count}")
    print(f"准确率: {correct_count/total_count*100:.1f}%")
    
    if correct_count == total_count:
        print(f"🎉 所有测试通过！向量检索功能工作正常")
    elif correct_count >= total_count * 0.8:
        print(f"✅ 大部分测试通过，向量检索功能基本正常")
    else:
        print(f"⚠️  部分测试失败，可能需要调整embedding模型或测试数据")
    
    print(f"\n💡 建议:")
    print(f"   - 观察相似度分数的分布和差距")
    print(f"   - 检查失败案例的原因（语义理解、关键词匹配等）")
    print(f"   - 考虑调整查询表述或增加更多样化的测试数据")
    
    print(f"\n🏁 测试完成")

if __name__ == "__main__":
    test_vector_search()