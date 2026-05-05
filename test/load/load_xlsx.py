"""传统加载Excel文件并智能分块(未启用)"""



# from langchain_community.document_loaders import UnstructuredExcelLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
 
# # ==========================================
# # 3. 加载Excel文件
# def load_excel_file(file_path):
#     loader = UnstructuredExcelLoader(
#         file_path,
#         mode="elements",
#         strategy="fast",  # 快速模式，无需额外NLP依赖
#         include_header_in_cell_text=True  # 自动绑定表头和单元格内容
#     )
#     docs = loader.load()
#     print(f"成功加载Excel，共解析出 {len(docs)} 个元素")
# # ==========================================
# # 4. 智能分块（中文场景优化）
# # ==========================================
    
#     text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,  # 通用场景500-1000最佳
#     chunk_overlap=50,  # 重叠5%-10%，避免上下文断裂
#     separators=[
#         "\n## ", "\n### ",  # 优先按标题拆分
#         "\n\n",  # 按段落拆分
#         "。", "！", "？",  # 中文句子结束符
#         "\n", " ", ""  # 兜底分隔符
#     ]
# )
#     chunks = text_splitter.split_documents(docs)
#     print(f"成功分块，共生成 {len(chunks)} 个Chunk")
#     return chunks









