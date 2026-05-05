from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import json
from typing import List, Dict
from config import MILVUS_URI, COLLECTION_NAME


"""连接Milvus数据库"""
def get_client():
    global client
    client = MilvusClient(uri=MILVUS_URI)
    print("✅ Milvus 连接成功！")
    return client

"""创建新集合（名称+描述）"""
def create_collection_entry(collection_name: str, description: str):

    client = get_client()

    if client.has_collection(collection_name):
        raise ValueError(f"集合 '{collection_name}' 已存在")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="source_id", dtype=DataType.INT64),
    ]

    schema = CollectionSchema(fields, description=description)
    client.create_collection(collection_name, schema=schema)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"nlist": 128},
    )
    client.create_index(collection_name, index_params)
    client.load_collection(collection_name)
    print(f"✅ 集合 '{collection_name}' 创建并加载完成！")

    return {"name": collection_name, "description": description}

"""插入BGE-M3两种向量到Milvus"""
def insert_bge_m3_vectors(
    texts: List[str],
    dense_embeddings: np.ndarray,
    sparse_embeddings: List[Dict[int, float]],
    collection_name: str = None,
    source_id: int = 0,
):

    target = collection_name or COLLECTION_NAME
    client = get_client()

    if not client.has_collection(target):
        raise ValueError(f"集合 '{target}' 不存在，请先在集合管理页创建")
    client.load_collection(target)

    desc = client.describe_collection(target)
    field_names = {f["name"] for f in desc.get("fields", [])}
    if "dense_vector" not in field_names or "sparse_vector" not in field_names:
        raise ValueError(f"集合 '{target}' 缺少向量字段，不能写入 chunk")

    dense_data = dense_embeddings
    sparse_data = sparse_embeddings

    data = [
        {
        "dense_vector": dense_data[i],
        "sparse_vector": sparse_data[i],
        "text": texts[i],
        "source_id": source_id
        }
        for i in range(len(texts))
    ]

    insert_result = client.insert(target, data)
    chunk_ids = [int(i) for i in insert_result["ids"]]
    print(f"✅ 成功插入 {len(texts)} 条BGE-M3向量数据！")

    return chunk_ids


"""搜索密集向量"""
def search_dense_vectors(query_dense, dense_top_k, collection_name=None):
    target = collection_name or COLLECTION_NAME
    client = get_client()
    client.load_collection(target)

    results = client.search(
        target,
        data=query_dense,
        anns_field="dense_vector",
        search_params={"metric_type": "COSINE", "params": {"nprobe": 15}},
        limit=dense_top_k,
        output_fields=["text"]
    )
    return results



"""搜索稀疏向量"""
def search_sparse_vectors(query_sparse, sparse_top_k, collection_name=None):
    target = collection_name or COLLECTION_NAME
    client = get_client()
    client.load_collection(target)
    
    results = client.search(
        target,
        data=query_sparse,
        anns_field="sparse_vector",
        search_params={"metric_type": "IP", "params": {}},
        limit=sparse_top_k,
        output_fields=["text"]
    )
    return results


"""将密集嵌入转换为Milvus格式"""
def dense_to_milvus_format(query_dense: np.ndarray) -> List[List[float]]:
    
    if len(query_dense.shape) == 1:
        return [query_dense.tolist()]
    return query_dense.tolist()


"""将稀疏向量转换为Milvus格式"""
def sparse_to_milvus_format(query_sparse: List[Dict[int, float]]):
    
    query_sparse = [query_sparse]
    return query_sparse


""" BGE-M3 双路混合检索（dense + sparse）"""
def hybrid_search_bge_m3(query_dense, query_sparse, dense_top_k, sparse_top_k, alpha, collection_name=None):

    dense_res = search_dense_vectors(dense_to_milvus_format(query_dense), dense_top_k, collection_name)
    sparse_res = search_sparse_vectors(sparse_to_milvus_format(query_sparse), sparse_top_k, collection_name)

    doc_dict = {}

    for hit in dense_res[0]:
        doc_dict[hit.id] = {
            "text": hit.entity["text"],
            "d_score": hit.distance,
            "s_score": 0
        }

    for hit in sparse_res[0]:
        if hit.id in doc_dict:
            doc_dict[hit.id]["s_score"] = hit.distance
        else:
            doc_dict[hit.id] = {
                "text": hit.entity["text"],
                "d_score": 0,
                "s_score": hit.distance
            }

    combined = []
    for doc_id, info in doc_dict.items():
        final_score = alpha * info["d_score"] + (1 - alpha) * info["s_score"]
        combined.append({
            "id": doc_id,
            "text": info["text"],
            "distance": final_score
        })

    combined = sorted(combined, key=lambda x: x["distance"], reverse=True)
    return [combined[:dense_top_k + sparse_top_k]]


"""列出所有集合（名称+描述）"""
def list_collections():
    client = get_client()
    names = client.list_collections()
    result = []
    for name in names:
        try:
            info = client.describe_collection(name)
            result.append({
                "name": name,
                "description": info.get("description", ""),
            })
        except Exception:
            result.append({"name": name, "description": ""})
    return result


""" JSON 文件注册表 — 文件 → 集合 → source_id """
import os as _os

_FILE_REGISTRY_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "file_registry.json")


def _load_registry() -> list:
    if _os.path.exists(_FILE_REGISTRY_PATH):
        with open(_FILE_REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_registry(data: list):
    with open(_FILE_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _next_source_id() -> int:
    registry = _load_registry()
    if not registry:
        return 1
    return max(item.get("source_id", 0) for item in registry) + 1


def register_file_source(file_name: str, collection_name: str, source_id: int):
    registry = _load_registry()
    registry.append({
        "file_name": file_name,
        "collection_name": collection_name,
        "source_id": source_id,
    })
    _save_registry(registry)
    print(f"✅ 文件 '{file_name}' 注册完成，source_id={source_id} → 集合 '{collection_name}'")


def list_registered_files() -> list:
    return _load_registry()


def delete_file_by_source(file_name: str):
    registry = _load_registry()
    entry = None
    for item in registry:
        if item["file_name"] == file_name:
            entry = item
            break
    if not entry:
        raise ValueError(f"未找到文件 '{file_name}' 的记录")

    source_id = entry["source_id"]
    collection_name = entry["collection_name"]

    client = get_client()
    if client.has_collection(collection_name):
        client.load_collection(collection_name)
        filter_expr = f"source_id == {source_id}"
        try:
            client.delete(collection_name=collection_name, filter=filter_expr)
            print(f"✅ 从集合 '{collection_name}' 删除 source_id={source_id} 的所有chunk")
        except Exception as e:
            print(f"⚠️ 删除chunk失败: {e}")
            raise RuntimeError(f"删除chunk失败: {e}")

    registry = [item for item in registry if item["file_name"] != file_name]
    _save_registry(registry)
    print(f"✅ 文件 '{file_name}' 注册记录已删除")

    return {"file_name": file_name, "source_id": source_id, "collection_name": collection_name}



