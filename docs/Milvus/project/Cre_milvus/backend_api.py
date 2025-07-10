# 文件: backend_api.py
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import shutil
import os
import yaml
import numpy as np
from System.start import load_config, Cre_VectorDataBaseStart_from_config, Cre_Search
from ColBuilder.visualization import get_all_embeddings_and_texts
import hdbscan
from umap import UMAP
import pandas as pd


# python -m uvicorn backend_api:app --reload --port 8500
app = FastAPI()

@app.post("/update_config")
async def update_config(request: Request):
    data = await request.json()
    with open("../Cre_milvus/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.update(data)
    with open("../Cre_milvus/config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)
    return {"message": "配置已更新"}

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...), folder_name: str = None):
    if not folder_name:
        return {"message": "未指定目标文件夹名"}
    upload_dir = f"data/upload/{folder_name}"
    print(f"上传目录: {upload_dir}")
    os.makedirs(upload_dir, exist_ok=True)
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    # 读取配置并处理数据
    config = load_config()
    Cre_VectorDataBaseStart_from_config(config)
    return {"message": "文件上传并处理完成"}

# @app.post("/upload")
# async def upload(files: list[UploadFile] = File(...), folder_name: str = None):
#     if not folder_name:
#         return {"message": "未指定目标文件夹名"}
#     upload_dir = f"data/upload/{folder_name}"
#     os.makedirs(upload_dir, exist_ok=True)
#     image_rows = []
#     img_id = 1
#     for file in files:
#         file_path = os.path.join(upload_dir, file.filename)
#         with open(file_path, "wb") as f:
#             shutil.copyfileobj(file.file, f)
#         # 如果是图片，记录到csv
#         if file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
#             image_rows.append({"id": img_id, "path": file_path})
#             img_id += 1
#     # 如果有图片，生成images.csv
#     if image_rows:
#         import csv
#         csv_path = os.path.join(upload_dir, "images.csv")
#         with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=["id", "path"])
#             writer.writeheader()
#             writer.writerows(image_rows)
#     # 读取配置并处理数据
#     config = load_config()
#     Cre_VectorDataBaseStart_from_config(config)
#     return {"message": "文件上传并处理完成"}

@app.post("/search")
async def search_api(question: str = Form(...)):
    config = load_config()
    result = Cre_Search(config, question)
    return JSONResponse(content=result)


@app.post("/visualization")
async def cluster_visualization(collection_name: str = Form(...)):
    try:
        ids, embeddings, texts = get_all_embeddings_and_texts(collection_name)
        if len(embeddings) == 0:
            return JSONResponse(content={"message": "没有可用的嵌入向量数据", "data": []}, status_code=404)

        # HDBSCAN聚类
        clusterer = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=3)
        labels = clusterer.fit_predict(embeddings)

        # UMAP降维
        umap = UMAP(n_components=2, random_state=42, n_neighbors=min(80, len(embeddings)-1), min_dist=0.1)
        umap_result = umap.fit_transform(embeddings)

        df = pd.DataFrame(umap_result, columns=["x", "y"])
        df["cluster"] = labels.astype(str)
        df["text"] = texts
        df["id"] = ids

        # 过滤噪声点
        df = df[df["cluster"] != "-1"]

        return JSONResponse(content=df.to_dict(orient="records"))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"聚类可视化失败: {str(e)}"}
        )