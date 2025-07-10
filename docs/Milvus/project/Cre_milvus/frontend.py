import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time
import json

# python -m streamlit run frontend.py
st.set_page_config(page_title="智能向量检索系统", layout="wide", page_icon="🔍")
st.title("🔍 智能向量检索系统")
st.markdown("---")

# 自定义指标卡片样式
def style_metric_cards(background_color="#FFFFFF", border_left_color="#0078ff"):
    st.markdown(
        f"""
        <style>
        div[data-testid="metric-container"] {{
            background-color: {background_color};
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            border-left: 5px solid {border_left_color};
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }}
        div[data-testid="metric-container"]:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.1);
        }}
        div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] {{
            color: #555;
            font-weight: 600;
        }}
        div[data-testid="metric-container"] > div[data-testid="stMetricValue"] {{
            color: #0078ff;
            font-size: 1.8rem;
            font-weight: 700;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# 初始化会话状态
if "config" not in st.session_state:
    st.session_state.config = {
        "milvus": {
            "host": "127.0.0.1",
            "port": "19530",
            "vector_name": "default",
            "collection_name": "Test_one",
            "index_name": "IVF_FLAT",
            "replica_num": 1,
            "index_device": "cpu"
        },
        "system": {
            "url_split": False
        },
        "search": {
            "top_k": 20,
            "col_choice": "hdbscan",
            "reorder_strategy": "distance"
        },
        "data": {
            "data_location": ""
        }
    }

if "last_search" not in st.session_state:
    st.session_state.last_search = None

# 配置参数设置
with st.expander("⚙️ 配置参数设置", expanded=True):
    with st.form("config_form"):
        st.subheader("Milvus 配置")
        col1, col2, col3 = st.columns(3)
        with col1:
            milvus_host = st.text_input("Milvus Host", value=st.session_state.config["milvus"]["host"])
            vector_name = st.text_input("Vector DB Name", value=st.session_state.config["milvus"]["vector_name"])
            insert_mode = st.selectbox(
                "数据插入模式",
                ["覆盖（删除原有数据）", "追加（保留原有数据）"],
                index=0  # 默认覆盖
            )
        with col2:
            milvus_port = st.text_input("Milvus Port", value=st.session_state.config["milvus"]["port"])
            collection_name = st.text_input("Collection Name", value=st.session_state.config["milvus"]["collection_name"])
            url_split = st.selectbox(
                "是否启用URL切分", 
                ["True", "False"],
                index=0 if st.session_state.config["system"]["url_split"] else 1
            )
        with col3:
            index_name = st.selectbox(
                "Index Name", 
                ["IVF_FLAT", "HNSW", "HNSW_SQ8"],
                index=["IVF_FLAT", "HNSW", "HNSW_SQ8"].index(st.session_state.config["milvus"]["index_name"])
            )
            replica_num = st.number_input(
                "Replica Num", 
                value=st.session_state.config["milvus"]["replica_num"], 
                min_value=1
            )
            index_device = st.selectbox(
                "Index Device", 
                ["cpu", "gpu"],
                index=0 if st.session_state.config["milvus"]["index_device"] == "cpu" else 1
            )

        st.subheader("检索参数")
        col4, col5 = st.columns(2)
        with col4:
            search_top_k = st.number_input(
                "Search Top K", 
                value=st.session_state.config["search"]["top_k"], 
                min_value=1
            )
            search_col_choice = st.selectbox(
                "Search Col Choice", 
                ["hdbscan", "kmeans"],
                index=0 if st.session_state.config["search"]["col_choice"] == "hdbscan" else 1
            )
        with col5:
            search_reorder_strategy = st.selectbox(
                "Search Reorder Strategy", 
                ["distance", "cluster_size", "cluster_center"],
                index=["distance", "cluster_size", "cluster_center"].index(
                    st.session_state.config["search"]["reorder_strategy"]
                )
            )

        submitted = st.form_submit_button("💾 保存配置")
        if submitted:
            config_data = {
                "milvus": {
                    "host": milvus_host,
                    "port": milvus_port,
                    "vector_name": vector_name,
                    "collection_name": collection_name,
                    "index_name": index_name,
                    "replica_num": replica_num,
                    "index_device": index_device
                },
                "system": {
                    "url_split": url_split == "True",
                    "insert_mode": insert_mode  

                },
                "search": {
                    "top_k": search_top_k,
                    "col_choice": search_col_choice,
                    "reorder_strategy": search_reorder_strategy
                }
            }
            
            # 更新会话状态
            st.session_state.config["milvus"] = config_data["milvus"]
            st.session_state.config["system"] = config_data["system"]
            st.session_state.config["search"] = config_data["search"]
            
            # 发送到后端
            try:
                response = requests.post("http://localhost:8500/update_config", json=config_data)
                if response.status_code == 200:
                    st.success("✅ 配置已保存并生效")
                else:
                    st.error(f"❌ 配置保存失败，状态码: {response.status_code}")
            except Exception as e:
                st.error(f"❌ 连接后端失败: {str(e)}")

st.markdown("---")

# 上传文件夹
with st.expander("📁 上传数据文件夹", expanded=True):
    st.info("请全选文件夹下所有文件上传，并输入一个文件夹名，系统会自动保存到该目录。")
    folder_name = st.text_input("请输入目标文件夹名（如20240501）", key="folder_name")
    uploaded_files = st.file_uploader(
        "选择文件夹中的文件（支持csv, md, pdf, txt, jpg, png）", 
        accept_multiple_files=True, 
        type=["csv", "md", "pdf", "txt", "jpg", "jpeg", "png"]
    )
    
    if st.button("⬆️ 上传并构建向量库", key="upload_btn"):
        if not folder_name:
            st.warning("⚠️ 请先输入目标文件夹名。")
        elif not uploaded_files:
            st.warning("⚠️ 请先选择要上传的文件。")
        else:
            with st.spinner("上传文件中，请稍候..."):
                # 1. 上传文件
                files = [("files", (file.name, file, file.type)) for file in uploaded_files]
                try:
                    response = requests.post(
                        "http://localhost:8500/upload",
                        params={"folder_name": folder_name},
                        files=files
                    )
                    
                    if response.status_code == 200:
                        # 2. 更新配置文件中的 data_location 字段
                        config_update = {
                            "data": {
                                "data_location": f"./data/upload/{folder_name}"
                            }
                        }
                        st.session_state.config["data"] = config_update["data"]
                        
                        # 发送更新请求
                        update_response = requests.post("http://localhost:8500/update_config", json=config_update)
                        
                        if update_response.status_code == 200:
                            st.success(f"✅ 成功上传 {len(uploaded_files)} 个文件并更新配置")
                            st.balloons()
                        else:
                            st.error(f"❌ 配置更新失败: {update_response.text}")
                    else:
                        st.error(f"❌ 文件上传失败: {response.text}")
                except Exception as e:
                    st.error(f"❌ 连接后端失败: {str(e)}")

st.markdown("---")

# with st.expander("📁 上传数据文件夹", expanded=True):
#     st.info("请全选文件夹下所有文件上传，并输入一个文件夹名，系统会自动保存到该目录。")
#     folder_name = st.text_input("请输入目标文件夹名（如20240501）", key="folder_name")
#     uploaded_files = st.file_uploader(
#         "选择文件夹中的文件（支持csv, md, pdf, txt, jpg, png）", 
#         accept_multiple_files=True, 
#         type=["csv", "md", "pdf", "txt", "jpg", "jpeg", "png"]
#     )

# 检索与可视化
with st.expander("🔎 检索与可视化", expanded=True):
    question = st.text_input("请输入检索问题", key="search_question")
    col_choice = st.selectbox(
        "聚类算法", 
        ["hdbscan", "kmeans"],
        index=0 if st.session_state.config["search"]["col_choice"] == "hdbscan" else 1,
        key="col_choice"
    )
    
    # 添加结果展示选项
    result_display = st.radio("结果展示方式", ["摘要视图", "详细视图"], index=0, horizontal=True)
    
    if st.button("🚀 开始检索与可视化", key="search_btn", type="primary"):
        if not question:
            st.warning("⚠️ 请输入检索问题！")
        else:
            with st.spinner("检索中，请稍候..."):
                try:
                    # 1. 执行搜索
                    search_response = requests.post(
                        "http://localhost:8500/search",
                        data={"question": question, "col_choice": col_choice}
                    )
                    
                    if search_response.status_code == 200:
                        search_result = search_response.json()
                        st.session_state.last_search = search_result
                        
                        # 显示基本信息
                        if "clusters" in search_result and search_result["clusters"]:
                            cluster_count = len(search_result["clusters"])
                            doc_count = sum(len(cluster["documents"]) for cluster in search_result["clusters"])
                            st.success(f"✅ 检索完成: 找到 {cluster_count} 个集群, 共 {doc_count} 个文档")
                            
                            # 显示所有召回结果
                            st.subheader("所有召回结果")
                            
                            # 创建选项卡布局
                            tab1, tab2 = st.tabs(["文档列表", "集群视图"])
                            
                            with tab1:
                                # 按距离排序的所有文档
                                all_docs = []
                                for cluster in search_result["clusters"]:
                                    all_docs.extend(cluster["documents"])
                                all_docs_sorted = sorted(all_docs, key=lambda x: x["distance"],reverse=True)
                                
                                st.write(f"共召回 {len(all_docs_sorted)} 个文档（按相似度排序）:")
                                
                                # 分页显示结果
                                page_size = 5
                                max_page = max(1, (len(all_docs_sorted) + page_size - 1) // page_size)
                                page_number = st.number_input("页码", min_value=1, 
                                                             max_value=max_page, 
                                                             value=1)
                                
                                start_idx = (page_number - 1) * page_size
                                end_idx = min(start_idx + page_size, len(all_docs_sorted))
                                
                                for i in range(start_idx, end_idx):
                                    doc = all_docs_sorted[i]
                                    
                                    # 使用容器而不是嵌套的 expander
                                    with st.container():
                                        st.subheader(f"文档 #{i+1} (ID: {doc['id']}, 距离: {doc['distance']:.4f})")
                                        st.markdown(f"**ID:** {doc['id']}")
                                        st.markdown(f"**相似度距离:** {doc['distance']:.4f}")
                                        if "url" in doc:
                                            st.markdown(f"**URL:** [{doc['url']}]({doc['url']})")
                                        
                                        content = doc['content']
                                        if result_display == "摘要视图":
                                            # 显示摘要
                                            preview = content[:300] + "..." if len(content) > 300 else content
                                            st.markdown(f"**内容摘要:**")
                                            st.write(preview)
                                        else:
                                            # 显示完整内容
                                            st.markdown(f"**完整内容:**")
                                            st.text_area("", value=content, height=200, key=f"full_content_{doc['id']}", label_visibility="collapsed")
                                        
                                        st.markdown("---")
                            
                            with tab2:
                                # 显示集群指标卡
                                cluster_count = len(search_result["clusters"])
                                doc_count = sum(len(cluster["documents"]) for cluster in search_result["clusters"])
                                avg_docs = doc_count / cluster_count if cluster_count > 0 else 0
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("集群数量", cluster_count)
                                with col2:
                                    st.metric("文档总数", doc_count)
                                with col3:
                                    st.metric("平均文档数", f"{avg_docs:.1f}")
                                
                                # 应用自定义样式
                                style_metric_cards()
                                
                                # 显示每个集群的文档
                                for cluster in search_result["clusters"]:
                                    st.subheader(f"集群 #{cluster['cluster_id']} (文档数: {len(cluster['documents'])})")
                                    
                                    for doc in cluster["documents"]:
                                        with st.container():
                                            st.markdown(f"**ID:** {doc['id']} | **距离:** {doc['distance']:.4f}")
                                            if "url" in doc:
                                                st.markdown(f"**URL:** [{doc['url']}]({doc['url']})")
                                            
                                            content = doc['content']
                                            if result_display == "摘要视图":
                                                # 显示摘要
                                                preview = content[:300] + "..." if len(content) > 300 else content
                                                st.markdown(f"**内容摘要:**")
                                                st.write(preview)
                                            else:
                                                # 显示完整内容
                                                st.markdown(f"**完整内容:**")
                                                st.text_area("", value=content, height=200, key=f"cluster_content_{doc['id']}", label_visibility="collapsed")
                                            
                                            st.markdown("---")
                        
                        else:
                            st.info("ℹ️ 未找到相关文档")
                        
                        # 2. 执行可视化（仅限HDBSCAN）
                        if col_choice.lower() == "hdbscan" and "clusters" in search_result and search_result["clusters"]:
                            vis_response = requests.post(
                                "http://localhost:8500/visualization",
                                data={"collection_name": collection_name}
                            )
                            
                            if vis_response.status_code == 200:
                                vis_data = vis_response.json()
                                
                                if isinstance(vis_data, list) and vis_data:
                                    df = pd.DataFrame(vis_data)
                                    
                                    # 显示可视化图表
                                    st.subheader("HDBSCAN聚类可视化（UMAP降维）")
                                    fig = px.scatter(
                                        df, x="x", y="y", color="cluster", 
                                        hover_data=["text"],
                                        title="",
                                        width=1000, height=600,
                                        color_continuous_scale=px.colors.sequential.Viridis
                                    )
                                    fig.update_traces(
                                        marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')),
                                        selector=dict(mode='markers')
                                    )
                                    fig.update_layout(
                                        hoverlabel=dict(bgcolor="white", font_size=12),
                                        legend_title_text='集群ID'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                else:
                                    st.info("ℹ️ 无可视化数据")
                            else:
                                st.error(f"❌ 可视化失败: {vis_response.text}")
                    else:
                        st.error(f"❌ 检索失败: {search_response.text}")
                except Exception as e:
                    st.error(f"❌ 连接后端失败: {str(e)}")
# 显示原始数据
if "df" in locals() and df is not None and not df.empty:
    with st.expander("查看原始数据"):
        st.dataframe(df)
# 显示当前配置信息
with st.expander("📋 当前配置信息", expanded=False):
    st.json(st.session_state.config)
# for i in range(start_idx, end_idx):
#     doc = all_docs_sorted[i]
#     with st.container():
#         st.subheader(f"文档 #{i+1} (ID: {doc['id']}, 距离: {doc['distance']:.4f})")
#         # ...existing code...
#         content = doc['content']
#         # 如果content是图片路径，显示图片
#         if isinstance(content, str) and content.lower().endswith((".jpg", ".jpeg", ".png")):
#             st.image(content, caption=f"图片ID: {doc['id']}")
#         else:
#             # 原有摘要/全文逻辑
#             if result_display == "摘要视图":
#                 preview = content[:300] + "..." if len(content) > 300 else content
#                 st.markdown(f"**内容摘要:**")
#                 st.write(preview)
#             else:
#                 st.markdown(f"**完整内容:**")
#                 st.text_area("", value=content, height=200, key=f"full_content_{doc['id']}", label_visibility="collapsed")
#         st.markdown("---")
# 页脚
st.markdown("---")
st.caption("© 2025 智能向量检索系统 | 版本 1.0.0")