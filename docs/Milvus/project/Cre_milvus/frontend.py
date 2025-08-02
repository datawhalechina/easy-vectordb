import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time
import json
from datetime import datetime
import logging
from typing import Dict, Any, Optional, List

# python -m streamlit run frontend.py

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BACKEND_URL = "http://localhost:8509"
DEFAULT_TIMEOUT = 10

st.set_page_config(
    page_title="DataWhale-easyVectorDB", 
    layout="wide", 
    page_icon="🔍",
    initial_sidebar_state="collapsed"
)

# 添加自定义CSS样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .status-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .success-card {
        border-left-color: #28a745;
    }
    .warning-card {
        border-left-color: #ffc107;
    }
    .error-card {
        border-left-color: #dc3545;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown("""
<div class="main-header">
    <h1>🔍 DataWhale-easyVectorDB Enhanced</h1>
    <p style="text-align: center; color: white; margin: 0; opacity: 0.9;">
        智能向量数据库管理系统- 集成聚类搜索、智能分块、性能监控与压测
    </p>
</div>
""", unsafe_allow_html=True)

# Helper functions
def safe_request(method: str, url: str, timeout: int = DEFAULT_TIMEOUT, **kwargs) -> Optional[requests.Response]:
    """Safe HTTP request with error handling"""
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None

def build_chunking_config(strategy: str, chunk_length: int, ppl_threshold: float, 
                         confidence_threshold: float, similarity_threshold: float, 
                         overlap: int) -> Dict[str, Any]:
    """根据策略构建分块配置"""
    config = {
        "strategy": strategy,
        "chunk_length": chunk_length,
        "language": "zh"
    }
    
    if strategy == "meta_ppl":
        config["ppl_threshold"] = ppl_threshold
    elif strategy == "msp":
        config["confidence_threshold"] = confidence_threshold
    elif strategy == "semantic":
        config["similarity_threshold"] = similarity_threshold
        config["min_chunk_size"] = 100
        config["max_chunk_size"] = chunk_length
    elif strategy == "traditional":
        config["overlap"] = overlap
    elif strategy == "margin_sampling":
        config["confidence_threshold"] = confidence_threshold
    
    return config

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
        },
        "chunking": {  # 添加默认分块配置
            "strategy": "traditional",
            "chunk_length": 512,
            "overlap": 50
        },
        "multimodal": {  # 添加多模态配置
            "enable_image": False,
            "clip_model": "ViT-B/32",
            "image_formats": ["jpg", "jpeg", "png", "bmp"]
        }
    }

if "last_search" not in st.session_state:
    st.session_state.last_search = None

# 侧边栏快速状态
with st.sidebar:
    st.header("🚀 系统快速状态")
    
    # 快速状态检查
    status_response = safe_request("GET", f"{BACKEND_URL}/system/status", timeout=3)
    if status_response and status_response.status_code == 200:
        try:
            status_data = status_response.json()
            health = status_data.get("health", {})
            overall_status = health.get("overall_status", "unknown")
            
            if overall_status == "healthy":
                st.success(" 系统健康")
            elif overall_status == "degraded":
                st.warning("⚠️ 系统降级")
            else:
                st.error("❌ 系统异常")
            
            # 显示关键指标
            status = status_data.get("status", {})
            
            st.markdown("**核心服务:**")
            milvus_ok = status.get("milvus", {}).get("connected", False)
            st.write(f"🗄️ Milvus: {'' if milvus_ok else '❌'}")
            
            embedding_ok = status.get("embedding_model", {}).get("available", False)
            st.write(f"🧠 嵌入模型: {'✅' if embedding_ok else '❌'}")
            
            chunking_ok = status.get("chunking_system", {}).get("available", False)
            st.write(f"✂️ 分块系统: {'✅' if chunking_ok else '❌'}")
            
            clustering_ok = status.get("clustering_service", {}).get("available", False)
            st.write(f"📊 聚类服务: {'✅' if clustering_ok else '❌'}")
            
            # LLM状态
            llm_config = status.get("llm_config", {})
            if llm_config.get("available"):
                active_config = llm_config.get("active_config")
                if active_config:
                    st.markdown("**LLM配置:**")
                    st.write(f"🤖 {active_config.get('provider', 'N/A')}")
                    st.write(f"📝 {active_config.get('model', 'N/A')}")
                else:
                    st.info("🤖 LLM未配置")
        except json.JSONDecodeError:
            st.error("❌ 响应格式错误")
    else:
        st.error("❌ 无法连接后端")
        if status_response:
            st.caption(f"状态码: {status_response.status_code}")
    
    st.markdown("---")
    
    # 快速操作
    st.markdown("**快速操作**")
    if st.button("🔄 刷新状态", key="sidebar_refresh"):
        st.rerun()
    
    if st.button("🧪 快速测试", key="sidebar_test"):
        with st.spinner("运行快速测试..."):
            test_response = safe_request("POST", f"{BACKEND_URL}/system/integration_test", timeout=10)
            if test_response and test_response.status_code == 200:
                try:
                    test_data = test_response.json()
                    summary = test_data.get("summary", {})
                    success_rate = summary.get("success_rate", 0)
                    
                    if success_rate >= 0.8:
                        st.success(f" 测试通过 ({success_rate:.0%})")
                    elif success_rate >= 0.5:
                        st.warning(f"⚠️ 部分通过 ({success_rate:.0%})")
                    else:
                        st.error(f"❌ 测试失败 ({success_rate:.0%})")
                except json.JSONDecodeError:
                    st.error("❌ 测试响应格式错误")
            else:
                st.error("❌ 测试请求失败")
                if test_response:
                    st.caption(f"状态码: {test_response.status_code}")
    
    # 版本信息
    st.markdown("---")
    st.caption("📦 Enhanced Version 2.0")
    st.caption("🔧 集成聚类、智能分块、压测功能")

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
                index=0 if st.session_state.config["search"]["col_choice"] == "hdbscan" else 1,
                key="search_col_choice_unique"
            )
        with col5:
            search_reorder_strategy = st.selectbox(
                "Search Reorder Strategy", 
                ["distance", "cluster_size", "cluster_center"],
                index=["distance", "cluster_size", "cluster_center"].index(
                    st.session_state.config["search"]["reorder_strategy"]
                )
            )

        st.subheader("文本切分配置")
        col6, col7, col8 = st.columns(3)
        with col6:
            chunking_strategy = st.selectbox(
                "切分策略",
                ["traditional", "meta_ppl", "margin_sampling", "msp", "semantic"],
                index=["traditional", "meta_ppl", "margin_sampling", "msp", "semantic"].index(
                    st.session_state.config.get("chunking", {}).get("strategy", "traditional")
                ),
                help="选择文本切分策略：\n- traditional: 固定长度切分\n- meta_ppl: PPL困惑度切分\n- margin_sampling: 边际采样切分\n- msp: MSP高级切分\n- semantic: 语义切分"
            )
        with col7:
            chunk_length = st.number_input(
                "块长度",
                value=st.session_state.config.get("chunking", {}).get("chunk_length", 512),
                min_value=100,
                max_value=2048,
                help="文本块的最大长度"
            )
        
        # 初始化所有可能的参数变量
        ppl_threshold = st.session_state.config.get("chunking", {}).get("ppl_threshold", 0.3)
        confidence_threshold = st.session_state.config.get("chunking", {}).get("confidence_threshold", 0.7)
        similarity_threshold = st.session_state.config.get("chunking", {}).get("similarity_threshold", 0.8)
        overlap = st.session_state.config.get("chunking", {}).get("overlap", 50)
        
        with col8:
            if chunking_strategy == "meta_ppl":
                ppl_threshold = st.slider(
                    "PPL阈值",
                    min_value=0.0,
                    max_value=1.0,
                    value=ppl_threshold,
                    step=0.1,
                    help="PPL困惑度切分的阈值",
                    key="ppl_threshold_slider"
                )
            elif chunking_strategy == "msp":
                confidence_threshold = st.slider(
                    "置信度阈值",
                    min_value=0.5,
                    max_value=0.95,
                    value=confidence_threshold,
                    step=0.05,
                    help="MSP切分的置信度阈值",
                    key="confidence_threshold_slider"
                )
            elif chunking_strategy == "semantic":
                similarity_threshold = st.slider(
                    "相似度阈值",
                    min_value=0.5,
                    max_value=0.95,
                    value=similarity_threshold,
                    step=0.05,
                    help="语义切分的相似度阈值",
                    key="similarity_threshold_slider"
                )
                min_chunk_size = st.number_input(
                    "最小块大小",
                    value=100,
                    min_value=50,
                    max_value=200,
                    key="min_chunk_size_input"
                )
            elif chunking_strategy == "traditional":
                overlap = st.slider(
                    "重叠长度",
                    min_value=0,
                    max_value=200,
                    value=overlap,
                    step=10,
                    help="传统切分的重叠长度",
                    key="overlap_slider"
                )

        st.subheader("LLM配置（用于高级分块策略）")
        
        # 获取LLM配置状态
        llm_configs = {}
        llm_providers = []
        active_config_id = None
        
        # 获取LLM提供商
        providers_response = safe_request("GET", f"{BACKEND_URL}/llm/providers")
        if providers_response and providers_response.status_code == 200:
            try:
                llm_providers = providers_response.json().get("providers", [])
            except json.JSONDecodeError:
                st.warning("LLM提供商数据格式错误")
        
        # 获取现有配置
        configs_response = safe_request("GET", f"{BACKEND_URL}/llm/configs")
        if configs_response and configs_response.status_code == 200:
            try:
                config_data = configs_response.json()
                llm_configs = config_data.get("configs", {})
                summary = config_data.get("summary", {})
                active_config_info = summary.get("active_config", {})
                active_config_id = active_config_info.get("id") if active_config_info else None
            except json.JSONDecodeError:
                st.warning("LLM配置数据格式错误")
        
        col_llm1, col_llm2 = st.columns(2)
        
        with col_llm1:
            st.markdown("**当前LLM配置状态**")
            if active_config_id:
                active_config = llm_configs.get(active_config_id, {})
                st.success(f"已激活 {active_config_id}")
                st.write(f"- 提供商: {active_config.get('provider', 'N/A')}")
                st.write(f"- 模型: {active_config.get('model_name', 'N/A')}")
            else:
                st.warning("⚠️ 未配置LLM，MSP和PPL分块将不可用")
            
            # 显示现有配置列表
            if llm_configs:
                st.markdown("**已保存的配置:**")
                for config_id, config in llm_configs.items():
                    status = "🟢 激活" if config_id == active_config_id else "⚪未激活"
                    st.write(f"- {config_id}: {config.get('provider', 'N/A')} ({status})")
        
        with col_llm2:
            st.markdown("**添加新的LLM配置:**")
            st.info("💡 LLM配置将在主配置保存后可用")

        st.subheader("多模态配置")
        col9, col10 = st.columns(2)
        with col9:
            enable_image = st.checkbox(
                "启用图像处理",
                value=st.session_state.config.get("multimodal", {}).get("enable_image", False)
            )
        with col10:
            clip_model = st.selectbox(
                "CLIP模型",
                ["ViT-B/32", "ViT-B/16", "ViT-L/14"],
                index=["ViT-B/32", "ViT-B/16", "ViT-L/14"].index(
                    st.session_state.config.get("multimodal", {}).get("clip_model", "ViT-B/32")
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
                    "insert_mode": "overwrite" if insert_mode == "覆盖（删除原有数据）" else "append"
                },
                "search": {
                    "top_k": search_top_k,
                    "col_choice": search_col_choice,
                    "reorder_strategy": search_reorder_strategy
                },
                "chunking": build_chunking_config(
                    chunking_strategy, 
                    chunk_length, 
                    ppl_threshold, 
                    confidence_threshold, 
                    similarity_threshold, 
                    overlap
                ),
                "multimodal": {
                    "enable_image": enable_image,
                    "clip_model": clip_model,
                    "image_formats": ["jpg", "jpeg", "png", "bmp"]
                }
            }
            
            # 更新会话状态
            st.session_state.config = config_data
            
            # 发送到后端
            response = safe_request("POST", f"{BACKEND_URL}/update_config", json=config_data)
            if response and response.status_code == 200:
                st.success(" 配置已保存并生效")
            else:
                st.error(f"❌ 配置保存失败")
                if response:
                    st.caption(f"状态码: {response.status_code}")
                    try:
                        error_detail = response.json().get("message", "未知错误")
                        st.caption(f"错误详情: {error_detail}")
                    except:
                        pass

# LLM配置管理（独立表单）
with st.expander("🤖 LLM配置管理", expanded=False):
    # 获取LLM配置状态
    llm_configs = {}
    llm_providers = []
    active_config_id = None
    
    try:
        # 获取LLM提供商
        providers_response = requests.get("http://localhost:8509/llm/providers")
        if providers_response.status_code == 200:
            llm_providers = providers_response.json().get("providers", [])
        
        # 获取现有配置
        configs_response = requests.get("http://localhost:8509/llm/configs")
        if configs_response.status_code == 200:
            config_data = configs_response.json()
            llm_configs = config_data.get("configs", {})
            summary = config_data.get("summary", {})
            active_config_info = summary.get("active_config", {})
            active_config_id = active_config_info.get("id") if active_config_info else None
    except Exception as e:
        st.warning(f"获取LLM配置失败: {str(e)}")
    
    col_llm1, col_llm2 = st.columns(2)
    
    with col_llm1:
        st.markdown("**当前LLM配置状态**")
        if active_config_id:
            active_config = llm_configs.get(active_config_id, {})
            st.success(f"已激活 {active_config_id}")
            st.write(f"- 提供商: {active_config.get('provider', 'N/A')}")
            st.write(f"- 模型: {active_config.get('model_name', 'N/A')}")
        else:
            st.warning("⚠️ 未配置LLM，MSP和PPL分块将不可用")
        
        # 显示现有配置列表
        if llm_configs:
            st.markdown("**已保存的配置:**")
            for config_id, config in llm_configs.items():
                status = "🟢 激活" if config_id == active_config_id else "⚪未激活"
                st.write(f"- {config_id}: {config.get('provider', 'N/A')} ({status})")
    
    with col_llm2:
        st.markdown("**添加新的LLM配置:**")
        
        # LLM配置表单（独立）
        with st.form("llm_config_form"):
            config_id = st.text_input(
                "配置名称",
                help="为这个LLM配置起一个名称"
            )
            
            provider_options = [p["name"] for p in llm_providers] if llm_providers else ["openai", "claude", "local","qwen","zhipu"]
            selected_provider = st.selectbox(
                "LLM提供商",
                provider_options,
                help="选择LLM服务提供商"
            )
            
            # 根据选择的提供商显示模型选项
            if llm_providers:
                provider_info = next((p for p in llm_providers if p["name"] == selected_provider), None)
                if provider_info:
                    model_options = provider_info.get("models", [])
                    selected_model = st.selectbox("模型", model_options)
                    
                    # 显示提供商描述
                    st.info(provider_info.get("description", ""))
                else:
                    selected_model = st.text_input("模型名称", placeholder="例如: glm-4.1v-thinking-flash")
            else:
                selected_model = st.text_input("模型名称", placeholder="例如: gpt-3.5-turbo")
            
            api_key = st.text_input(
                "API密钥",
                type="password",
                help="输入LLM服务的API密钥"
            )
            
            api_endpoint = st.text_input(
                "API端点（可选）",
                placeholder="例如: https://open.bigmodel.cn/api/paas/v4/chat/completions",
                help="自定义API端点，留空使用默认值"
            )
            
            set_as_active = st.checkbox(
                "设为激活配置",
                value=True,
                help="添加后立即激活此配置"
            )
            
            submitted_llm = st.form_submit_button("💾 保存LLM配置")
            
            if submitted_llm:
                if not config_id or not selected_provider or not api_key:
                    st.error("请填写配置名称、提供商和API密钥")
                else:
                    # 保存LLM配置
                    llm_config_data = {
                        "config_id": config_id,
                        "provider": selected_provider,
                        "model_name": selected_model,
                        "api_key": api_key,
                        "api_endpoint": api_endpoint if api_endpoint else None,
                        "is_active": set_as_active
                    }
                    
                    try:
                        response = requests.post(
                            "http://localhost:8509/llm/configs",
                            json=llm_config_data
                        )
                        
                        if response.status_code == 200:
                            st.success(f"LLM配置 '{config_id}' 保存成功")
                            if set_as_active:
                                st.info("🔄 配置已激活，MSP和PPL分块现在可用")
                            st.rerun()
                        else:
                            error_msg = response.json().get("message", "未知错误")
                            st.error(f"❌保存失败: {error_msg}")
                    except Exception as e:
                        st.error(f"❌连接失败: {str(e)}")

st.markdown("---")

# 上传文件区
with st.expander("📁 上传数据文件区", expanded=True):
    st.info("请全选文件夹下所有文件上传，并输入一个文件夹名，系统会自动保存到该目录")
    folder_name = st.text_input("请输入目标文件夹名（如：0240501）", key="folder_name")
    uploaded_files = st.file_uploader(
        "选择文件夹中的文件（支持csv, md, pdf, txt, jpg, png）", 
        accept_multiple_files=True, 
        type=["csv", "md", "pdf", "txt", "jpg", "jpeg", "png"]
    )
    
    if st.button("⬆️ 上传并构建向量库", key="upload_btn"):
        if not folder_name:
            st.warning("⚠️ 请先输入目标文件夹名")
        elif not uploaded_files:
            st.warning("⚠️ 请先选择要上传的文件")
        else:
            with st.spinner("上传文件中，请稍候..."):
                # 1. 上传文件
                files = [("files", (file.name, file, file.type)) for file in uploaded_files]
                data = {"folder_name": folder_name}
                try:
                    response = requests.post(
                        "http://localhost:8509/upload",
                        files=files,
                        data=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # 显示上传结果
                        if result.get("status") == "success":
                            if result.get("vectorized", False):
                                st.success(f"成功上传 {len(uploaded_files)} 个文件并完成向量化存储")
                                st.info("📊 数据已向量化，可以进行检索查询")
                            else:
                                st.success(f"✅成功上传 {len(uploaded_files)} 个文档")
                                st.warning("⚠️ 向量化存储未完成，可能影响检索功能")
                            st.balloons()
                        
                        # 2. 更新配置文件中的 data_location 字段
                        config_update = {
                            "data": {
                                "data_location": f"./data/upload/{folder_name}"
                            }
                        }
                        st.session_state.config["data"] = config_update["data"]
                        
                        # 发送更新请
                        update_response = requests.post("http://localhost:8509/update_config", json=config_update)
                        
                        if update_response.status_code != 200:
                            st.error(f"✅配置更新失败: {update_response.text}")
                    else:
                        st.error(f"✅文件上传失败: {response.text}")
                except Exception as e:
                    st.error(f"✅连接后端失败: {str(e)}")

st.markdown("---")

# 检索与可视
with st.expander("🔎 检索与可视", expanded=True):
    question = st.text_input("请输入检索问", key="search_question")
    col_choice = st.selectbox(
        "聚类算法", 
        ["hdbscan", "kmeans"],
        index=0 if st.session_state.config["search"]["col_choice"] == "hdbscan" else 1,
        key="col_choice"
    )
    
    # 添加结果展示选项
    result_display = st.radio("结果展示方式", ["摘要视图", "详细视图"], index=0, horizontal=True)
    
    if st.button("🚀 开始检索与可视", key="search_btn", type="primary"):
        if not question:
            st.warning("⚠️ 请输入检索问题！")
        else:
            with st.spinner("检索中，请稍.."):
                try:
                    # 1. 执行搜索
                    search_response = requests.post(
                        "http://localhost:8509/search",
                        json={
                            "question": question, 
                            "col_choice": col_choice,
                            "collection_name": st.session_state.config["milvus"]["collection_name"]
                        }
                    )
                    
                    if search_response.status_code == 200:
                        search_result = search_response.json()
                        st.session_state.last_search = search_result
                        
                        # 显示基本信息和质量指
                        if "clusters" in search_result and search_result["clusters"]:
                            cluster_count = len(search_result["clusters"])
                            doc_count = sum(len(cluster["documents"]) for cluster in search_result["clusters"])
                            execution_time = search_result.get("execution_time", 0.0)
                            clustering_method = search_result.get("clustering_method", "unknown")
                            
                            st.success(f"✅检索完✅ 找到 {cluster_count} 个集✅ ✅{doc_count} 个文档(用时: {execution_time:.2f}s, 方法: {clustering_method})")
                            
                            # 显示搜索质量指标
                            if "quality_metrics" in search_result:
                                quality = search_result["quality_metrics"]
                                st.subheader("🎯 搜索质量指标")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    relevance = quality.get("relevance_score", 0.0)
                                    st.metric("相关", f"{relevance:.2f}", 
                                             delta=f"{'优秀' if relevance > 0.7 else '良好' if relevance > 0.5 else '需改进'}")
                                with col2:
                                    diversity = quality.get("diversity_score", 0.0)
                                    st.metric("多样", f"{diversity:.2f}",
                                             delta=f"{'优秀' if diversity > 0.6 else '良好' if diversity > 0.3 else '需改进'}")
                                with col3:
                                    coverage = quality.get("coverage_score", 0.0)
                                    st.metric("覆盖", f"{coverage:.2f}",
                                             delta=f"{'优秀' if coverage > 0.6 else '良好' if coverage > 0.3 else '需改进'}")
                                with col4:
                                    avg_dist = quality.get("avg_distance", 0.0)
                                    st.metric("平均距离", f"{avg_dist:.3f}")
                                
                                # 应用自定义样
                                style_metric_cards()
                            
                            # 显示聚类指标
                            if "cluster_metrics" in search_result:
                                cluster_metrics = search_result["cluster_metrics"]
                                st.subheader("📊 聚类分析指标")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("聚类数量", cluster_metrics.get("num_clusters", 0))
                                with col2:
                                    st.metric("平均聚类大小", f"{cluster_metrics.get('avg_cluster_size', 0):.1f}")
                                with col3:
                                    st.metric("最大聚类", cluster_metrics.get("largest_cluster_size", 0))
                                with col4:
                                    st.metric("聚类内方差", f"{cluster_metrics.get('intra_cluster_variance', 0):.3f}")
                                
                                style_metric_cards()
                            
                            # 显示所有召回结果
                            st.subheader("所有召回结果")
                            
                            # 创建选项卡布局
                            tab1, tab2 = st.tabs(["文档列表", "集群视图"])
                            
                            with tab1:
                                # 按距离排序的所有文档
                                all_docs = []
                                for cluster_idx, cluster in enumerate(search_result["clusters"]):
                                    for doc in cluster["documents"]:
                                        doc_with_cluster = doc.copy()
                                        doc_with_cluster["cluster_id"] = cluster.get("cluster_id", cluster_idx)
                                        all_docs.append(doc_with_cluster)
                                
                                # 排序选项
                                sort_by = st.selectbox(
                                    "排序方式",
                                    ["相似度（距离）", "集群ID", "文档ID"],
                                    key="doc_sort_option"
                                )
                                
                                if sort_by == "相似度（距离）":
                                    all_docs_sorted = sorted(all_docs, key=lambda x: x["distance"], reverse=True)
                                elif sort_by == "集群ID":
                                    all_docs_sorted = sorted(all_docs, key=lambda x: x.get("cluster_id", 0))
                                else:  # 文档ID
                                    all_docs_sorted = sorted(all_docs, key=lambda x: str(x["id"]))
                                
                                st.write(f"共召{len(all_docs_sorted)} 个文档")
                                
                                # 分页显示结果
                                page_size = st.selectbox("每页显示", [5, 10, 20], index=0, key="page_size_select")
                                max_page = max(1, (len(all_docs_sorted) + page_size - 1) // page_size)
                                page_number = st.number_input("页码", min_value=1, 
                                                             max_value=max_page, 
                                                             value=1, key="page_number_input")
                                
                                start_idx = (page_number - 1) * page_size
                                end_idx = min(start_idx + page_size, len(all_docs_sorted))
                                
                                # 显示页面信息
                                st.info(f"显示第 {start_idx + 1}-{end_idx} 个文档，共 {len(all_docs_sorted)} 个")
                                
                                for i in range(start_idx, end_idx):
                                    doc = all_docs_sorted[i]
                                    
                                    with st.container():
                                        # 文档标题行
                                        col_title, col_cluster, col_distance = st.columns([2, 1, 1])
                                        
                                        with col_title:
                                            st.subheader(f"📄 文档 #{i+1}")
                                            st.caption(f"ID: {doc['id']}")
                                        
                                        with col_cluster:
                                            cluster_id = doc.get('cluster_id', 'N/A')
                                            st.metric("所属集群", f"#{cluster_id}")
                                        
                                        with col_distance:
                                            distance = doc['distance']
                                            # 距离颜色编码
                                            if distance > 0.7:
                                                color = "🟢"
                                                quality = "优秀"
                                            elif distance > 0.5:
                                                color = "🟡"
                                                quality = "良好"
                                            else:
                                                color = "🔴"
                                                quality = "一般"
                                            st.metric("相似度", f"{distance:.4f}", delta=f"{color} {quality}")
                                        
                                        # URL信息
                                        if "url" in doc and doc["url"]:
                                            st.markdown(f"🔗 **来源:** [{doc['url']}]({doc['url']})")
                                        
                                        # 内容展示
                                        content = doc['content']
                                        if result_display == "摘要视图":
                                            preview = content[:300] + "..." if len(content) > 300 else content
                                            st.markdown("**内容摘要:**")
                                            st.write(preview)
                                            
                                            # 添加展开按钮
                                            if len(content) > 300:
                                                if st.button(f"展开完整内容", key=f"expand_{doc['id']}_{i}"):
                                                    st.text_area(
                                                        "完整内容", 
                                                        value=content, 
                                                        height=200, 
                                                        key=f"expanded_content_{doc['id']}_{i}"
                                                    )
                                        else:
                                            st.markdown("**完整内容:**")
                                            st.text_area(
                                                "", 
                                                value=content, 
                                                height=200, 
                                                key=f"full_content_{doc['id']}_{i}", 
                                                label_visibility="collapsed"
                                            )
                                        
                                        st.markdown("---")
                            
                            with tab2:
                                # 显示集群概览指标
                                cluster_count = len(search_result["clusters"])
                                doc_count = sum(len(cluster["documents"]) for cluster in search_result["clusters"])
                                avg_docs = doc_count / cluster_count if cluster_count > 0 else 0
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("集群数量", cluster_count)
                                with col2:
                                    st.metric("文档总数", doc_count)
                                with col3:
                                    st.metric("平均文档", f"{avg_docs:.1f}")
                                
                                # 应用自定义样
                                style_metric_cards()
                                
                                # 添加排序选项
                                sort_option = st.selectbox(
                                    "集群排序方式",
                                    ["按平均距离", "按集群大小", "按集群ID"],
                                    key="cluster_sort_option"
                                )
                                
                                # 根据选择排序集群
                                clusters_to_display = search_result["clusters"].copy()
                                if sort_option == "按平均距离":
                                    clusters_to_display.sort(key=lambda x: x.get("avg_distance", 0))
                                elif sort_option == "按集群大小":
                                    clusters_to_display.sort(key=lambda x: len(x["documents"]), reverse=True)
                                else:  # 按集群ID
                                    clusters_to_display.sort(key=lambda x: x.get("cluster_id", 0))
                                
                                # 显示每个集群的详细信
                                for i, cluster in enumerate(clusters_to_display):
                                    cluster_id = cluster.get('cluster_id', i)
                                    cluster_size = len(cluster['documents'])
                                    avg_distance = cluster.get('avg_distance', 0.0)
                                    
                                    # 集群标题和统计信
                                    st.subheader(f"🔍 集群 #{cluster_id}")
                                    
                                    # 集群统计信息
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("文档数量", cluster_size)
                                    with col_b:
                                        st.metric("平均距离", f"{avg_distance:.4f}")
                                    with col_c:
                                        # 计算集群质量评分
                                        quality_score = max(0, min(1, avg_distance)) if avg_distance > 0 else 0
                                        quality_label = "优秀" if quality_score > 0.7 else "良好" if quality_score > 0.5 else "一"
                                        st.metric("质量评分", f"{quality_score:.2f}", delta=quality_label)
                                    
                                    # 使用expander显示集群文档
                                    with st.expander(f"查看集群 #{cluster_id} {cluster_size} 个文", expanded=(i == 0)):
                                        for j, doc in enumerate(cluster["documents"]):
                                            with st.container():
                                                # 文档标题
                                                col_doc1, col_doc2 = st.columns([3, 1])
                                                with col_doc1:
                                                    st.markdown(f"**📄 文档 #{j+1}** - ID: {doc['id']}")
                                                with col_doc2:
                                                    # 距离颜色编码
                                                    distance = doc['distance']
                                                    if distance > 0.7:
                                                        color = "🟢"
                                                    elif distance > 0.5:
                                                        color = "🟡"
                                                    else:
                                                        color = "🔴"
                                                    st.markdown(f"{color} **距离:** {distance:.4f}")
                                                
                                                # URL信息
                                                if "url" in doc and doc["url"]:
                                                    st.markdown(f"🔗 **来源:** [{doc['url']}]({doc['url']})")
                                                
                                                # 内容展示
                                                content = doc['content']
                                                if result_display == "摘要视图":
                                                    preview = content[:300] + "..." if len(content) > 300 else content
                                                    st.markdown("**内容摘要:**")
                                                    st.write(preview)
                                                else:
                                                    st.markdown("**完整内容:**")
                                                    st.text_area(
                                                        "", 
                                                        value=content, 
                                                        height=150, 
                                                        key=f"cluster_{cluster_id}_doc_{j}_{doc['id']}", 
                                                        label_visibility="collapsed"
                                                    )
                                                
                                                st.markdown("---")
                        
                        else:
                            st.info("ℹ️ 未找到相关文")
                        
                        # 2. 执行可视化（仅限HDBSCAN
                        if col_choice.lower() == "hdbscan" and "clusters" in search_result and search_result["clusters"]:
                            vis_response = requests.post(
                                "http://localhost:8509/visualization",
                                json={"collection_name": st.session_state.config["milvus"]["collection_name"]}
                            )
                            
                            if vis_response.status_code == 200:
                                vis_data = vis_response.json()
                                
                                if isinstance(vis_data, list) and vis_data:
                                    df = pd.DataFrame(vis_data)
                                    
                                    # 显示可视化图
                                    st.subheader("HDBSCAN聚类可视化（UMAP降维")
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
                                    
                                    # 显示原始数据
                                    with st.expander("查看原始数据"):
                                        st.dataframe(df)
                                else:
                                    st.info("ℹ️ 无可视化数据")
                            else:
                                st.error(f"可视化失✅ {vis_response.text}")
                    else:
                        st.error(f"✅检索失✅ {search_response.text}")
                except Exception as e:
                    st.error(f"✅连接后端失败: {str(e)}")

st.markdown("---")

# 新增功能面板
with st.expander("🧪 文本切分测试", expanded=False):
    st.info("测试不同的文本切分策略效")
    
    # 获取可用策略和状
    try:
        strategies_response = requests.get("http://localhost:8509/chunking/strategies")
        if strategies_response.status_code == 200:
            strategies_data = strategies_response.json().get("strategies", [])
            
            # 显示策略状
            st.markdown("**可用策略状态**")
            col_status1, col_status2 = st.columns(2)
            
            with col_status1:
                for strategy in strategies_data[:3]:
                    name = strategy.get("display_name", strategy.get("name", ""))
                    if strategy.get("llm_required", False):
                        status = "🟢 可用" if strategy.get("llm_available", False) else "🔴 需要LLM配置"
                    else:
                        status = "🟢 可用"
                    st.write(f"- {name}: {status}")
            
            with col_status2:
                for strategy in strategies_data[3:]:
                    name = strategy.get("display_name", strategy.get("name", ""))
                    if strategy.get("llm_required", False):
                        status = "🟢 可用" if strategy.get("llm_available", False) else "🔴 需要LLM配置"
                    else:
                        status = "🟢 可用"
                    st.write(f"- {name}: {status}")
        else:
            st.warning("无法获取策略状")
    except Exception as e:
        st.warning(f"获取策略状态失✅ {str(e)}")
    
    test_text = st.text_area("输入测试文本", height=150, key="test_text")
    
    col_test1, col_test2 = st.columns(2)
    with col_test1:
        test_strategy = st.selectbox(
            "选择切分策略", 
            ["traditional", "meta_ppl", "margin_sampling", "msp", "semantic"], 
            key="test_strategy",
            help="选择要测试的文本切分策略"
        )
    
    with col_test2:
        # 根据策略显示相关参数
        if test_strategy == "traditional":
            chunk_size = st.number_input("块大", value=512, min_value=100, max_value=2048, key="test_chunk_size")
            overlap = st.number_input("重叠大小", value=50, min_value=0, max_value=200, key="test_overlap")
        elif test_strategy == "meta_ppl":
            ppl_threshold = st.slider("PPL阈", 0.0, 1.0, 0.3, 0.1, key="test_ppl_threshold")
        elif test_strategy == "msp":
            confidence_threshold = st.slider("置信度阈", 0.5, 0.95, 0.7, 0.05, key="test_confidence_threshold")
        elif test_strategy == "semantic":
            similarity_threshold = st.slider("相似度阈", 0.5, 0.95, 0.8, 0.05, key="test_similarity_threshold")
    
    if st.button("🔄 执行切分测试", key="chunking_test_btn"):
        if test_text:
            # 检查LLM依赖策略
            llm_required_strategies = ["msp", "meta_ppl"]
            if test_strategy in llm_required_strategies:
                try:
                    configs_response = requests.get("http://localhost:8509/llm/configs")
                    if configs_response.status_code == 200:
                        summary = configs_response.json().get("summary", {})
                        if not summary.get("active_config"):
                            st.error(f"策略 '{test_strategy}' 需要LLM配置，请先在上方配置LLM")
                            st.stop()
                except Exception:
                    st.warning("⚠️ 无法检查LLM配置状")
            
            with st.spinner("正在执行文本切分..."):
                try:
                    # 构建测试参数
                    test_params = st.session_state.config.get("chunking", {}).copy()
                    test_params["strategy"] = test_strategy
                    
                    # 添加策略特定参数
                    if test_strategy == "traditional":
                        test_params.update({"chunk_size": chunk_size, "overlap": overlap})
                    elif test_strategy == "meta_ppl":
                        test_params.update({"threshold": ppl_threshold})
                    elif test_strategy == "msp":
                        test_params.update({"confidence_threshold": confidence_threshold})
                    elif test_strategy == "semantic":
                        test_params.update({"similarity_threshold": similarity_threshold})
                    
                    response = requests.post(
                        "http://localhost:8509/chunking/process",
                        json={
                            "text": test_text,
                            "strategy": test_strategy,
                            "params": test_params
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"切分完成，共生成 {result['chunk_count']} 个文本块")
                        
                        # 显示切分结果统计
                        chunks = result['chunks']
                        if chunks:
                            avg_length = sum(len(chunk) for chunk in chunks) / len(chunks)
                            min_length = min(len(chunk) for chunk in chunks)
                            max_length = max(len(chunk) for chunk in chunks)
                            
                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            with col_stat1:
                                st.metric("平均长度", f"{avg_length:.0f}")
                            with col_stat2:
                                st.metric("最短块", f"{min_length}")
                            with col_stat3:
                                st.metric("最长块", f"{max_length}")
                        
                        # 显示每个文本
                        for i, chunk in enumerate(chunks):
                            with st.expander(f"文本✅{i+1} (长度: {len(chunk)})", expanded=(i == 0)):
                                st.text_area("", value=chunk, height=100, key=f"chunk_{i}", label_visibility="collapsed")
                    else:
                        error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
                        error_msg = error_data.get("message", response.text)
                        st.error(f"✅切分失败: {error_msg}")
                        
                except Exception as e:
                    st.error(f"✅连接后端失败: {str(e)}")
        else:
            st.warning("⚠️ 请输入测试文")

with st.expander("🖼✅文搜图功", expanded=False):
    st.info("使用文本描述搜索相关图像")
    
    if st.session_state.config.get("multimodal", {}).get("enable_image", False):
        search_text = st.text_input("输入图像描述", key="image_search_text")
        search_top_k = st.number_input("返回图像数量", min_value=1, max_value=50, value=10, key="image_search_k")
        
        if st.button("🔍 搜索图像", key="image_search_btn"):
            if search_text:
                with st.spinner("正在搜索图像..."):
                    try:
                        response = requests.post(
                            "http://localhost:8509/multimodal/text_to_image_search",
                            json={
                                "query_text": search_text,
                                "top_k": search_top_k,
                                "collection_name": st.session_state.config["milvus"]["collection_name"]
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("results"):
                                st.success(f"✅找到 {len(result['results'])} 个相关图")
                                # 显示图像结果
                                cols = st.columns(3)
                                for i, img_info in enumerate(result["results"]):
                                    with cols[i % 3]:
                                        st.image(img_info["image_path"], caption=f"相似 {img_info['distance']:.4f}")
                            else:
                                st.info("ℹ️ " + result.get("message", "未找到相关图"))
                        else:
                            st.error(f"✅搜索失败: {response.text}")
                            
                    except Exception as e:
                        st.error(f"✅连接后端失败: {str(e)}")
            else:
                st.warning("⚠️ 请输入图像描")
    else:
        st.warning("⚠️ 图像处理功能未启用，请在配置中启用多模态功")

with st.expander("📊 性能监控与压", expanded=False):
    st.info("实时监控系统性能指标并进行Milvus集合压力测试")
    
    # 创建选项
    perf_tab1, perf_tab2, perf_tab3 = st.tabs(["系统监控", "压力测试", "测试历史"])
    
    with perf_tab1:
        st.subheader("🖥✅系统性能监控")
        
        col_monitor1, col_monitor2 = st.columns([1, 1])
        
        with col_monitor1:
            if st.button("🔄 刷新性能数据", key="refresh_perf_btn"):
                try:
                    response = requests.get("http://localhost:8509/performance/current")
                    if response.status_code == 200:
                        metrics = response.json().get("metrics", {})
                        
                        if metrics:
                            # 系统指标
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                cpu_percent = metrics.get('cpu', {}).get('percent', 0)
                                cpu_color = "🔴" if cpu_percent > 80 else "🟡" if cpu_percent > 60 else "🟢"
                                st.metric("CPU使用", f"{cpu_percent:.1f}%", delta=f"{cpu_color}")
                            with col2:
                                mem_percent = metrics.get('memory', {}).get('percent', 0)
                                mem_color = "🔴" if mem_percent > 80 else "🟡" if mem_percent > 60 else "🟢"
                                st.metric("内存使用", f"{mem_percent:.1f}%", delta=f"{mem_color}")
                            with col3:
                                disk_percent = metrics.get('disk', {}).get('percent', 0)
                                disk_color = "🔴" if disk_percent > 90 else "🟡" if disk_percent > 70 else "🟢"
                                st.metric("磁盘使用", f"{disk_percent:.1f}%", delta=f"{disk_color}")
                            with col4:
                                # Milvus连接状
                                milvus_info = metrics.get('milvus', {})
                                milvus_status = "🟢 已连接" if milvus_info.get('connected') else "🔴 未连接"
                                collections_count = milvus_info.get('collections_count', 0)
                                st.metric("Milvus状态", milvus_status, delta=f"{collections_count} 个集合")
                            
                            # 应用样式
                            style_metric_cards()
                            
                            # 显示Milvus详细信息
                            if milvus_info.get('connected') and milvus_info.get('collections'):
                                st.subheader("📊 Milvus集合信息")
                                collections = milvus_info.get('collections', [])
                                for i, collection in enumerate(collections):
                                    st.write(f"{collection}")
                                    if i >= 4:  # 最多显✅
                                        remaining = len(collections) - 5
                                        if remaining > 0:
                                            st.write(f"... 还有 {remaining} 个集")
                                        break
                        else:
                            st.info("暂无性能数据")
                    else:
                        st.error("获取性能数据失败")
                except Exception as e:
                    st.error(f"连接失败: {str(e)}")
        
        with col_monitor2:
            # 实时监控选项
            st.subheader("⚙️ 监控设置")
            
            auto_refresh = st.checkbox("启用自动刷新", value=False, key="auto_refresh_monitoring")
            if auto_refresh:
                refresh_interval = st.selectbox("刷新间隔", ["5秒", "10秒", "30秒"], index=1, key="refresh_interval_select")
                refresh_seconds = {"5秒": 5, "10秒": 10, "30秒": 30}[refresh_interval]
                
                # 使用Streamlit的缓存机制实现真正的自动刷新
                @st.cache_data(ttl=refresh_seconds)
                def get_performance_data():
                    try:
                        return requests.get("http://localhost:8509/performance/current").json()
                    except:
                        return {}
                
                metrics = get_performance_data()
            
            # 可选：保留或删除现有数据的选项
            keep_existing_data = st.checkbox("保留现有监控数据", value=True, key="keep_monitoring_data")
            if not keep_existing_data:
                st.warning("⚠️ 现有监控数据将被清除")
                if st.button("清除监控数据", key="clear_monitoring_data"):
                    st.success("监控数据已清")
            
            # 导出监控报告
            if st.button("📊 导出性能报告", key="export_performance_report"):
                try:
                    response = requests.get("http://localhost:8509/performance/export_report")
                    if response.status_code == 200:
                        report_data = response.json()
                        st.download_button(
                            label="下载性能报告",
                            data=json.dumps(report_data, indent=2, ensure_ascii=False),
                            file_name=f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    else:
                        st.error("导出报告失败")
                except Exception as e:
                    st.error(f"导出报告失败: {str(e)}")
    
    with perf_tab2:
        st.subheader("🚀 Milvus集合压力测试")
        
        # 压测配置表单
        with st.form("load_test_config"):
            st.markdown("### 压测参数配置")
            
            col_test1, col_test2 = st.columns(2)
            
            with col_test1:
                test_users = st.number_input(
                    "并发用户", 
                    min_value=1, 
                    max_value=100, 
                    value=10,
                    help="同时执行测试的虚拟用户数"
                )
                
                spawn_rate = st.number_input(
                    "用户启动速率 (用户/", 
                    min_value=0.1, 
                    max_value=10.0, 
                    value=1.0, 
                    step=0.1,
                    help="每秒启动的用户数"
                )
                
                run_time = st.selectbox(
                    "测试持续时间",
                    ["30s", "60s", "120s", "300s", "600s"],
                    index=1,
                    help="压测运行的总时"
                )
            
            with col_test2:
                # 测试场景选择
                st.markdown("**测试场景选择:**")
                scenario_single = st.checkbox("单向量搜", value=True, help="最常见的搜索操")
                scenario_batch = st.checkbox("批量向量搜索", value=True, help="批量搜索操作")
                scenario_precision = st.checkbox("高精度搜", value=False, help="高精度但较慢的搜")
                scenario_fast = st.checkbox("快速搜", value=False, help="快速但精度较低的搜")
                
                # 搜索参数
                st.markdown("**搜索参数:**")
                metric_type = st.selectbox("距离度量", ["L2", "IP", "COSINE"], index=0)
                nprobe = st.slider("nprobe参数", min_value=1, max_value=128, value=16, help="搜索时探测的聚类数量")
                top_k = st.slider("返回结果", min_value=1, max_value=100, value=10, help="每次搜索返回的结果数")
            
            # 目标集合
            target_collection = st.text_input(
                "目标集合名称", 
                value=st.session_state.config["milvus"]["collection_name"],
                help="要进行压测的Milvus集合名称"
            )
            
            # 提交按钮
            submitted = st.form_submit_button("🚀 启动压力测试", type="primary")
            
            if submitted:
                # 构建测试场景列表
                test_scenarios = []
                if scenario_single:
                    test_scenarios.append("single_vector_search")
                if scenario_batch:
                    test_scenarios.append("batch_vector_search")
                if scenario_precision:
                    test_scenarios.append("high_precision_search")
                if scenario_fast:
                    test_scenarios.append("fast_search")
                
                if not test_scenarios:
                    st.error("请至少选择一个测试场")
                else:
                    # 构建测试参数
                    test_params = {
                        "users": test_users,
                        "spawn_rate": spawn_rate,
                        "run_time": run_time,
                        "host": st.session_state.config["milvus"]["host"],
                        "port": st.session_state.config["milvus"]["port"],
                        "collection_name": target_collection,
                        "test_scenarios": test_scenarios,
                        "search_params": {
                            "metric_type": metric_type,
                            "nprobe": nprobe,
                            "top_k": top_k
                        }
                    }
                    
                    # 启动压测
                    with st.spinner("正在启动压力测试..."):
                        try:
                            response = requests.post(
                                "http://localhost:8509/testing/start_load_test",
                                json=test_params
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                test_id = result.get("test_id")
                                
                                if result.get("status") == "success":
                                    st.success(f"压力测试已启动！测试ID: {test_id}")
                                    st.info("测试正在后台运行，请✅测试历史'选项卡中查看进度")
                                    
                                    # 显示测试配置
                                    st.json(test_params)
                                else:
                                    st.error(f"启动测试失败: {result.get('message', '未知错误')}")
                            else:
                                st.error(f"请求失败: {response.status_code}")
                                
                        except Exception as e:
                            st.error(f"启动压测失败: {str(e)}")
        
        # 当前运行的测试状
        st.markdown("### 当前测试状")
        if st.button("🔍 检查运行中的测", key="check_running_tests"):
            try:
                response = requests.get("http://localhost:8509/testing/list_tests")
                if response.status_code == 200:
                    tests = response.json().get("tests", [])
                    running_tests = [t for t in tests if t.get("status") == "running"]
                    
                    if running_tests:
                        for test in running_tests:
                            st.info(f"🏃 测试 {test['test_id']} 正在运行..")
                            
                            # 显示停止按钮
                            if st.button(f"⏹️ 停止测试 {test['test_id']}", key=f"stop_{test['test_id']}"):
                                stop_response = requests.post(
                                    f"http://localhost:8509/testing/stop_test/{test['test_id']}"
                                )
                                if stop_response.status_code == 200:
                                    st.success("测试已停")
                                else:
                                    st.error("停止测试失败")
                    else:
                        st.info("当前没有运行中的测试")
                else:
                    st.error("获取测试状态失")
            except Exception as e:
                st.error(f"检查测试状态失 {str(e)}")
    
    with perf_tab3:
        st.subheader("📈 测试历史与结")
        
        if st.button("🔄 刷新测试历史", key="refresh_test_history"):
            try:
                response = requests.get("http://localhost:8509/testing/list_tests")
                if response.status_code == 200:
                    tests = response.json().get("tests", [])
                    
                    if tests:
                        # 按状态分组显
                        completed_tests = [t for t in tests if t.get("status") == "completed"]
                        failed_tests = [t for t in tests if t.get("status") == "failed"]
                        running_tests = [t for t in tests if t.get("status") == "running"]
                        
                        # 显示统计
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("已完", len(completed_tests))
                        with col_stat2:
                            st.metric("运行", len(running_tests))
                        with col_stat3:
                            st.metric("失败", len(failed_tests))
                        
                        style_metric_cards()
                        
                        # 显示详细结果
                        for test in completed_tests[:5]:  # 只显示最✅
                            with st.expander(f"测试 {test['test_id']} - {test.get('status', 'unknown')}"):
                                col_detail1, col_detail2 = st.columns(2)
                                
                                with col_detail1:
                                    st.write("**配置信息:**")
                                    config = test.get('config', {})
                                    st.write(f"- 用户✅ {config.get('users', 'N/A')}")
                                    st.write(f"- 持续时间: {config.get('run_time', 'N/A')}")
                                    st.write(f"- 集合: {config.get('collection_name', 'N/A')}")
                                
                                with col_detail2:
                                    st.write("**测试结果:**")
                                    metrics = test.get('metrics', {})
                                    if metrics:
                                        st.write(f"- 总请求数: {metrics.get('total_requests', 'N/A')}")
                                        st.write(f"- 失败✅ {metrics.get('failures', 'N/A')}")
                                        st.write(f"- 平均响应时间: {metrics.get('avg_response_time', 'N/A')}ms")
                                        st.write(f"- 每秒请求✅ {metrics.get('requests_per_second', 'N/A')}")
                                    else:
                                        st.write("暂无详细指标")
                    else:
                        st.info("暂无测试历史")
                else:
                    st.error("获取测试历史失败")
            except Exception as e:
                st.error(f"获取测试历史失败: {str(e)}")

with st.expander("🔧 系统状态与诊断", expanded=False):
    st.info("检查系统各模块状态和运行健康")
    
    col_diag1, col_diag2 = st.columns(2)
    
    with col_diag1:
        if st.button("📋 获取系统状", key="system_status_btn"):
            try:
                with st.spinner("正在检查系统状态.."):
                    response = requests.get("http://localhost:8509/system/status")
                    if response.status_code == 200:
                        status_data = response.json()
                        health = status_data.get("health", {})
                        overall_status = health.get("overall_status", "unknown")
                        
                        # 显示整体健康状
                        if overall_status == "healthy":
                            st.success(f"✅系统健康 (评分: {health.get('health_score', 0):.2f})")
                        elif overall_status == "degraded":
                            st.warning(f"⚠️ 系统降级运行 (评分: {health.get('health_score', 0):.2f})")
                        else:
                            st.error(f"✅系统异常 (评分: {health.get('health_score', 0):.2f})")
                        
                        status = status_data.get("status", {})
                        
                        # 显示核心服务状
                        st.subheader("🔧 核心服务状")
                        col_core1, col_core2 = st.columns(2)
                        
                        with col_core1:
                            # 嵌入模型状
                            embedding = status.get("embedding_model", {})
                            embedding_status = "✅正常" if embedding.get("available") else "✅异常"
                            st.write(f"🧠 嵌入模型: {embedding_status}")
                            if embedding.get("available"):
                                st.write(f"  - 模型: {embedding.get('model_name', 'N/A')}")
                                st.write(f"  - 设备: {embedding.get('device', 'N/A')}")
                            
                            # Milvus状
                            milvus = status.get("milvus", {})
                            milvus_status = "✅已连接" if milvus.get("connected") else "❌未连接"
                            st.write(f"🗄️ Milvus: {milvus_status}")
                            if milvus.get("connected"):
                                st.write(f"  - 地址: {milvus.get('host')}:{milvus.get('port')}")
                                st.write(f"  - 集合✅ {milvus.get('collections_count', 0)}")
                        
                        with col_core2:
                            # 聚类服务状
                            clustering = status.get("clustering_service", {})
                            clustering_status = "✅可用" if clustering.get("available") else "✅不可"
                            st.write(f"📊 聚类服务: {clustering_status}")
                            if clustering.get("available"):
                                st.write(f"  - HDBSCAN: {'✅' if clustering.get('hdbscan_available') else '❌'}")
                                st.write(f"  - Sklearn: {'✅' if clustering.get('sklearn_available') else '❌'}")
                            
                            # 分块系统状
                            chunking = status.get("chunking_system", {})
                            chunking_status = "✅可用" if chunking.get("available") else "✅不可"
                            st.write(f"✂️ 分块系统: {chunking_status}")
                            if chunking.get("available"):
                                llm_status = chunking.get("llm_status", {})
                                st.write(f"  - LLM支持: {'✅' if llm_status.get('api_client_available') else '❌'}")
                                st.write(f"  - 策略数: {chunking.get('strategies_count', 0)}")
                        
                        # 显示扩展功能状态
                        st.subheader("🚀 扩展功能状态")
                        col_ext1, col_ext2 = st.columns(2)
                        
                        with col_ext1:
                            # LLM配置状
                            llm_config = status.get("llm_config", {})
                            llm_config_status = "✅可用" if llm_config.get("available") else "✅不可"
                            st.write(f"🤖 LLM配置: {llm_config_status}")
                            if llm_config.get("available"):
                                active_config = llm_config.get("active_config")
                                if active_config:
                                    st.write(f"  - 激活 {active_config.get('id', 'N/A')}")
                                    st.write(f"  - 提供商 {active_config.get('provider', 'N/A')}")
                                else:
                                    st.write("  - 激活 ")
                                st.write(f"  - 总配置 {llm_config.get('total_configs', 0)}")
                            
                            # 搜索优化状
                            search_opt = status.get("search_optimization", {})
                            search_opt_status = "✅可用" if search_opt.get("available") else "✅不可"
                            st.write(f"🔍 搜索优化: {search_opt_status}")
                        
                        with col_ext2:
                            # 压测功能状
                            load_test = status.get("load_testing", {})
                            load_test_status = "✅可用" if load_test.get("available") else "✅不可"
                            st.write(f"✅压力测试: {load_test_status}")
                            if load_test.get("available"):
                                st.write(f"  - 活跃测试: {load_test.get('active_tests_count', 0)}")
                                st.write(f"  - 运行✅ {load_test.get('running_tests', 0)}")
                            
                            # 性能监控状态
                            perf_monitor = status.get("performance_monitor", False)
                            perf_status = "✅运行中" if perf_monitor else "❌未运行"
                            st.write(f"📈 性能监控: {perf_status}")
                            
                            # CLIP编码器状态
                            clip_status = "✅已加载" if status.get("clip_encoder") else "❌未加载"
                            st.write(f"🖼️ CLIP编码器: {clip_status}")
                        
                        # 显示配置摘要
                        st.subheader("⚙️ 当前配置")
                        config_info = status.get("config", {})
                        if config_info:
                            st.write(f"- Milvus地址: {config_info.get('milvus_host')}:{config_info.get('milvus_port')}")
                            st.write(f"- 默认集合: {config_info.get('collection_name')}")
                            st.write(f"- 多模 {'启用' if config_info.get('multimodal_enabled') else '禁用'}")
                            st.write(f"- 分块策略: {config_info.get('chunking_strategy')}")
                    else:
                        st.error(f"获取系统状态失✅ {response.status_code}")
            except Exception as e:
                st.error(f"连接失败: {str(e)}")
    
    with col_diag2:
        if st.button("🧪 运行集成测试", key="integration_test_btn"):
            try:
                with st.spinner("正在运行系统集成测试..."):
                    response = requests.post("http://localhost:8509/system/integration_test")
                    if response.status_code == 200:
                        test_data = response.json()
                        test_results = test_data.get("test_results", {})
                        summary = test_data.get("summary", {})
                        
                        # 显示测试摘要
                        overall = test_results.get("overall", {})
                        overall_status = overall.get("status", "unknown")
                        
                        if overall_status == "passed":
                            st.success(f"集成测试通过 ({summary.get('passed', 0)}/{summary.get('total', 0)})")
                        elif overall_status == "partial":
                            st.warning(f"⚠️ 部分测试通过 ({summary.get('passed', 0)}/{summary.get('total', 0)})")
                        else:
                            st.error(f"✅集成测试失败 ({summary.get('passed', 0)}/{summary.get('total', 0)})")
                        
                        st.write(overall.get("message", ""))
                        
                        # 显示详细测试结果
                        st.subheader("📋 详细测试结果")
                        
                        test_items = [
                            ("search_clustering", "🔍 搜索聚类"),
                            ("chunking_strategies", "✂️ 分块策略"),
                            ("performance_monitoring", "📈 性能监控"),
                            ("llm_integration", "🤖 LLM集成")
                        ]
                        
                        for test_key, test_name in test_items:
                            test_result = test_results.get(test_key, {})
                            status = test_result.get("status", "not_tested")
                            message = test_result.get("message", "")
                            
                            if status == "passed":
                                st.success(f"{test_name}: 通过")
                            elif status == "failed":
                                st.error(f"{test_name}: ✅失败")
                            else:
                                st.info(f"{test_name}: ⏸️ 未测")
                            
                            if message:
                                st.write(f"  {message}")
                        
                        # 显示成功
                        success_rate = summary.get("success_rate", 0)
                        st.metric("测试成功", f"{success_rate:.1%}")
                        
                    else:
                        st.error(f"集成测试失败: {response.status_code}")
            except Exception as e:
                st.error(f"运行集成测试失败: {str(e)}")
    
    # 添加系统重新加载功能
    st.markdown("---")
    st.subheader("🔄 系统维护")
    
    col_maint1, col_maint2 = st.columns(2)
    
    with col_maint1:
        if st.button("🔄 重新加载配置", key="reload_config_btn"):
            try:
                with st.spinner("正在重新加载系统配置..."):
                    response = requests.post("http://localhost:8509/system/reload_config")
                    if response.status_code == 200:
                        st.success("系统配置已重新加")
                        st.info("所有模块已重新初始化，新配置已生效")
                    else:
                        st.error("✅重新加载配置失败")
            except Exception as e:
                st.error(f"重新加载配置失败: {str(e)}")
    
    with col_maint2:
        if st.button("📊 导出系统报告", key="export_report_btn"):
            try:
                # 获取系统状
                status_response = requests.get("http://localhost:8509/system/status")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    
                    # 生成报告
                    report = {
                        "report_time": datetime.now().isoformat(),
                        "system_status": status_data,
                        "report_type": "system_health_report"
                    }
                    
                    # 提供下载
                    st.download_button(
                        label="📥 下载系统报告",
                        data=json.dumps(report, indent=2, ensure_ascii=False),
                        file_name=f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.error("获取系统状态失败，无法生成报告")
            except Exception as e:
                st.error(f"导出系统报告失败: {str(e)}")

# 显示当前配置信息
with st.expander("📋 当前配置信息", expanded=False):
    st.json(st.session_state.config)

# 页脚
st.markdown("---")
st.caption("© 2025 智能向量检索系| 版本 2.0.0 - 整合")
