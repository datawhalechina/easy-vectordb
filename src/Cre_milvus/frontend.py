import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import json
import random
from datetime import datetime
import logging
from typing import Dict, Any, Optional, List

# python -m streamlit run frontend.py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BACKEND_URL = "http://localhost:12089"
DEFAULT_TIMEOUT = 60

st.set_page_config(
    page_title="DataWhale-easyVectorDB", 
    layout="wide", 
    page_icon="🔍",
    initial_sidebar_state="collapsed"
)


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
    
    /* 聚类可视化样式 */
    .cluster-viz-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .cluster-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
        transition: all 0.3s ease;
    }
    
    .cluster-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .keyword-tag {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 4px 12px;
        border-radius: 16px;
        margin: 2px;
        display: inline-block;
        font-size: 0.85em;
        font-weight: 500;
        border: 1px solid #bbdefb;
    }
    
    .doc-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .doc-card:hover {
        border-color: #2196f3;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.1);
    }
    
    .quality-indicator {
        display: inline-flex;
        align-items: center;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
    }
    
    .quality-excellent {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 1px solid #c8e6c9;
    }
    
    .quality-good {
        background-color: #fff3e0;
        color: #f57c00;
        border: 1px solid #ffcc02;
    }
    
    .quality-fair {
        background-color: #ffebee;
        color: #d32f2f;
        border: 1px solid #ffcdd2;
    }
    
    @media (max-width: 768px) {
        .cluster-viz-container {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .cluster-card {
            padding: 0.75rem;
        }
        
        .doc-card {
            padding: 0.75rem;
        }
    }
    
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .loading-spinner {
        border: 3px solid #f3f3f3;
        border-top: 3px solid #3498db;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .viz-chart-container {
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        border-radius: 8px 8px 0px 0px;
        background-color: #f0f2f6;
        border: 1px solid #d0d4da;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 1px solid #ffffff;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🔍 DataWhale-easyVectorDB Enhanced</h1>
    <p style="text-align: center; color: white; margin: 0; opacity: 0.9;">
        智能向量数据库管理系统- 集成聚类搜索、智能分块、性能监控与压测
    </p>
</div>
""", unsafe_allow_html=True)

def safe_request(method: str, url: str, timeout: int = DEFAULT_TIMEOUT, **kwargs) -> Optional[requests.Response]:
    """Safe HTTP request with error handling"""
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None

# class GLMConfigManager:
#     """GLM配置状态管理器 - 通过后端API管理"""
    
#     def __init__(self):
#         self._config_cache = {}
#         self._cache_timestamp = 0
#         self._cache_duration = 30  # 缓存30秒
    
#     def get_config_status(self) -> Dict[str, Any]:
#         """获取GLM配置状态"""
#         try:
#             # 检查缓存
#             current_time = time.time()
#             if (self._config_cache and 
#                 current_time - self._cache_timestamp < self._cache_duration):
#                 return self._config_cache
            
#             # 从后端API获取配置状态
#             response = safe_request("GET", f"{BACKEND_URL}/glm/config", timeout=5)
#             if response and response.status_code == 200:
#                 config_data = response.json()
                
#                 # 更新缓存
#                 self._config_cache = config_data
#                 self._cache_timestamp = current_time
                
#                 return config_data
#             else:
#                 logger.error(f"获取GLM配置状态失败: {response.status_code if response else 'No response'}")
#                 return {"configured": False, "error": "无法连接后端服务"}
                
#         except Exception as e:
#             logger.error(f"获取GLM配置状态失败: {e}")
#             return {"configured": False, "error": str(e)}
    
#     def validate_config(self, config: Dict) -> bool:
#         """验证配置的有效性"""
#         return config.get("configured", False) and config.get("api_key_configured", False)
    
#     def get_config_ui_state(self) -> Dict[str, Any]:
#         """获取配置UI状态"""
#         config = self.get_config_status()
#         is_configured = config.get("configured", False)
        
#         return {
#             "is_configured": is_configured,
#             "should_expand": not is_configured,  # 未配置时展开
#             "status_message": self._get_status_message(config),
#             "status_type": self._get_status_type(config),
#             "config_preview": self._get_config_preview(config)
#         }
    
#     def _get_status_message(self, config: Dict) -> str:
#         """获取状态消息"""
#         if config.get("error"):
#             return f"⚠️ 获取配置状态失败: {config['error']}"
#         elif not config.get("configured", False):
#             return "⚠️ **重要提示**: GLM未配置，高级分块功能（PPL、MSP、边际采样）将不可用！"
#         else:
#             return "✅ GLM已配置，所有高级功能已启用"
    
#     def _get_status_type(self, config: Dict) -> str:
#         """获取状态类型"""
#         if config.get("error"):
#             return "error"
#         elif not config.get("configured", False):
#             return "warning"
#         else:
#             return "success"
    
#     def _get_config_preview(self, config: Dict) -> Dict[str, str]:
#         """获取配置预览信息"""
#         if not config.get("configured", False):
#             return {}
        
#         return {
#             "model": config.get("model_name", "N/A"),
#             "api_key_preview": config.get("api_key_preview", "N/A"),
#             "last_validated": config.get("last_validated", "N/A")[:19] if config.get("last_validated") else "N/A"
#         }
    
#     def save_config(self, model_name: str, api_key: str) -> bool:
#         """保存GLM配置"""
#         try:
#             # 调用后端API保存配置
#             response = safe_request(
#                 "POST", 
#                 f"{BACKEND_URL}/glm/config",
#                 json={
#                     "model_name": model_name,
#                     "api_key": api_key
#                 },
#                 timeout=10
#             )
            
#             if response and response.status_code == 200:
#                 result = response.json()
#                 if result.get("success"):
#                     logger.info("GLM配置保存成功")
#                     # 清除缓存，强制重新加载
#                     self.clear_cache()
#                     return True
#                 else:
#                     logger.error(f"GLM配置保存失败: {result.get('message', '未知错误')}")
#                     return False
#             else:
#                 logger.error(f"GLM配置保存请求失败: {response.status_code if response else 'No response'}")
#                 return False
                
#         except Exception as e:
#             logger.error(f"保存GLM配置失败: {e}")
#             return False
    
#     def clear_config(self) -> bool:
#         """清除GLM配置"""
#         try:
#             # 调用后端API清除配置
#             response = safe_request("DELETE", f"{BACKEND_URL}/glm/config", timeout=10)
            
#             if response and response.status_code == 200:
#                 result = response.json()
#                 if result.get("success"):
#                     logger.info("GLM配置清除成功")
#                     # 清除缓存
#                     self.clear_cache()
#                     return True
#                 else:
#                     logger.error(f"GLM配置清除失败: {result.get('message', '未知错误')}")
#                     return False
#             else:
#                 logger.error(f"GLM配置清除请求失败: {response.status_code if response else 'No response'}")
#                 return False
                
#         except Exception as e:
#             logger.error(f"清除GLM配置失败: {e}")
#             return False
    
#     def test_connection(self) -> Dict[str, Any]:
#         """测试GLM连接"""
#         try:
#             # 调用后端API测试连接
#             response = safe_request("POST", f"{BACKEND_URL}/glm/test-connection", timeout=15)
            
#             if response and response.status_code == 200:
#                 result = response.json()
#                 return {
#                     "valid": result.get("success", False),
#                     "message": result.get("message", "连接测试完成")
#                 }
#             else:
#                 return {
#                     "valid": False,
#                     "message": f"连接测试请求失败: {response.status_code if response else 'No response'}"
#                 }
                
#         except Exception as e:
#             return {"valid": False, "message": f"连接测试失败: {str(e)}"}
    
#     def clear_cache(self):
#         """清除缓存"""
#         self._config_cache = {}
#         self._cache_timestamp = 0

# def get_glm_config_status():
#     """获取GLM配置状态（向后兼容）"""
#     if not hasattr(st.session_state, 'glm_config_manager'):
#         st.session_state.glm_config_manager = GLMConfigManager()
#     return st.session_state.glm_config_manager.get_config_status()

def handle_api_error(response, operation_name: str = "操作") -> bool:
    
    if response.status_code == 200:
        return True
    
    try:
        error_data = response.json()
        error_message = error_data.get("detail", error_data.get("message", "未知错误"))
        
        # 根据错误内容判断错误类型
        error_type = "general"
        # if "glm" in error_message.lower() or "api" in error_message.lower():
        #     error_type = "glm_config"
        if "api" in error_message.lower():
            error_type = "api_error"
        elif "upload" in error_message.lower() or "file" in error_message.lower():
            error_type = "upload"
        elif "chunk" in error_message.lower() or "分块" in error_message.lower():
            error_type = "chunking"
        elif "connection" in error_message.lower() or "连接" in error_message.lower():
            error_type = "connection"
        
        st.error(f"{error_type}错误: {error_message}")
        
    except:
        st.error(f"{operation_name}失败，状态码: {response.status_code}")
    
    return False

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

def show_loading_state(message="处理中..."):
    """显示加载状态"""
    st.markdown(f"""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <span style="margin-left: 1rem; font-size: 1.1em; color: #666;">{message}</span>
    </div>
    """, unsafe_allow_html=True)

def show_empty_state(title="暂无数据", message="", icon="📭"):
    """显示空状态"""
    st.markdown(f"""
    <div style="text-align: center; padding: 3rem 1rem; color: #666;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
        <h3 style="color: #888; margin-bottom: 0.5rem;">{title}</h3>
        <p style="color: #aaa; margin: 0;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

def create_quality_badge(score, thresholds=(0.7, 0.5)):
    """创建质量评分徽章"""
    if score >= thresholds[0]:
        return f'<span class="quality-indicator quality-excellent">优秀 {score:.2f}</span>'
    elif score >= thresholds[1]:
        return f'<span class="quality-indicator quality-good">良好 {score:.2f}</span>'
    else:
        return f'<span class="quality-indicator quality-fair">一般 {score:.2f}</span>'

def optimize_plotly_chart(fig, height=400):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

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
        "chunking": {  
            "strategy": "traditional",
            "chunk_length": 512,
            "overlap": 50
        },
        "multimodal": {  
            "enable_image": False,
            "clip_model": "ViT-B/32",
            "image_formats": ["jpg", "jpeg", "png", "bmp"]
        }
    }

if "last_search" not in st.session_state:
    st.session_state.last_search = None

# if not hasattr(st.session_state, 'glm_config_manager'):
#     st.session_state.glm_config_manager = GLMConfigManager()

# glm_manager = st.session_state.glm_config_manager
# ui_state = glm_manager.get_config_ui_state()

# with st.expander("🤖 GLM-4.5-flash 配置 - 高级分块功能必需", expanded=ui_state["should_expand"]):
#     # 显示状态消息
#     if ui_state["status_type"] == "warning":
#         st.warning(ui_state["status_message"])
#         st.info("💡 请先配置GLM-4.5-flash模型以启用完整功能")
#     else:
#         st.success(ui_state["status_message"])
    
#     col_glm1, col_glm2 = st.columns(2)
    
#     with col_glm1:
#         st.markdown("**当前GLM配置状态**")
#         if ui_state["is_configured"]:
#             st.success("✅ GLM已配置")
#             config_preview = ui_state["config_preview"]
#             st.write(f"- 模型: {config_preview.get('model', 'N/A')}")
#             st.write(f"- API密钥: {config_preview.get('api_key_preview', 'N/A')}")
#             if config_preview.get("last_validated") != "N/A":
#                 st.write(f"- 最后验证：{config_preview.get('last_validated', 'N/A')}")
            
#             # 连接测试按钮
#             if st.button("🔍 测试连接", key="test_glm_connection_top"):
#                 with st.spinner("测试GLM连接..."):
#                     test_result = glm_manager.test_connection()
#                     if test_result.get("valid", False):
#                         st.success(f"✅ {test_result.get('message', '连接成功')}")
#                     else:
#                         st.error(f"❌ {test_result.get('message', '连接失败')}")
            
#             # 清除配置按钮
#             if st.button("🗑️ 清除配置", key="clear_glm_config_top"):
#                 with st.spinner("清除GLM配置..."):
#                     if glm_manager.clear_config():
#                         st.success("✅ GLM配置已清除")
#                         st.rerun()
#                     else:
#                         st.error("❌ 清除配置失败")
#         else:
#             st.error("❌ GLM未配置")
    
#     with col_glm2:
#         st.markdown("**GLM-4.5-flash 配置**")
        
#         # GLM配置表单（简化版）
#         with st.form("glm_config_form_top"):
#             model_name = st.text_input(
#                 "模型名称",
#                 value="glm-4.5-flash",
#                 help="GLM模型名称，默认为glm-4.5-flash"
#             )
            
#             api_key = st.text_input(
#                 "智谱AI API密钥",
#                 type="password",
#                 help="请输入您的智谱AI API密钥",
#                 placeholder="请输入API密钥..."
#             )
            
#             # API密钥验证按钮
#             col_validate, col_save = st.columns(2)
            
#             with col_validate:
#                 validate_clicked = st.form_submit_button("🔍 验证密钥")
            
#             with col_save:
#                 save_clicked = st.form_submit_button("💾 保存配置", type="primary")
            
#             if validate_clicked:
#                 if not api_key:
#                     st.error("请输入API密钥")
#                 else:
#                     with st.spinner("验证API密钥..."):
#                         # 简单的API密钥格式验证
#                         if len(api_key) < 10:
#                             st.error("❌ API密钥格式不正确，长度过短")
#                         elif not api_key.strip():
#                             st.error("❌ API密钥不能为空")
#                         else:
#                             st.success("✅ API密钥格式验证通过")
            
#             if save_clicked:
#                 if not model_name or not api_key:
#                     st.error("请填写模型名称和API密钥")
#                 else:
#                     with st.spinner("保存GLM配置..."):
#                         if glm_manager.save_config(model_name, api_key):
#                             st.success("✅ GLM配置保存成功")
#                             st.info("🔄 GLM配置已激活，高级分块功能现在可用")
#                             time.sleep(0.5)
#                             st.rerun()
#                         else:
#                             st.error("❌ GLM配置保存失败")

with st.sidebar:
    st.header("🚀 系统快速状态")
    
    status_response = safe_request("GET", f"{BACKEND_URL}/system/status", timeout=DEFAULT_TIMEOUT)
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
            
            clustering_status = status.get("clustering_service", {})
            clustering_ok = clustering_status.get("available", False)
            model_name = clustering_status.get("model", "未加载")
            st.write(f"📊 聚类服务: {'✅' if clustering_ok else '❌'} ({model_name})")
            
            # GLM状态显示
            # if hasattr(st.session_state, 'glm_config_manager'):
            #     sidebar_glm_manager = st.session_state.glm_config_manager
            # else:
            #     sidebar_glm_manager = GLMConfigManager()
            #     st.session_state.glm_config_manager = sidebar_glm_manager
            
            # sidebar_glm_status = sidebar_glm_manager.get_config_status()
            # if sidebar_glm_status.get("configured", False):
            #     st.markdown("**GLM配置:**")
            #     st.write(f"🤖 {sidebar_glm_status.get('model_name', 'N/A')}")
            #     st.write(f"🔑 已配置API密钥")
            # else:
            #     st.info("🤖 GLM未配置")
        except json.JSONDecodeError:
            st.error("❌ 响应格式错误")
    else:
        st.error("❌ 无法连接后端")
        if status_response:
            st.caption(f"状态码: {status_response.status_code}")
    
    st.markdown("---")
    
    
    st.markdown("**快速操作**")
    if st.button("🔄 刷新状态", key="sidebar_refresh"):
        st.rerun()
    
    if st.button("🧪 快速测试", key="sidebar_test"):
        with st.spinner("运行快速测试..."):
            test_response = safe_request("POST", f"{BACKEND_URL}/system/integration_test", timeout=DEFAULT_TIMEOUT)
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
        
        # 添加GLM依赖提示
        # glm_status = glm_manager.get_config_status()
        # if not glm_status.get("configured", False):
        #     st.warning("⚠️ 注意：meta_ppl、msp、margin_sampling策略需要GLM配置才能正常工作")
        
        col6, col7, col8 = st.columns(3)

        # 初始化配置
        if 'chunking_config' not in st.session_state:
            st.session_state.chunking_config = {
                "strategy": "traditional",
                "chunk_length": 512,
                "ppl_threshold": 0.3,
                "confidence_threshold": 0.7,
                "similarity_threshold": 0.8,
                "overlap": 50,
                "min_chunk_size": 100
            }

        with col6:
            chunking_strategy = st.selectbox(
                "切分策略",
                ["traditional", "meta_ppl", "margin_sampling", "msp", "semantic"],
                index=["traditional", "meta_ppl", "margin_sampling", "msp", "semantic"].index(
                    st.session_state.chunking_config.get("strategy", "traditional")
                ),
                help="选择文本切分策略...",
                key="strategy_selector"
            )
            st.session_state.chunking_config["strategy"] = chunking_strategy

        with col7:
            chunk_length = st.number_input(
                "块长度",
                value=st.session_state.chunking_config.get("chunk_length", 512),
                min_value=100,
                max_value=2048,
                help="文本块的最大长度",
                key="chunk_length_input"
            )
            st.session_state.chunking_config["chunk_length"] = chunk_length

        with col8:
            current_strategy = st.session_state.chunking_config["strategy"]
            
            if current_strategy == "meta_ppl":
                ppl_threshold = st.slider(
                    "PPL阈值",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.chunking_config.get("ppl_threshold", 0.3),
                    step=0.1,
                    help="PPL困惑度切分的阈值",
                    key="ppl_threshold_slider"
                )
                st.session_state.chunking_config["ppl_threshold"] = ppl_threshold
                
            elif current_strategy == "msp":
                confidence_threshold = st.slider(
                    "置信度阈值",
                    min_value=0.5,
                    max_value=0.95,
                    value=st.session_state.chunking_config.get("confidence_threshold", 0.7),
                    step=0.05,
                    help="MSP切分的置信度阈值",
                    key="confidence_threshold_slider"
                )
                st.session_state.chunking_config["confidence_threshold"] = confidence_threshold
                
            elif current_strategy == "semantic":
                similarity_threshold = st.slider(
                    "相似度阈值",
                    min_value=0.5,
                    max_value=0.95,
                    value=st.session_state.chunking_config.get("similarity_threshold", 0.8),
                    step=0.05,
                    help="语义切分的相似度阈值",
                    key="similarity_threshold_slider"
                )
                st.session_state.chunking_config["similarity_threshold"] = similarity_threshold
                
                min_chunk_size = st.number_input(
                    "最小块大小",
                    value=st.session_state.chunking_config.get("min_chunk_size", 100),
                    min_value=50,
                    max_value=200,
                    key="min_chunk_size_input"
                )
                st.session_state.chunking_config["min_chunk_size"] = min_chunk_size
                
            elif current_strategy == "traditional":
                overlap = st.slider(
                    "重叠长度",
                    min_value=0,
                    max_value=200,
                    value=st.session_state.chunking_config.get("overlap", 50),
                    step=10,
                    help="传统切分的重叠长度",
                    key="overlap_slider"
                )
                st.session_state.chunking_config["overlap"] = overlap
        
        ppl_threshold = st.session_state.chunking_config.get("ppl_threshold", 0.3)
        confidence_threshold = st.session_state.chunking_config.get("confidence_threshold", 0.7)
        similarity_threshold = st.session_state.chunking_config.get("similarity_threshold", 0.8)
        overlap = st.session_state.chunking_config.get("overlap", 50)
        
        # st.subheader("GLM配置状态（用于高级分块策略）")
        
        # # 获取当前GLM配置状态
        # if hasattr(st.session_state, 'glm_config_manager'):
        #     form_glm_manager = st.session_state.glm_config_manager
        # else:
        #     form_glm_manager = GLMConfigManager()
        #     st.session_state.glm_config_manager = form_glm_manager
        
        # form_ui_state = form_glm_manager.get_config_ui_state()
        
        col_glm_status1, col_glm_status2 = st.columns(2)
        
        # with col_glm_status1:
        #     st.markdown("**当前GLM配置状态**")
        #     if form_ui_state["is_configured"]:
        #         st.success("✅ GLM已配置")
        #         config_preview = form_ui_state["config_preview"]
        #         st.write(f"- 模型: {config_preview.get('model', 'N/A')}")
        #         st.write(f"- API密钥: {config_preview.get('api_key_preview', 'N/A')}")
        #     else:
        #         st.error("❌ GLM未配置")
        
        # with col_glm_status2:
        #     if form_ui_state["is_configured"]:
        #         st.success("🎉 高级分块功能已可用")
        #     else:
        #         st.warning("⚠️ 高级分块功能不可用，请先配置GLM")

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
                handle_api_error(response, "配置保存")



# 上传文件区
with st.expander("📁 上传数据文件区", expanded=True):
    st.info("请全选文件夹下所有文件上传，并输入一个文件夹名，系统会自动保存到该目录")
    folder_names = st.text_input("请输入目标文件夹名（如：0240501）", key="folder_name")
    uploaded_files = st.file_uploader(
        "选择文件夹中的文件（支持csv, md, pdf, txt, jpg, png）", 
        accept_multiple_files=True, 
        type=["csv", "md", "pdf", "txt", "jpg", "jpeg", "png"]
    )
    
    if st.button("⬆️ 上传并构建向量库", key="upload_btn"):
        if not folder_names:
            st.warning("⚠️ 请先输入目标文件夹名")
        elif not uploaded_files:
            st.warning("⚠️ 请先选择要上传的文件")
        else:
            # 创建进度显示区域
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
            with status_container:
                result_placeholder = st.empty()
            
            try:
                status_text.text("📤 正在上传文件...")
                uploaded_results = []
                total_files = len(uploaded_files)
                
                for i, file in enumerate(uploaded_files):
                    progress_percentage = (i / total_files) * 50  
                    progress_bar.progress(progress_percentage / 100)
                    status_text.text(f"📤 正在上传文件 {i+1}/{total_files}: {file.name}")
                    
                    file.seek(0)  
                    file_content = file.read()
                    
                    from io import BytesIO
                    file_obj = BytesIO(file_content)
                    file_obj.name = file.name
                    
                    files = {"file": (file.name, file_obj, file.type)}
                    data = {"folder_name": folder_names}  
                    
                    response = requests.post(
                        f"{BACKEND_URL}/upload",
                        files=files,  
                        data=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        uploaded_results.append(result)
                        logger.info(f"文件上传成功: {file.name}")
                    else:
                        logger.error(f"文件上传失败: {file.name}, 状态码: {response.status_code}")
                        with result_placeholder.container():
                            handle_api_error(response, f"文件 {file.name} 上传")
                        continue  
                
                if not uploaded_results:
                    with result_placeholder.container():
                        st.error("❌ 没有文件成功上传")
                
                
                progress_bar.progress(0.5)  
                status_text.text("✅ 文件上传完成，开始处理...")
                
                last_result = uploaded_results[-1]
                tracking_id = last_result.get("tracking_id")
                
                if tracking_id:
                    status_text.text("🔄 正在处理数据，请稍候...")
                    # max_attempts = 300  
                    # attempt = 0
                    
                    # while attempt < max_attempts:
                    try:
                        progress_response = safe_request("GET", f"{BACKEND_URL}/progress/{tracking_id}")
                        if progress_response and progress_response.status_code == 200:
                            progress_data = progress_response.json()
                            # if progress_data.get("status") == "not_found":
                            #     break
                            
                            # 更新进度条 (50% + 处理进度的50%)
                            processing_percentage = progress_data.get("progress_percentage", 0)
                            total_progress = 50 + (processing_percentage * 0.5)
                            progress_bar.progress(min(total_progress / 100, 1.0))
                            
                            # 更新状态文本
                            current_status = progress_data.get("current_status", "处理中")
                            processed = progress_data.get("processed_items", 0)
                            total = progress_data.get("total_items", 0)
                            
                            if total > 0:
                                status_text.text(f"📊 {current_status}: {processed}/{total} ({processing_percentage:.1f}%)")
                            else:
                                status_text.text(f"📊 {current_status}")
                            
                            # 检查是否完成
                            # if current_status in ["completed", "failed"]:
                            #     break
                        else:
                            logger.warning(f"无法获取进度状态: {tracking_id}")
                            # break
                    except Exception as e:
                        logger.error(f"获取进度状态失败: {e}")
                        # break
                    
                    time.sleep(1)  # 每秒检查一次
                    # attempt += 1
                
                # 完成进度条
                progress_bar.progress(1.0)
                
                # 显示最终结果
                with result_placeholder.container():
                    successful_uploads = len(uploaded_results)
                    failed_uploads = len(uploaded_files) - successful_uploads
                    
                    if successful_uploads > 0:
                        if failed_uploads == 0:
                            st.success(f"✅ 成功上传 {successful_uploads} 个文件")
                        else:
                            st.warning(f"⚠️ 部分成功：上传了 {successful_uploads} 个文件，{failed_uploads} 个文件失败")
                        
                        # 检查向量化状态
                        vectorized_count = sum(1 for result in uploaded_results if result.get("vectorized", False))
                        if vectorized_count > 0:
                            st.info("📊 数据已向量化，可以进行检索查询")
                        else:
                            st.warning("⚠️ 向量化存储未完成，可能影响检索功能")
                        
                        # 显示处理时间
                        if tracking_id and 'progress_data' in locals():
                            processing_time = progress_data.get("processing_time", 0)
                            if processing_time > 0:
                                st.info(f"⏱️ 处理耗时: {processing_time:.2f} 秒")
                        
                        st.balloons()
                        
                        # 2. 更新配置文件中的 data_location 字段
                        if 'folder_name' in locals() and folder_names:
                            config_update = {"data": {"data_location": f"./data/upload/{folder_names}"}}
                            st.session_state.config["data"] = config_update["data"]
                            
                            # 发送更新请求
                            update_response = safe_request("POST", f"{BACKEND_URL}/update_config", json=config_update)
                            if not update_response or update_response.status_code != 200:
                                st.warning("⚠️ 配置更新失败，但文件上传成功")
                    else:
                        st.error("❌ 所有文件上传失败")

            except requests.exceptions.Timeout:
                with result_placeholder.container():
                    st.error("❌ 上传超时，请检查网络连接或减少文件数量")
            except requests.exceptions.ConnectionError:
                with result_placeholder.container():
                    st.error("❌ 无法连接到服务器，请确保后端服务正在运行")
            except Exception as e:
                with result_placeholder.container():
                    logger.error(f"上传错误: {e}")
                    st.error(f"❌ 上传过程中发生错误: {str(e)}")
            finally:
                # 清理进度显示
                status_text.text("✅ 处理完成")

st.markdown("---")

with st.container():
    st.markdown("### 检索与可视化")
    st.markdown("---")

    question = st.text_input("请输入检索问题", key="search_question")
    col_choice = st.selectbox(
        "聚类算法", 
        ["hdbscan", "kmeans"],
        index=0 if st.session_state.config["search"]["col_choice"] == "hdbscan" else 1,
        key="col_choice"
    )
    
    # 添加结果展示选项
    col_display, col_viz = st.columns(2)
    with col_display:
        result_display = st.radio("结果展示方式", ["摘要视图", "详细视图"], index=0, horizontal=True)
    with col_viz:
        enable_visualization = st.checkbox("启用聚类可视化", value=True, help="生成聚类散点图、饼图等可视化分析")
    
    if st.button("🚀 开始检索与可视化", key="search_btn", type="primary"):
        if not question:
            st.warning("⚠️ 请输入检索问题！")
        else:
            with st.spinner("检索中，请稍候..."):
                try:
                    # 1. 执行搜索
                    search_response = requests.post(
                        f"{BACKEND_URL}/search",
                        json={
                            "question": question, 
                            "col_choice": col_choice,
                            "collection_name": st.session_state.config["milvus"]["collection_name"],
                            "enable_visualization": enable_visualization
                        },
                        timeout=DEFAULT_TIMEOUT  # 添加超时设置
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
                            
                            st.success(f"✅ 检索完成！找到 {cluster_count} 个聚类，共 {doc_count} 个文档 (用时: {execution_time:.2f}s, 方法: {clustering_method})")
                            
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
                            
                            # 聚类可视化展示
                            if "visualization_data" in search_result:
                                st.subheader("🎨 聚类可视化分析")
                                
                                # 显示可视化生成时间
                                if "visualization_time" in search_result:
                                    st.caption(f"⏱️ 可视化生成耗时: {search_result['visualization_time']:.2f}秒")
                                
                                viz_data = search_result["visualization_data"]
                                
                                # 创建可视化选项卡
                                viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                                    "📊 聚类分布图", "🥧 聚类大小", "🔥 相似度热力图", "📋 聚类摘要"
                                ])
                                
                                with viz_tab1:
                                    # 聚类散点图
                                    if "scatter_plot" in viz_data and viz_data["scatter_plot"]["x"]:
                                        scatter_data = viz_data["scatter_plot"]
                                        
                                        total_points = len(scatter_data["x"])
                                        max_points = 1000  # 最大显示点数
                                        
                                        if total_points > max_points:
                                            st.info(f"⚡ 数据点较多({total_points}个)，为提升性能已采样显示{max_points}个点")
                                            indices = random.sample(range(total_points), max_points)
                                            scatter_data = {
                                                "x": [scatter_data["x"][i] for i in indices],
                                                "y": [scatter_data["y"][i] for i in indices],
                                                "cluster_ids": [scatter_data["cluster_ids"][i] for i in indices],
                                                "contents": [scatter_data["contents"][i] for i in indices],
                                                "distances": [scatter_data["distances"][i] for i in indices],
                                                "method_used": scatter_data.get("method_used", "unknown"),
                                                "total_points": total_points
                                            }
                                        
                                        # 创建散点图
                                        fig = px.scatter(
                                            x=scatter_data["x"],
                                            y=scatter_data["y"],
                                            color=[f"聚类 {cid}" for cid in scatter_data["cluster_ids"]],
                                            hover_data={
                                                "内容": scatter_data["contents"],
                                                "距离": scatter_data["distances"]
                                            },
                                            title=f"文档聚类分布图 (降维方法: {scatter_data.get('method_used', 'unknown')})",
                                            labels={"x": "维度 1", "y": "维度 2"}
                                        )
                                        
                                        # 优化图表样式和性能
                                        fig = optimize_plotly_chart(fig, height=500)
                                        fig.update_traces(
                                            marker=dict(size=8, opacity=0.7, line=dict(width=1, color='white')),
                                            selector=dict(mode='markers')
                                        )
                                        fig.update_layout(
                                            legend=dict(
                                                orientation="v",
                                                yanchor="top",
                                                y=1,
                                                xanchor="left",
                                                x=1.02
                                            )
                                        )
                                        
                                        with st.container():
                                            st.markdown('<div class="viz-chart-container">', unsafe_allow_html=True)
                                            st.plotly_chart(fig, use_container_width=True)
                                            st.markdown('</div>', unsafe_allow_html=True)
                                        
                                        st.info(f"📍 共 {scatter_data.get('total_points', 0)} 个文档点，使用 {scatter_data.get('method_used', 'unknown')} 降维方法")
                                    else:
                                        st.warning("⚠️ 无法生成散点图：缺少向量数据")
                                
                                with viz_tab2:
                                    # 聚类大小饼图
                                    if "size_chart" in viz_data and viz_data["size_chart"]["values"]:
                                        size_data = viz_data["size_chart"]
                                        
                                        fig = px.pie(
                                            values=size_data["values"],
                                            names=size_data["labels"],
                                            title="聚类大小分布",
                                            color_discrete_sequence=size_data.get("colors", px.colors.qualitative.Set3)
                                        )
                                        
                                        fig.update_traces(
                                            textposition='inside', 
                                            textinfo='percent+label',
                                            hovertemplate='<b>%{label}</b><br>文档数: %{value}<br>占比: %{percent}<extra></extra>'
                                        )
                                        fig = optimize_plotly_chart(fig, height=400)
                                        
                                        with st.container():
                                            st.markdown('<div class="viz-chart-container">', unsafe_allow_html=True)
                                            st.plotly_chart(fig, use_container_width=True)
                                            st.markdown('</div>', unsafe_allow_html=True)
                                        
                                        # 显示详细统计
                                        st.markdown("**聚类大小统计：**")
                                        for label, value in zip(size_data["labels"], size_data["values"]):
                                            st.write(f"- {label}: {value} 个文档")
                                    else:
                                        st.warning("⚠️ 无法生成饼图：缺少聚类数据")
                                
                                with viz_tab3:
                                    # 聚类相似度热力图
                                    if "heatmap" in viz_data and viz_data["heatmap"]["matrix"]:
                                        heatmap_data = viz_data["heatmap"]
                                        
                                        fig = px.imshow(
                                            heatmap_data["matrix"],
                                            labels=dict(x="聚类", y="聚类", color="相似度"),
                                            x=heatmap_data["labels"],
                                            y=heatmap_data["labels"],
                                            title="聚类间相似度热力图",
                                            color_continuous_scale="RdYlBu_r",
                                            aspect="auto"
                                        )
                                        
                                        fig = optimize_plotly_chart(fig, height=400)
                                        fig.update_layout(
                                            xaxis_title="聚类",
                                            yaxis_title="聚类"
                                        )
                                        
                                        with st.container():
                                            st.markdown('<div class="viz-chart-container">', unsafe_allow_html=True)
                                            st.plotly_chart(fig, use_container_width=True)
                                            st.markdown('</div>', unsafe_allow_html=True)
                                        
                                        st.info("💡 颜色越深表示聚类间相似度越高")
                                    else:
                                        st.warning("⚠️ 无法生成热力图：聚类数量不足")
                                
                                with viz_tab4:
                                    # 聚类摘要信息
                                    if "cluster_summary" in viz_data:
                                        summary = viz_data["cluster_summary"]
                                        
                                        st.markdown("**聚类总体信息：**")
                                        col_s1, col_s2, col_s3 = st.columns(3)
                                        with col_s1:
                                            st.metric("总聚类数", summary.get("total_clusters", 0), 
                                                     help="检索结果被分为多少个不同的主题聚类")
                                        with col_s2:
                                            st.metric("总文档数", summary.get("total_documents", 0),
                                                     help="所有聚类中包含的文档总数")
                                        with col_s3:
                                            st.metric("平均聚类大小", f"{summary.get('avg_cluster_size', 0):.1f}",
                                                     help="每个聚类平均包含的文档数量")
                                        
                                        # 应用自定义样式
                                        style_metric_cards(background_color="#f8f9fa", border_left_color="#28a745")
                                        
                                        # 显示每个聚类的详细信息
                                        st.markdown("**聚类详细信息：**")
                                        
                                        # 按聚类大小排序
                                        sorted_details = sorted(
                                            summary.get("cluster_details", []), 
                                            key=lambda x: x['size'], 
                                            reverse=True
                                        )
                                        
                                        for detail in sorted_details:
                                            # 创建聚类卡片
                                            cluster_quality = 1 - detail['avg_distance'] if detail['avg_distance'] < 1 else 0
                                            quality_badge = create_quality_badge(cluster_quality)
                                            
                                            with st.container():
                                                st.markdown(f"#### 聚类 {detail['cluster_id']} - {detail['size']}个文档")
                                                st.markdown('<div class="cluster-card">', unsafe_allow_html=True)
                                                
                                                # 聚类统计信息
                                                col_stat1, col_stat2, col_stat3 = st.columns(3)
                                                with col_stat1:
                                                    st.metric("文档数量", detail['size'])
                                                with col_stat2:
                                                    st.metric("平均距离", f"{detail['avg_distance']:.3f}")
                                                with col_stat3:
                                                    st.markdown(f"**质量评分：** {quality_badge}", unsafe_allow_html=True)
                                                
                                                # 关键词展示
                                                if detail.get("keywords"):
                                                    st.markdown("**🏷️ 聚类关键词：**")
                                                    keywords_html = " ".join([
                                                        f'<span class="keyword-tag">{keyword}</span>' 
                                                        for keyword in detail["keywords"]
                                                    ])
                                                    st.markdown(keywords_html, unsafe_allow_html=True)
                                                
                                                # 代表性内容
                                                if detail.get("representative_content"):
                                                    st.markdown("**📄 代表性内容：**")
                                                    st.markdown(f"""
                                                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #007bff; margin-top: 0.5rem;">
                                                        {detail["representative_content"]}
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                
                                                st.markdown('</div>', unsafe_allow_html=True)
                                    else:
                                        st.warning("⚠️ 无法显示聚类摘要：缺少摘要数据")
                        
                            elif "visualization_error" in search_result:
                                st.warning("⚠️ 聚类可视化生成失败")
                                with st.container():
                                    st.markdown("查看错误详细")
                                    st.error(search_result["visualization_error"])
                                    st.info("💡 可视化失败不影响基础搜索功能，您仍可以查看下方的检索结果")
                            
                            else:
                                st.info("ℹ️ 未启用聚类可视化功能，如需查看可视化分析，请在搜索时启用该功能")
                            
                            st.subheader("📄 检索结果详情")
                            
                            tab1, tab2 = st.tabs(["📋 文档列表", "🗂️ 聚类视图"])
                            
                            with tab1:
                                all_docs = []
                                for cluster_idx, cluster in enumerate(search_result["clusters"]):
                                    for doc in cluster["documents"]:
                                        doc_with_cluster = doc.copy()
                                        doc_with_cluster["cluster_id"] = cluster.get("cluster_id", cluster_idx)
                                        all_docs.append(doc_with_cluster)
                                
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
                                
                                style_metric_cards()
                                
                                sort_option = st.selectbox(
                                    "集群排序方式",
                                    ["按平均距离", "按集群大小", "按集群ID"],
                                    key="cluster_sort_option"
                                )
                                
                                clusters_to_display = search_result["clusters"].copy()
                                if sort_option == "按平均距离":
                                    clusters_to_display.sort(key=lambda x: x.get("avg_distance", 0))
                                elif sort_option == "按集群大小":
                                    clusters_to_display.sort(key=lambda x: len(x["documents"]), reverse=True)
                                else:  
                                    clusters_to_display.sort(key=lambda x: x.get("cluster_id", 0))
                                
                                for i, cluster in enumerate(clusters_to_display):
                                    cluster_id = cluster.get('cluster_id', i)
                                    cluster_size = len(cluster['documents'])
                                    avg_distance = cluster.get('avg_distance', 0.0)
                                    
                                    st.subheader(f"🔍 集群 #{cluster_id}")
                                    
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("文档数量", cluster_size)
                                    with col_b:
                                        st.metric("平均距离", f"{avg_distance:.4f}")
                                    with col_c:
                                        
                                        quality_score = max(0, min(1, avg_distance)) if avg_distance > 0 else 0
                                        quality_label = "优秀" if quality_score > 0.7 else "良好" if quality_score > 0.5 else "一"
                                        st.metric("质量评分", f"{quality_score:.2f}", delta=quality_label)
                                    
                                    # 显示集群文档
                                    st.markdown(f"**📋 集群 #{cluster_id} 文档列表 ({cluster_size} 个文档):**")
                                    
                                    # 使用容器显示文档，而不是嵌套expander
                                    if i == 0:  # 默认展开第一个集群
                                        show_docs = True
                                    else:
                                        show_docs = st.checkbox(f"显示集群 #{cluster_id} 的文档", key=f"show_cluster_{cluster_id}")
                                    
                                    if show_docs:
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
                            st.info("ℹ️ 未找到相关文档")
                        
                        if col_choice.lower() == "hdbscan" and "clusters" in search_result and search_result["clusters"]:
                            vis_response = requests.post(
                                f"{BACKEND_URL}/visualization",
                                json={"collection_name": st.session_state.config["milvus"]["collection_name"]}
                            )
                            
                            if vis_response.status_code == 200:
                                vis_data = vis_response.json()
                                
                                if isinstance(vis_data, list) and vis_data:
                                    df = pd.DataFrame(vis_data)
                                    
                                    # 显示可视化图
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
                                    
                                    # 显示原始数据
                                    if st.checkbox("显示原始数据", key="show_raw_data"):
                                        st.dataframe(df)
                                else:
                                    st.info("ℹ️ 无可视化数据")
                            else:
                                st.error(f"可视化失败: {vis_response.text}")
                    else:
                        st.error(f"❌ 检索失败: {search_response.text}")
                except Exception as e:
                    st.error(f"❌ 连接后端失败: {str(e)}")

st.markdown("---")

# 新增功能面板
with st.expander("🧪 文本切分测试", expanded=False):
    st.info("测试不同的文本切分策略效果")
    
    # 获取可用策略和状态
    try:
        strategies_response = requests.get(f"{BACKEND_URL}/chunking/strategies")
        if strategies_response.status_code == 200:
            strategies_data = strategies_response.json().get("strategies", [])
            
            # 显示策略状态
            st.markdown("**可用策略状态**")
            col_status1, col_status2 = st.columns(2)
            
            # 不再使用，但保留也不影响
            with col_status1:
                for strategy in strategies_data[:3]:
                    name = strategy.get("display_name", strategy.get("name", ""))
                    if strategy.get("llm_required", False):
                        status = "🟢 可用" if strategy.get("available", False) else "🔴 需要LLM配置"
                    else:
                        status = "🟢 可用"
                    st.write(f"- {name}: {status}")
            
            with col_status2:
                for strategy in strategies_data[3:]:
                    name = strategy.get("display_name", strategy.get("name", ""))
                    if strategy.get("llm_required", False):
                        status = "🟢 可用" if strategy.get("available", False) else "🔴 需要LLM配置"
                    else:
                        status = "🟢 可用"
                    st.write(f"- {name}: {status}")
        else:
            st.warning("无法获取策略状态")
    except Exception as e:
        st.warning(f"获取策略状态失败 {str(e)}")
    
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
            chunk_size = st.number_input("块大小", value=512, min_value=100, max_value=2048, key="test_chunk_size")
            overlap = st.number_input("重叠大小", value=50, min_value=0, max_value=200, key="test_overlap")
        elif test_strategy == "meta_ppl":
            ppl_threshold = st.slider("PPL阈值", 0.0, 1.0, 0.3, 0.1, key="test_ppl_threshold")
        elif test_strategy == "msp":
            confidence_threshold = st.slider("置信度阈值", 0.5, 0.95, 0.7, 0.05, key="test_confidence_threshold")
        elif test_strategy == "semantic":
            similarity_threshold = st.slider("相似度阈值", 0.5, 0.95, 0.8, 0.05, key="test_similarity_threshold")
    
    if st.button("🔄 执行切分测试", key="chunking_test_btn"):
        if test_text:
            # 检查LLM依赖策略（不再需要）
            llm_required_strategies = ["msp", "meta_ppl"]
            # if test_strategy in llm_required_strategies:
                # try:
                #     configs_response = requests.get(f"{BACKEND_URL}/llm/configs")
                #     if configs_response.status_code == 200:
                        # summary = configs_response.json().get("summary", {})
                        # if not summary.get("active_config"):
                        #     st.error(f"策略 '{test_strategy}' 需要LLM配置，请先在上方配置LLM")
                        #     st.stop()
                # except Exception:
                #     st.warning("⚠️ 无法检查LLM配置状态")
            
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
                        f"{BACKEND_URL}/chunking/process",
                        json={
                            "text": test_text,
                            "strategy": test_strategy,
                            "params": test_params
                        },
                        timeout=DEFAULT_TIMEOUT  
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
                        
                        # 显示每个文本块
                        st.markdown("**切分结果:**")
                        for i, chunk in enumerate(chunks):
                            st.markdown(f"**文本块 #{i+1} (长度: {len(chunk)}):**")
                            st.text_area(f"文本块 #{i+1}", value=chunk, height=100, key=f"chunk_{i}", label_visibility="collapsed")
                    else:
                        error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
                        error_msg = error_data.get("message", response.text)
                        st.error(f"❌ 切分失败: {error_msg}")
                        
                except Exception as e:
                    st.error(f"❌ 连接后端失败: {str(e)}")
        else:
            st.warning("⚠️ 请输入测试文本")

with st.expander("🖼️ 以文搜图功能", expanded=False):
    st.info("使用文本描述搜索相关图像")
    
    if st.session_state.config.get("multimodal", {}).get("enable_image", False):
        search_text = st.text_input("输入图像描述", key="image_search_text")
        search_top_k = st.number_input("返回图像数量", min_value=1, max_value=50, value=10, key="image_search_k")
        
        if st.button("🔍 搜索图像", key="image_search_btn"):
            if search_text:
                with st.spinner("正在搜索图像..."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/multimodal/text_to_image_search",
                            json={
                                "query_text": search_text,
                                "top_k": search_top_k,
                                "collection_name": st.session_state.config["milvus"]["collection_name"]
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("results"):
                                st.success(f"✅ 找到 {len(result['results'])} 个相关图像")
                                # 显示图像结果
                                cols = st.columns(3)
                                for i, img_info in enumerate(result["results"]):
                                    with cols[i % 3]:
                                        st.image(img_info["image_path"], caption=f"相似 {img_info['distance']:.4f}")
                            else:
                                st.info("ℹ️ " + result.get("message", "未找到相关图"))
                        else:
                            st.error(f"❌ 搜索失败: {response.text}")
                            
                    except Exception as e:
                        st.error(f"❌ 连接后端失败: {str(e)}")
            else:
                st.warning("⚠️ 请输入图像描述")
    else:
        st.warning("⚠️ 图像处理功能未启用，请在配置中启用多模态功能")

with st.expander("📊 性能监控与压测", expanded=False):
    st.info("实时监控系统性能指标并进行Milvus集合压力测试")
    
    # 创建选项
    perf_tab1, perf_tab2, perf_tab3 = st.tabs(["系统监控", "压力测试", "测试历史"])
    
    with perf_tab1:
        st.subheader("🖥️ 系统性能监控")
        
        col_monitor1, col_monitor2 = st.columns([1, 1])
        
        with col_monitor1:
            if st.button("🔄 刷新性能数据", key="refresh_perf_btn"):
                try:
                    response = requests.get(f"{BACKEND_URL}/performance/current")
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
                        return requests.get(f"{BACKEND_URL}/performance/current").json()
                    except:
                        return {}
                
                metrics = get_performance_data()
            
            # 可选：保留或删除现有数据的选项
            keep_existing_data = st.checkbox("保留现有监控数据", value=True, key="keep_monitoring_data")
            if not keep_existing_data:
                st.warning("⚠️ 现有监控数据将被清除")
                if st.button("清除监控数据", key=f"clear_monitoring_data_{datetime.now().timestamp()}"):
                    st.success("监控数据已清")
            
            # 导出监控报告
            if st.button("📊 导出性能报告", key="export_performance_report"):
                try:
                    response = requests.get(f"{BACKEND_URL}/performance/export_report")
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
                    "用户启动速率 (用户/秒)", 
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
                scenario_single = st.checkbox("单向量搜索", value=True, help="最常见的搜索操作")
                scenario_batch = st.checkbox("批量向量搜索", value=True, help="批量搜索操作")
                scenario_precision = st.checkbox("高精度搜", value=False, help="高精度但较慢的搜索")
                scenario_fast = st.checkbox("快速搜", value=False, help="快速但精度较低的搜索")
                
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
                    st.error("请至少选择一个测试场景")
                else:
                    # 构建测试参数
                    test_params = {
                        "users": test_users,
                        "spawn_rate": spawn_rate,
                        "run_time": run_time if run_time else "60s",  # 添加默认值
                        "host": st.session_state.config.get("milvus", {}).get("host", "localhost"),
                        "port": st.session_state.config.get("milvus", {}).get("port", 19530),
                        "collection_name": target_collection or "default_collection",
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
                                f"{BACKEND_URL}/load-test/start",
                                json=test_params,
                                timeout=DEFAULT_TIMEOUT  
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                test_id = result.get("test_id")
                                web_url = result.get("web_url")
                                
                                if result.get("status") == "success":
                                    st.success(f"✅ 压力测试已启动！测试ID: {test_id}")
                                    
                                    
                                    if web_url:
                                        st.markdown(f"""
                                        ### 🌐 Locust Web界面
                                        点击下方链接访问Locust官方监控界面，查看实时测试数据：
                                        
                                        **[🔗 打开Locust Web界面]({web_url})**
                                        
                                        或复制链接到浏览器：`{web_url}`
                                        """)
                                        
                                        
                                        if st.button("🚀 在新窗口中打开Locust界面", key="open_locust_web"):
                                            st.markdown(f'<script>window.open("{web_url}", "_blank");</script>', unsafe_allow_html=True)
                                    
                                    
                                    with st.container():
                                        st.markdown("查看测试配置")
                                        st.json(test_params)
                                else:
                                    st.error(f"启动测试失败: {result.get('message', '未知错误')}")
                            else:
                                st.error(f"请求失败: {response.status_code}")
                                
                        except Exception as e:
                            st.error(f"启动压测失败: {str(e)}")
        
        
        st.markdown("### 📊 测试状态管理")
        
        col_status1, col_status2 = st.columns(2)
        
        with col_status1:
            if st.button("🔍 刷新测试列表", key="refresh_tests"):
                with st.spinner("正在获取最新测试状态..."):
                    try:
                        response = requests.get(f"{BACKEND_URL}/load-test/list")
                        if response.status_code == 200:
                            st.session_state.tests = response.json().get("tests", [])
                            st.toast("刷新成功", icon="✅")
                        else:
                            st.error("刷新失败")
                    except Exception as e:
                        st.error(f"刷新异常: {str(e)}")
        
        with col_status2:
            if st.button("🧹 清理完成的测试", key="cleanup_tests"):
                # 待添加清理逻辑
                st.info("清理功能将在后续版本中实现")
        
        # 获取测试列表
        try:
            response = requests.get(f"{BACKEND_URL}/load-test/list")
            if response.status_code == 200:
                tests_data = response.json()
                tests = tests_data.get("tests", [])
                
                if tests:
                    st.markdown(f"**当前共有 {len(tests)} 个测试**")
                    
                    for test in tests:
                        test_id = test.get("test_id", "unknown")
                        status = test.get("status", "unknown")
                        start_time = test.get("start_time", "")
                        
                        # 状态颜色
                        if status == "running":
                            status_color = "🟢"
                            status_text = "运行中"
                        elif status == "completed":
                            status_color = "✅"
                            status_text = "已完成"
                        elif status == "failed":
                            status_color = "❌"
                            status_text = "失败"
                        else:
                            status_color = "⚪"
                            status_text = status
                        
                        with st.container():
                            st.markdown(f"{status_color} 测试 {test_id} - {status_text}")
                            col_info1, col_info2 = st.columns(2)
                            
                            with col_info1:
                                st.write(f"**测试ID:** {test_id}")
                                st.write(f"**状态:** {status_text}")
                                if start_time:
                                    st.write(f"**开始时间:** {start_time[:19]}")
                            
                            with col_info2:
                                # 获取Web界面URL
                                try:
                                    url_response = requests.get(f"{BACKEND_URL}/load-test/web-url/{test_id}")
                                    if url_response.status_code == 200:
                                        web_url = url_response.json().get("web_url")
                                        if web_url:
                                            st.markdown(f"**[🔗 打开Locust界面]({web_url})**")
                                except:
                                    pass
                                
                                # 停止按钮
                                if status == "running":
                                    if st.button(f"⏹️ 停止测试", key=f"stop_{test_id}"):
                                        try:
                                            stop_response = requests.post(f"{BACKEND_URL}/load-test/stop/{test_id}")
                                            if stop_response.status_code == 200:
                                                st.success("测试已停止")
                                                st.rerun()
                                            else:
                                                st.error("停止测试失败")
                                        except Exception as e:
                                            st.error(f"停止测试失败: {str(e)}")
                else:
                    st.info("📭 当前没有运行的测试")
            else:
                st.error("无法获取测试列表")
        except Exception as e:
            st.error(f"获取测试状态失败: {str(e)}")
            st.info("无法获取当前运行的测试信息")
    
    with perf_tab3:
        st.subheader("📈 测试历史与结果")
        
        if st.button("🔄 刷新测试历史", key="refresh_test_history"):
            try:
                response = requests.get(f"{BACKEND_URL}/load-test/history")
                if response.status_code == 200:
                    tests = response.json()
                    
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
                            with st.container():
                                st.markdown(f"测试 {test['test_id']} - {test.get('status', 'unknown')}")
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
                                        st.write(f"- 失败数: {metrics.get('failures', 'N/A')}")
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
                    response = requests.get(f"{BACKEND_URL}/system/status")
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
                            # 嵌入模型状态
                            embedding = status.get("embedding_model", {})
                            embedding_status = "✅正常" if embedding.get("available") else "✅异常"
                            st.write(f"🧠 嵌入模型: {embedding_status}")
                            if embedding.get("available"):
                                st.write(f"  - 模型: {embedding.get('model_name', 'N/A')}")
                                st.write(f"  - 设备: {embedding.get('device', 'N/A')}")
                            
                            # Milvus状态
                            milvus = status.get("milvus", {})
                            milvus_status = "✅已连接" if milvus.get("connected") else "❌未连接"
                            st.write(f"🗄️ Milvus: {milvus_status}")
                            if milvus.get("connected"):
                                st.write(f"  - 地址: {milvus.get('host')}:{milvus.get('port')}")
                                st.write(f"  - 集合✅ {milvus.get('collections_count', 0)}")
                        
                        with col_core2:
                            # 聚类服务状态
                            clustering = status.get("clustering_service", {})
                            clustering_status = "✅可用" if clustering.get("available") else "✅不可"
                            st.write(f"📊 聚类服务: {clustering_status}")
                            if clustering.get("available"):
                                st.write(f"  - HDBSCAN: {'✅' if clustering.get('hdbscan_available') else '❌'}")
                                st.write(f"  - Sklearn: {'✅' if clustering.get('sklearn_available') else '❌'}")
                            
                            # 分块系统状态
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
                            # # LLM配置状态
                            # llm_config = status.get("llm_config", {})
                            # llm_config_status = "✅ 可用" if llm_config.get("available") else "❌ 不可用"
                            # st.write(f"🤖 LLM配置: {llm_config_status}")
                            # if llm_config.get("available"):
                            #     active_config = llm_config.get("active_config")
                            #     if active_config:
                            #         st.write(f"  - 激活配置: {active_config.get('id', 'N/A')}")
                            #         st.write(f"  - 提供商: {active_config.get('provider', 'N/A')}")
                            #     else:
                            #         st.write("  - 激活配置: 无")
                            #     st.write(f"  - 总配置数: {llm_config.get('total_configs', 0)}")
                            
                            # 搜索优化状
                            search_opt = status.get("search_optimization", {})
                            search_opt_status = "✅ 可用" if search_opt.get("available") else "❌ 不可用"
                            st.write(f"🔍 搜索优化: {search_opt_status}")
                        
                        with col_ext2:
                            # 压测功能状
                            load_test = status.get("load_testing", {})
                            load_test_status = "✅ 可用" if load_test.get("available") else "❌ 不可用"
                            st.write(f"🧪 压力测试: {load_test_status}")
                            if load_test.get("available"):
                                st.write(f"  - 活跃测试: {load_test.get('active_tests_count', 0)}")
                                st.write(f"  - 运行中: {load_test.get('running_tests', 0)}")
                            
                            # 性能监控状态
                            perf_monitor = status.get("performance_monitor", False)
                            perf_status = "✅ 运行中" if perf_monitor else "❌ 未运行"
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
                            st.write(f"- 多模态: {'启用' if config_info.get('multimodal_enabled') else '禁用'}")
                            st.write(f"- 分块策略: {config_info.get('chunking_strategy')}")
                    else:
                        st.error(f"获取系统状态失✅ {response.status_code}")
            except Exception as e:
                st.error(f"连接失败: {str(e)}")
    
    with col_diag2:
        if st.button("🧪 运行集成测试", key="integration_test_btn"):
            try:
                with st.spinner("正在运行系统集成测试..."):
                    response = requests.post(f"{BACKEND_URL}/system/integration_test")
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
                            st.error(f"❌ 集成测试失败 ({summary.get('passed', 0)}/{summary.get('total', 0)})")
                        
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
                                st.error(f"{test_name}: ❌ 失败")
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
                    response = requests.post(f"{BACKEND_URL}/system/reload_config")
                    if response.status_code == 200:
                        st.success("系统配置已重新加")
                        st.info("所有模块已重新初始化，新配置已生效")
                    else:
                        st.error("❌ 重新加载配置失败")
            except Exception as e:
                st.error(f"重新加载配置失败: {str(e)}")
    
    with col_maint2:
        if st.button("📊 导出系统报告", key="export_report_btn"):
            try:
                # 获取系统状
                status_response = requests.get(f"{BACKEND_URL}/system/status")
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
