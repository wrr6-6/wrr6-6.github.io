import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
import os
import time
from utils import set_background   # 需确保 utils.py 存在并包含 set_background 函数
from streamlit_webrtc import webrtc_streamer
import av
# 导入 Coze 聊天机器人库
from cozepy import Coze, TokenAuth, Message, ChatEventType
from cozepy import COZE_CN_BASE_URL

# -------------------- 页面配置 --------------------
st.set_page_config(
    page_title="花卉识别系统",
    page_icon="🌻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- 自定义CSS美化 --------------------
st.markdown("""
<style>
    /* 隐藏默认工具栏中的菜单和footer，但保留侧边栏展开按钮 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stStatusWidget"] {visibility: hidden;}

    /* 全局字体与背景 */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Roboto, sans-serif;
    }

    /* 标题样式 */
    .main-title {
        text-align: center;
        color: #2c3e50;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* 卡片效果 */
    .css-1r6slb0, .css-12w0qpk {
        background: rgba(255,255,255,0.9);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3);
    }

    /* 按钮样式 */
    .stButton button {
        background: linear-gradient(45deg, #ff9a9e 0%, #fad0c4 99%, #fad0c4 100%);
        color: #2c3e50;
        border: none;
        border-radius: 50px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        background: linear-gradient(45deg, #fad0c4 0%, #ff9a9e 99%);
        color: #1a2634;
    }

    /* 滑块样式 */
    .stSlider .thumb {
        background-color: #ff9a9e !important;
    }

    /* 成功/信息提示框 */
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid #ff9a9e;
        background: rgba(255,255,255,0.7);
    }

    /* 图片容器 */
    .image-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin: 1rem 0;
    }

    /* 聊天气泡样式 */
    .stChatMessage {
        border-radius: 20px !important;
        padding: 10px 15px !important;
        margin: 5px 0 !important;
    }
    [data-testid="stChatMessageContent"] p {
        margin: 0 !important;
    }

    /* 页脚 */
    .footer {
        text-align: center;
        color: #7f8c8d;
        padding: 2rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- 设置背景图片（需 utils.py 文件） --------------------
set_background("./imgs/background.png")

# -------------------- 加载模型 --------------------
LICENSE_FLOWER_MODEL = './model/flowers_segmentation_model.pt'
flower_model = YOLO(LICENSE_FLOWER_MODEL)

# -------------------- 初始化 Coze 聊天机器人 --------------------
COZE_API_TOKEN = "cztei_hdaHBl2ikOm5p1mj47unrSKBM7EInSK48Hry6u3G3FW4ZpPQcGBUTFP1D5g1EzgKy"
BOT_ID = "7618127561478193158"
USER_ID = "123456789"  # 固定用户标识，可根据需要修改
coze = Coze(auth=TokenAuth(token=COZE_API_TOKEN), base_url=COZE_CN_BASE_URL)

# -------------------- 初始化session状态 --------------------
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.30
if "state" not in st.session_state:
    st.session_state.state = "Uploader"
if "live_key" not in st.session_state:
    st.session_state.live_key = "live"
if "messages" not in st.session_state:   # 聊天历史记录
    st.session_state.messages = [
        {"role": "assistant", "content": "你好呀！我是花精灵，有什么关于花花草草的问题都可以问我哦~"}
    ]

# -------------------- 侧边栏设置 --------------------
with st.sidebar:
    st.markdown("## ⚙️ 识别设置")
    # 置信度阈值滑块
    threshold = st.slider(
        "置信度阈值",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.threshold,
        step=0.05,
        help="高于此阈值的检测结果才会被视为有效花卉。实时检测模式下更改阈值会导致视频流重启。"
    )
    if threshold != st.session_state.threshold:
        st.session_state.threshold = threshold
        # 更新live_key以重启webrtc
        st.session_state.live_key = f"live_{threshold}_{np.random.rand()}"

    st.markdown("---")
    st.markdown("### 📊 模型信息")
    st.info(f"**模型**: 花卉识别模型\n\n**默认阈值**: 0.30\n\n**支持类型**: 菊花、玫瑰、向日葵等")

    st.markdown("---")
    st.markdown("### 📌 使用提示")
    st.caption("1. 上传图片或拍照进行单张识别\n2. 实时检测模式使用摄像头\n3. 上传视频进行批量帧识别\n4. 调整阈值控制识别灵敏度\n5. 右侧导航栏可切换到『聊天助手』咨询花卉知识")

# -------------------- 视频处理器类（支持动态阈值） --------------------
class VideoProcessor:
    def __init__(self, threshold):
        self.threshold = threshold

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        pred = flower_model.predict(img, verbose=False)[0]

        # 检测结果处理
        if pred.boxes is None or len(pred.boxes) == 0:
            # 无检测目标
            img_wth_box = img.copy()
            h, w = img_wth_box.shape[:2]
            overlay = img_wth_box.copy()
            cv2.rectangle(overlay, (0, h//2-60), (w, h//2+60), (0, 0, 0), -1)
            img_wth_box = cv2.addWeighted(overlay, 0.6, img_wth_box, 0.4, 0)
            cv2.putText(img_wth_box, "No flowers detected", (w//2-200, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            # 绘制检测框
            img_wth_box = pred.plot()
            max_conf = max([box.conf[0].item() for box in pred.boxes])
            if max_conf < self.threshold:
                cv2.putText(img_wth_box, f"Low confidence: {max_conf:.2f} < {self.threshold}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img_wth_box, format="bgr24")

# -------------------- 图片检测函数（使用session阈值） --------------------
def model_prediction(img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pred = flower_model.predict(img_bgr, verbose=False)[0]

    # 获取当前阈值
    thr = st.session_state.threshold

    if pred.boxes is None or len(pred.boxes) == 0:
        # 无检测结果
        img_wth_box = img_bgr.copy()
        h, w = img_wth_box.shape[:2]
        overlay = img_wth_box.copy()
        cv2.rectangle(overlay, (0, h//2 - 60), (w, h//2 + 60), (0, 0, 0), -1)
        img_wth_box = cv2.addWeighted(overlay, 0.6, img_wth_box, 0.4, 0)
        cv2.putText(img_wth_box, "错误: 未检测到花卉", (w//2 - 300, h//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.putText(img_wth_box, "Error: No flowers detected", (w//2 - 350, h//2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3, cv2.LINE_AA)
        flower_count = 0
        max_conf = 0.0
    else:
        # 计算最高置信度和花卉数量
        conf_list = [box.conf[0].item() for box in pred.boxes]
        max_conf = max(conf_list)
        flower_count = len(conf_list)

        if max_conf < thr:
            # 置信度过低
            img_wth_box = pred.plot()
            cv2.putText(img_wth_box, f"警告: 检测置信度过低 ({max_conf:.2f} < {thr})",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3, cv2.LINE_AA)
        else:
            img_wth_box = pred.plot()

    img_wth_box = cv2.cvtColor(img_wth_box, cv2.COLOR_BGR2RGB)
    return img_wth_box, flower_count, max_conf

# -------------------- 视频处理函数 --------------------
def process_video(video_path, threshold, output_path):
    """处理视频，逐帧检测并输出新视频"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("无法打开视频文件")
        return None

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 定义视频编码器并创建VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 创建进度条和状态显示
    progress_bar = st.progress(0, text="准备处理...")
    status_text = st.empty()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 模型预测
        pred = flower_model.predict(frame, verbose=False)[0]

        # 绘制检测框（如果无检测框，pred.plot()会返回原图）
        if pred.boxes is not None and len(pred.boxes) > 0:
            frame_with_box = pred.plot()
        else:
            frame_with_box = frame.copy()  # 无检测时保持原图

        out.write(frame_with_box)

        # 更新进度
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress, text=f"已处理 {frame_count}/{total_frames} 帧")
        status_text.text(f"当前帧率: {fps} fps | 预计剩余时间: {(total_frames - frame_count) / fps:.1f} 秒")

    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    st.success(f"处理完成！共处理 {frame_count} 帧，输出视频已保存。")
    return output_path

# -------------------- 状态切换函数 --------------------
def change_state_uploader():
    st.session_state.state = "Uploader"

def change_state_camera():
    st.session_state.state = "Camera"

def change_state_live():
    st.session_state.state = "Live"

def change_state_video():
    st.session_state.state = "Video"

def change_state_chat():
    st.session_state.state = "Chat"

# -------------------- 主界面 --------------------
st.markdown('<h1 class="main-title">🌻 花卉识别系统</h1>', unsafe_allow_html=True)

# 模式切换按钮（增加聊天助手）
col1, col2, col3, col4, col5, col6 = st.columns([0.1, 1, 1, 1, 1, 1])
with col1:
    st.write("")  # 占位
with col2:
    st.button("📁 上传图片", on_click=change_state_uploader, use_container_width=True)
with col3:
    st.button("📸 拍照", on_click=change_state_camera, use_container_width=True)
with col4:
    st.button("🎥 实时检测", on_click=change_state_live, use_container_width=True)
with col5:
    st.button("🎬 上传视频", on_click=change_state_video, use_container_width=True)
with col6:
    st.button("💬 聊天助手", on_click=change_state_chat, use_container_width=True)

st.markdown("---")

# 根据状态显示不同内容
if st.session_state.state == "Live":
    st.markdown("### 🎬 实时摄像头检测")
    st.caption(f"当前阈值: {st.session_state.threshold:.2f} (更改阈值将重启视频流)")

    webrtc_ctx = webrtc_streamer(
        key=st.session_state.get("live_key", "live"),
        video_processor_factory=lambda: VideoProcessor(st.session_state.get("threshold", 0.30)),
        async_processing=True,
        media_stream_constraints={"video": True, "audio": False}
    )

elif st.session_state.state == "Video":
    st.markdown("### 🎞️ 上传视频进行检测")
    st.caption(f"当前阈值: {st.session_state.threshold:.2f}")

    video_file = st.file_uploader("选择MP4视频文件", type=["mp4"], key="video_uploader")

    if video_file is not None:
        st.video(video_file)

        if st.button("▶️ 开始检测", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
                tmp_input.write(video_file.read())
                input_path = tmp_input.name

            output_path = tempfile.NamedTemporaryFile(delete=False, suffix="_detected.mp4").name

            with st.spinner("视频处理中，请稍候..."):
                result_path = process_video(input_path, st.session_state.threshold, output_path)

            os.unlink(input_path)

            if result_path and os.path.exists(result_path):
                with open(result_path, "rb") as f:
                    video_bytes = f.read()

                st.markdown("### ✅ 检测结果")
                st.video(video_bytes)

                st.download_button(
                    label="⬇️ 下载检测结果",
                    data=video_bytes,
                    file_name="detected_flowers.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
            else:
                st.error("视频处理失败")

elif st.session_state.state == "Chat":
    # ---------- 聊天助手界面 ----------
    st.markdown("### 💬 花精灵 · 智能问答")
    st.caption("你可以问我关于花卉养护、品种识别、花语等问题。")

    # 显示聊天历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 聊天输入框
    if prompt := st.chat_input("输入你的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用 Coze API 获取回复
        with st.chat_message("assistant"):
            with st.spinner("花精灵正在思考..."):
                try:
                    full_reply = ""
                    # 流式调用，收集完整回复
                    for event in coze.chat.stream(
                        bot_id=BOT_ID,
                        user_id=USER_ID,
                        additional_messages=[
                            Message.build_user_question_text(prompt)
                        ]
                    ):
                        if event.event == ChatEventType.CONVERSATION_MESSAGE_DELTA:
                            full_reply += event.message.content
                    st.markdown(full_reply)
                    st.session_state.messages.append({"role": "assistant", "content": full_reply})
                except Exception as e:
                    error_msg = f"调用出错: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

else:
    # 上传图片或拍照模式
    if st.session_state.state == "Uploader":
        img_file = st.file_uploader("上传花卉图片", type=["png", "jpg", "jpeg"], key="uploader")
    else:  # Camera
        img_file = st.camera_input("拍摄照片", key="camera")

    if img_file is not None:
        image = np.array(Image.open(img_file))
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🔍 应用检测", use_container_width=True):
            with st.spinner("正在识别中..."):
                result_img, count, max_conf = model_prediction(image)

            st.markdown("### ✅ 检测结果")
            if count == 0:
                st.error("未检测到花卉，请尝试调整阈值或更换图片")
            elif max_conf < st.session_state.threshold:
                st.warning(f"检测到 {count} 个目标，但最高置信度 {max_conf:.2f} 低于阈值 {st.session_state.threshold:.2f}，可能不是花卉")
            else:
                st.success(f"检测到 {count} 朵花卉，最高置信度: {max_conf:.2f}")

            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(result_img, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown('<div class="footer">🌻 花卉识别系统 | 基于YOLOv8 | 支持上传/拍照/实时检测/视频识别 | 💬 内置花精灵助手</div>',
            unsafe_allow_html=True)