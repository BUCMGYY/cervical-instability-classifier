import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

# ResNeXt50 图像特征提取模块（输出128维）
class CNN2DModel_xt50(nn.Module):
    def __init__(self, pretrained=False):  # 推理时pretrained建议为False
        super(CNN2DModel_xt50, self).__init__()
        self.backbone = models.resnext50_32x4d(pretrained=pretrained)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 128)  # 提取图像特征

    def forward(self, x):
        return self.backbone(x)

# 多模态模型（图像 + 结构化数值特征）
class MultiModalNN_xt50(nn.Module):
    def __init__(self):
        super(MultiModalNN_xt50, self).__init__()
        self.cnn_model = CNN2DModel_xt50(pretrained=False)  # 部署时不加载预训练权重
        self.fc_numeric = nn.Linear(9, 64)                 # 数值型特征处理层
        self.fc_combined = nn.Linear(128 + 64, 2)          # 合并后输出两类

    def forward(self, mri_data, numeric_data):
        img_features = self.cnn_model(mri_data)                    # [B, 128]
        num_features = F.relu(self.fc_numeric(numeric_data))       # [B, 64]
        combined = torch.cat((img_features, num_features), dim=1)  # [B, 192]
        return self.fc_combined(combined)

# Streamlit缓存加载模型
@st.cache_resource
def load_model():
    model = MultiModalNN_xt50()
    checkpoint = torch.load("best_xt50.pt", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

model = load_model()

# 分类标签
class_names = ['有退行性颈椎失稳', '没有退行性颈椎失稳']  # 0有 1没有

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet101 默认输入224x224
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("退行性颈椎失稳多模态智能检测辅助系统")
st.write("请上传三个不同体位的X线图像，并输入9项结构化特征，模型将判断是否存在退行性颈椎失稳。")

col1, col2 = st.columns(2)
with col1:
    st.subheader("X线图像上传")
    image1 = st.file_uploader("上传侧位 X线图像", type=["jpg", "jpeg", "png"], key="img1")
    image2 = st.file_uploader("上传过屈位 X线图像", type=["jpg", "jpeg", "png"], key="img2")
    image3 = st.file_uploader("上传过伸位 X线图像", type=["jpg", "jpeg", "png"], key="img3")
    images = []
    view_names = ["侧位", "过屈位", "过伸位"]
    for idx, img_file in enumerate([image1, image2, image3]):
        if img_file:
            img = Image.open(img_file)
            st.image(img, caption=f"{view_names[idx]}上传图像", use_container_width=True)
            images.append(transform(img))
        else:
            images.append(None)
with col2:
    st.subheader("结构化特征输入")
    sex = st.radio("性别 (Sex)", options=[0, 1], format_func=lambda x: "女" if x == 0 else "男")
    age = st.number_input("年龄 (Age)", min_value=0, max_value=120, value=30)
    dizziness = st.radio("头晕 (Dizziness)", options=[0, 1], format_func=lambda x: "无" if x == 0 else "有")
    headache = st.radio("头痛 (Headache)", options=[0, 1], format_func=lambda x: "无" if x == 0 else "有")
    tenderness = st.radio("压痛 (Tenderness)", options=[0, 1], format_func=lambda x: "无" if x == 0 else "有")
    tinnitus = st.radio("耳鸣 (Tinnitus)", options=[0, 1], format_func=lambda x: "无" if x == 0 else "有")
    limited_flexion_extension = st.radio("屈伸运动受限", options=[0, 1], format_func=lambda x: "无" if x == 0 else "有")
    nausea_vomiting = st.radio("恶心呕吐", options=[0, 1], format_func=lambda x: "无" if x == 0 else "有")
    chest_tightness_palpitations = st.radio("胸闷心悸", options=[0, 1], format_func=lambda x: "无" if x == 0 else "有")
    numeric_features = [
        sex, age, dizziness, headache, tenderness,
        tinnitus, limited_flexion_extension, nausea_vomiting, chest_tightness_palpitations
    ]

st.markdown("---")

if None not in images and st.button("开始预测"):
    with st.spinner("模型推理中，请稍候..."):
        with torch.no_grad():
            features = []
            for img in images:
                img = img.unsqueeze(0)  # [1, 3, 224, 224]
                feat = model.cnn_model(img)
                features.append(feat)
            img_features = torch.stack(features).mean(dim=0)  # [1, 128]

            numeric_tensor = torch.tensor([numeric_features], dtype=torch.float32)  # [1, 9]
            num_features = F.relu(model.fc_numeric(numeric_tensor))
            combined = torch.cat((img_features, num_features), dim=1)
            output = model.fc_combined(combined)  # [1, 2]
            prob = F.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()

        st.success(f"检测结果：{class_names[pred]}")
        st.progress(float(prob[0][pred]))
        st.write(f"预测概率：{prob[0][pred].item():.4f}")

        # ---- Grad-CAM 可视化 ----
        st.subheader("Grad-CAM 可视化结果")
        view_names = ['侧位', '过屈位', '过伸位']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            # 对每一张图单独做 Grad-CAM
            input_tensor = images[i].unsqueeze(0)  # [1, 3, 224, 224]
            # Grad-CAM目标层
            target_layers = [model.cnn_model.backbone.layer4[-1]]
            cam = GradCAM(model=model.cnn_model.backbone, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor)[0, :]  # [224, 224]
            # 反归一化原图
            img_np = np.array(Image.open([image1, image2, image3][i]).resize((224, 224))).astype(np.float32) / 255.0
            if img_np.ndim == 2:  # 灰度图转RGB
                img_np = np.stack([img_np]*3, axis=-1)
            elif img_np.shape[2] == 4:
                img_np = img_np[:, :, :3]
            cam_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            axes[i].imshow(cam_img)
            axes[i].set_title(view_names[i])
            axes[i].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
else:
    st.info("请上传三个位体位的X线图像，并输入全部结构化特征。")

st.caption("注：本工具仅供科研参考，不作为临床诊断依据。")