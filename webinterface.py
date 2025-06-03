from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
from generator import generate

st.set_page_config(layout="wide")
st.title("Stable Diffusion Web Interface")

st.markdown("""
<style>
.stImage>img {
    max-width: 400px !important;
    height: auto !important;
}
</style>
""", unsafe_allow_html=True)


# Создаем вкладки для разных режимов
tab1, tab2, tab3 = st.tabs(["Text2Image", "Image2Image", "Denoising"])

with tab1:
    st.header("Text to Image Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prompt = st.text_area("Prompt", "An owl play tennis")
        negative_prompt = st.text_area("Negative Prompt", "")
        seed = st.number_input("Seed", value=42)
        
    with col2:
        num_inference_steps = st.slider("Inference Steps", 10, 100, 50)
    
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            try:
                output_image = generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    n_inference_steps=num_inference_steps,
                    seed=seed
                )
                st.image(output_image, caption="Generated Image", use_column_width=True)
                
                # Добавляем возможность скачать изображение
                buf = BytesIO()
                Image.fromarray(output_image).save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="generated_image.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Ошибка генерации: {str(e)}")

with tab2:
    st.header("Image to Image Transformation")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        try:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Original Image", use_column_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                prompt = st.text_area("Prompt (i2i)", "An owl smoking a pipe")

                negative_prompt = st.text_area("Negative Prompt (i2i)", "")
                seed = st.number_input("Seed (i2i)", value=42)
                
            with col2:
                num_inference_steps = st.slider("Inference Steps (i2i)", 10, 100, 50)
                strength = st.slider("Transformation Strength", 0.1, 1.0, 0.8)
            
            if st.button("Transform Image"):
                with st.spinner("Transforming image..."):
                    try:
                        output_image = generate(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            input_image=input_image,
                            strength=strength,
                            n_inference_steps=num_inference_steps,
                            seed=seed
                        )
                        st.image(output_image, caption="Transformed Image", use_column_width=True)
                        
                        # Download button
                        buf = BytesIO()
                        Image.fromarray(output_image).save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        st.download_button(
                            label="Download Image",
                            data=byte_im,
                            file_name="transformed_image.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.error(f"Ошибка трансформации: {str(e)}")
        except Exception as e:
            st.error(f"Ошибка загрузки изображения: {str(e)}")

with tab3:
    st.header("Image Denoising")
    
    uploaded_file = st.file_uploader("Upload a noisy image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        try:
            original_img = np.array(Image.open(uploaded_file).convert("RGB"))
            original_img = cv2.resize(original_img, (512, 512))
            
            # Добавление шума (для демонстрации, если нужно)
            if st.checkbox("Add artificial noise for demo"):
                noise_level = st.slider("Noise level", 0.1, 1.0, 0.4)
                noisy_img = add_noise(original_img, noise_level)
            else:
                noisy_img = original_img.copy()
            
            st.image(noisy_img, caption="Noisy Image", use_column_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                prompt = st.text_area("Denoising Prompt", "high-quality clean image, no noise")
                denoise_strength = st.slider("Denoising Strength", 0.01, 0.3, 0.09)
                denoise_steps = st.slider("Denoising Steps", 10, 100, 40)
                
            with col2:
                gaussian_kernel = st.slider("Gaussian Kernel Size", 1, 15, 5, step=2)
                gaussian_sigma = st.slider("Gaussian Sigma", 0.1, 3.0, 1.2)
        except Exception as e:
            st.error(f"Ошибка загрузки изображения: {str(e)}")
