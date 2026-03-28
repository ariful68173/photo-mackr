import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io

def process_image(input_image):
    # ১. ব্যাকগ্রাউন্ড রিমুভ করা
    img_bytes = input_image.getvalue()
    result_bytes = remove(img_bytes)
    img = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
    
    # ২. প্রফেশনাল সাদা ব্যাকগ্রাউন্ড যোগ করা
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    combined = Image.alpha_composite(white_bg, img).convert("RGB")
    
    # ৩. OpenCV ফরম্যাটে কনভার্ট করা (স্মুথিং ও কালার কারেকশনের জন্য)
    img_cv = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)
    
    # ৪. স্কিন স্মুথিং (Bilateral Filter - যা ডিটেইল ঠিক রেখে স্কিন স্মুথ করে)
    smooth = cv2.bilateralFilter(img_cv, d=9, sigmaColor=75, sigmaSpace=75)
    
    # ৫. অটো কালার কারেকশন (CLAHE on LAB color space)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # ৬. রিসাইজ করা (45mm x 55mm @ 300 DPI ≈ 531x650 pixels)
    # আপনার রিকোয়েস্ট অনুযায়ী ৫৫০x৪৫০ পিক্সেল হিসেবে সেট করছি
    final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    pil_final = Image.fromarray(final_img_rgb)
    passport_photo = pil_final.resize((531, 650), Image.Resampling.LANCZOS)
    
    return passport_photo

# Streamlit UI
st.set_page_config(page_title="Professional Passport Photo Maker", layout="centered")
st.title("📸 AI Passport Photo Maker")
st.write("আপনার ছবি আপলোড করুন, এটি অটোমেটিক স্মুথ, কালার কারেকশন এবং পাসপোর্ট সাইজ হয়ে যাবে।")

uploaded_file = st.file_uploader("একটি ছবি সিলেক্ট করুন...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Original Image", width=300)
    
    if st.button("Generate Professional Photo"):
        with st.spinner("Processing... দয়া করে অপেক্ষা করুন।"):
            processed_img = process_image(uploaded_file)
            
            st.success("সম্পন্ন হয়েছে!")
            st.image(processed_img, caption="Final Passport Size Photo (45mm x 55mm)")
            
            # ডাউনলোড বাটন
            buf = io.BytesIO()
            processed_img.save(buf, format="JPEG", quality=95)
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Image",
                data=byte_im,
                file_name="passport_photo.jpg",
                mime="image/jpeg"
            )