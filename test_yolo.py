import streamlit as st
import subprocess
import os
import shutil

# Başlık
st.title("YOLOv7 Çatlak Tespiti")

# Kullanıcıdan resim yüklemesini iste
uploaded_image = st.file_uploader("Bir resim yükleyin", type=["jpg", "png", "jpeg"])

# Geçici klasörü oluştur
temp_dir = "tempDir"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Eğer bir resim yüklendiyse
if uploaded_image is not None:
    # Yüklenen resmi geçici bir dosya olarak kaydet
    img_path = os.path.join(temp_dir, uploaded_image.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Kullanıcıya yüklenen resmi göster
    st.image(img_path, caption="Yüklenen Resim", use_column_width=True)

    # Detect.py scriptini çalıştır
    detect_command = f"python yolov7/detect.py --weights weights/best.pt --source {img_path} --device cpu"

    # subprocess ile detect.py'yi çalıştır
    try:
        result = subprocess.run(detect_command, shell=True, check=True, capture_output=True)
        st.success("Model başarıyla çalıştırıldı!")
    except subprocess.CalledProcessError as e:
        st.error(f"Bir hata oluştu: {e}")

    # Çıktı resmini doğru şekilde bulma
    # runs/detect/exp dizininde, çalıştırıldıkça exp1, exp2, exp3 vs. şeklinde isim değişir.
    # Bu yüzden önce en son işlenen sonucu bulmamız gerekiyor.
    
    # Çıktı dosyasının bulunduğu yolu dinamik olarak belirleme
    output_dir = "runs/detect"
    latest_exp_dir = sorted(os.listdir(output_dir), reverse=True)[0]  # En son çalıştırılan exp klasörünü bul
    result_img_path = os.path.join(output_dir, latest_exp_dir, uploaded_image.name)  # Yüklenen resmin ismiyle kaydedilen çıktı

    # Çıktı resmini göster
    if os.path.exists(result_img_path):
        st.image(result_img_path, caption="Model Sonucu", use_column_width=True)
    else:
        st.warning("Sonuç görüntüsü bulunamadı.")
