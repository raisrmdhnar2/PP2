from roboflow import Roboflow

# 1. Inisialisasi API
rf = Roboflow(api_key="73NIym1UdHbYCkClpwMk")

# 2. Akses project workspace
project = rf.workspace("donut-ep62e").project("sexing-chicken")

# 3. Pilih versi dan format ekspor
# Format umum: "yolov8", "yolov11", "coco", "voc", atau "folder" (jika klasifikasi)
dataset = project.version(2).download("folder")

print(f"Dataset berhasil diunduh di: {dataset.location}")

#klo filenya zip nanti unzip dlu