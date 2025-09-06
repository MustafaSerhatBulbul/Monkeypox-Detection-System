import os

def check_dataset_structure(root_dir):
    print(f"Klasör: {root_dir}")
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            num_images = len([f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            print(f"  Sınıf: {subdir}, Görüntü sayısı: {num_images}")
        else:
            print(f"  Hata: {subdir} bir klasör değil, etiketler eksik olabilir!")

check_dataset_structure('d:/dataset/train')
check_dataset_structure('d:/dataset/val')
check_dataset_structure('d:/dataset/test')