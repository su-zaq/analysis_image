import os
import cv2
import numpy as np

def get_all_image_paths(folder):
    """
    指定フォルダ以下の全てのPNG画像ファイルのパスを再帰的に取得
    """
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def color_mask(img, color, tol=10):
    return np.all(np.abs(img - color) <= tol, axis=-1)

def compare_masks(membrane_dir, membrane_plus_dir, output_dir):
    """
    membraneフォルダとmembrane+フォルダの同名マスク画像を比較し、
    どちらか一方だけマスクされている領域はその色（シアンまたはマゼンタ）、
    両方マスクされている領域は白、どちらもマスクされていない領域は黒で出力
    """
    # membrane側の全画像パスを取得
    membrane_images = get_all_image_paths(membrane_dir)
    for mem_path in membrane_images:
        # membrane_dir以下の相対パスを取得
        rel_path = os.path.relpath(mem_path, membrane_dir)
        memplus_path = os.path.join(membrane_plus_dir, rel_path)
        if not os.path.exists(memplus_path):
            print(f"対応するファイルが見つかりません: {memplus_path}")
            continue

        # 画像読み込み
        img_mem = cv2.imread(mem_path, cv2.IMREAD_COLOR)
        img_memplus = cv2.imread(memplus_path, cv2.IMREAD_COLOR)
        if img_mem is None or img_memplus is None:
            print(f"画像の読み込みに失敗: {mem_path} または {memplus_path}")
            continue

        # BGR→RGBに変換
        img_mem = cv2.cvtColor(img_mem, cv2.COLOR_BGR2RGB)
        img_memplus = cv2.cvtColor(img_memplus, cv2.COLOR_BGR2RGB)

        # 画像サイズが異なる場合はスキップ
        if img_mem.shape != img_memplus.shape:
            print(f"画像サイズ不一致: {mem_path} と {memplus_path}")
            continue

        # シアン(RGB: 0,255,255)、マゼンタ(RGB: 255,0,255)
        cyan_rgb = np.array([0,255,255])
        magenta_rgb = np.array([255,0,255])

        mask_mem = color_mask(img_mem, cyan_rgb)
        mask_memplus = color_mask(img_memplus, magenta_rgb)

        # 出力画像作成
        diff_img = np.zeros_like(img_mem)

        # 両方マスクされている部分 → 白
        both_mask = mask_mem & mask_memplus
        diff_img[both_mask] = [255,255,255]

        # membrane側だけマスク → シアン
        only_mem = mask_mem & ~mask_memplus
        diff_img[only_mem] = [0,255,255]

        # membrane+側だけマスク → マゼンタ
        only_memplus = ~mask_mem & mask_memplus
        diff_img[only_memplus] = [255,0,255]

        # 保存時はRGB→BGRに変換
        diff_img_bgr = cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR)

        # 保存先パス作成
        save_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, diff_img_bgr)

        print(f"差分画像を保存: {save_path}")

if __name__ == "__main__":
    membrane_dir = "./colored_data/compare_data/membrane"
    membrane_plus_dir = "./colored_data/compare_data/membrane+"
    output_dir = "./colored_data/diff"
    compare_masks(membrane_dir, membrane_plus_dir, output_dir)
