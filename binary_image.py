import os
import cv2
import csv
import glob

def get_image_paths(folder_path):
    """
    指定フォルダ以下の全てのPNG画像ファイルのパスを再帰的に取得
    """
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def binarize_image(image, threshold):
    """
    グレースケール画像を二値化する
    """
    # 画像が3チャンネルの場合はグレースケールに変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # 二値化
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

def save_binarized_image(save_path, binary_image):
    """
    二値化画像を保存する。保存先ディレクトリがなければ作成
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, binary_image)

def main(input_folder, output_folder, threshold):
    """
    指定フォルダ内の全てのグレースケール画像を二値化して保存
    """
    image_paths = get_image_paths(input_folder)
    for img_path in image_paths:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"画像の読み込みに失敗: {img_path}")
            continue
        binary = binarize_image(image, threshold)
        # 入力フォルダからの相対パスを維持して保存
        rel_path = os.path.relpath(img_path, input_folder)
        save_path = os.path.join(output_folder, rel_path)
        save_binarized_image(save_path, binary)
        print(f"保存: {save_path}")

if __name__ == "__main__":
    # 画像を比較するプログラム
    config_experiments = ["membrane", "membrane+"]

    for config_experiment in config_experiments:
        csv_path = f"./csv_files/{config_experiment}.csv"
        # exp_numをキー、各行のデータ（辞書）を値として全て保存
        exp_data = {}
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                exp_num = int(row['exp_num'])
                # 必要なカラムだけ抽出して保存（全カラム保存したい場合はrowそのままでもOK）
                exp_data[exp_num] = {
                    'epoch_num': int(row['epoch_num']),
                    'threshold': int(row['threshold']),
                    'deleted_area': int(row['deleted_area']),
                    'precision': float(row['precision']),
                    'recall': float(row['recall']),
                    'fmeasure': float(row['fmeasure']),
                    'membrane_length': int(row['membrane_length']),
                    'tip_length': int(row['tip_length']),
                    'miss_length': int(row['miss_length'])
                }

        for i in range(1, 8):
            print(f"exp{i:04d}の二値化開始\n")
            exp_dir = f"./eval_data_membrane/{config_experiment}/exp{i:04d}"
            epoch_num = f"{exp_data[i]['epoch_num']:02d}"
            # exp_dirの一階層下の全フォルダを取得
            subfolders = [f for f in glob.glob(os.path.join(exp_dir, '*')) if os.path.isdir(f)]

            for subfolder in subfolders:
                epoch_dir = os.path.join(subfolder, f"epoch{epoch_num}")
                if os.path.exists(epoch_dir):
                    # epoch_dir配下の画像を処理
                    print(epoch_dir)
                    print(f"二値化閾値{exp_data[i]['threshold']}")
                    main(epoch_dir, f"./compare_data/{config_experiment}/exp{i:04d}", exp_data[i]['threshold'])
