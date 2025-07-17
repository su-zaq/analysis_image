"""
指定したフォルダ内の全てのPNG画像（二値画像）を読み込み、
画像中の白色（RGB:255,255,255）ピクセルのみを任意の色に一括変換し、
変換後の画像を指定した出力フォルダに保存するプログラム

・サブディレクトリも再帰的に探索
・変換後の画像は、元画像のパス構造の最後から5階層分を維持して保存
・保存時、必要に応じて出力先ディレクトリも自動作成
"""
import os
import cv2
import numpy as np

# 指定したパス以下の全てのPNG画像ファイルのパスをリストで返す関数
# path: 画像が格納されているディレクトリのパス
# return: 画像ファイルのパスリスト

def path_operate(path:str) -> list:

    image_path_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.png'):
                image_path_list.append(os.path.join(root, file))
    return image_path_list

# 画像パスリストから画像を読み込み、画像データとパスのペアをリストで返す関数
# image_path_list: 画像ファイルのパスリスト
# return: [画像データ, 画像パス] のリスト

def imread(image_path_list:list) -> list:

    image_list = []
    for image_path in image_path_list:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGRで読み込み
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGBに変換
        image_list.append([img_rgb, image_path])
        # print(image_path)  # デバッグ用
    return image_list

# メイン処理関数
# image_path: 入力画像ディレクトリ
# output_path: 出力先ディレクトリ
# new_color: 変換後の色 (R, G, B) タプル

def main(image_path:str, output_path:str, new_color:tuple) -> None:

    # 出力先ディレクトリがなければ作成
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 画像リストを取得
    image_list = imread(path_operate(image_path))

    if image_list is None:
        print(f"image_list is empty")
        return

    # 画像サイズを取得（全画像同じサイズ前提）
    height, width, _ = image_list[0][0].shape

    for image, file_path in image_list:
        if image is None:
            print(f"画像の読み込みに失敗: {file_path}")
            continue
        print(f"ファイル: {file_path}, shape: {image.shape}, dtype: {image.dtype}")
        white_mask = np.all(image == [255, 255, 255], axis=-1)
        print(f"白色ピクセル数: {np.sum(white_mask)}")
        # 白色画素をnew_colorに一括変換
        image[white_mask] = new_color
        # パスを正規化して区切る
        path_elements = os.path.normpath(file_path).split(os.sep)
        # 最後から5個だけ使う
        last_five = path_elements[-5:]
        # 保存先パスを作成
        save_path = os.path.join(output_path, *last_five)
        # ディレクトリがなければ作成
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 保存時にBGRに変換
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    return

# スクリプトとして実行された場合の処理
if __name__ == "__main__":
    image_path = './compare_data/membrane/'
    output_path = './colored_data/'
    new_color = (0, 255, 255)  # 変換後の色（例：マゼンタ）

    main(image_path, output_path, new_color)
    

