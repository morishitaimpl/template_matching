import sys, cv2, os, pathlib
import numpy as np
import matplotlib.pyplot as plt

# 1. 画像パス取得
target_path = sys.argv[1]
template_path = sys.argv[2]
# out_dir = pathlib.Path(sys.argv[3])
# if not os.path.exists(out_dir): os.mkdir(out_dir)

# 2. 画像読み込み
target_gray = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
target_color = cv2.imread(target_path, cv2.IMREAD_COLOR)

# 3. エッジ処理（Canny）
edges_target = cv2.Canny(target_gray, 50, 150)
edges_template = cv2.Canny(template_gray, 50, 150)

cv2.imwrite("edges_target.png", edges_target)
cv2.imwrite("edges_template.png", edges_template)

# 4. テンプレートマッチング（エッジ画像）
result_edges = cv2.matchTemplate(edges_target, edges_template, cv2.TM_CCOEFF_NORMED)
_, max_val_edge, _, max_loc_edge = cv2.minMaxLoc(result_edges)
print(f"[Edge Match] 座標: {max_loc_edge}, スコア: {max_val_edge:.4f}")

# 5. 矩形描画（エッジマッチの結果位置）
h, w = template_gray.shape
bottom_right = (max_loc_edge[0] + w, max_loc_edge[1] + h)
result_img = target_color.copy()
cv2.rectangle(result_img, max_loc_edge, bottom_right, (0, 0, 255), 2)

# 6. 保存
cv2.imwrite("dst_matchImg.png", result_img)