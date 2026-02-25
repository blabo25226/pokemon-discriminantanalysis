# 虫タイプとドラゴンタイプの判別分析-ロジスティック回帰分析

# データの読み取り
from poke_preparation import load_pokemon_data
df, df_target, dfd, df_analysis, num_cols  = load_pokemon_data()

import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


# ロジスティック回帰分析
y = df_target['is_dragon']
X = df_target[num_cols]
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
X_scaled = sm.add_constant(X_scaled)
logit_model = sm.Logit(y, X_scaled)
result_lasso = logit_model.fit_regularized(method='l1', alpha=1)#ラッソ回帰分析で変数選択をする
print("="*60)
print("=== ドラゴンタイプ(1) vs むしタイプ(0) の判別分析 (Lasso) ===")
print("="*60)
print(result_lasso.summary())

# 正解率と混同行列
y_pred_prob = result_lasso.predict(X_scaled)
y_pred = (y_pred_prob >= 0.5).astype(int)
acc = accuracy_score(y, y_pred)
print(f"正解率 (Accuracy): {acc:.3f} ({acc * 100:.1f}%)")
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['むし(予測)', 'ドラゴン(予測)'],
            yticklabels=['むし(正解)', 'ドラゴン(正解)'])
plt.title('混同行列 (Confusion Matrix)')
plt.ylabel('実際のタイプ (Actual)')
plt.xlabel('予測されたタイプ (Predicted)')
plt.show()

# 主成分分析を用いた散布図
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled.drop(columns=['const'])) # statsmodels用のconstを除外
# 図示用にsklearnのロジスティック回帰を使用
logit_2d = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
logit_2d.fit(X_pca, y)
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
Z_logit = logit_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z_logit, alpha=0.3, cmap=plt.cm.coolwarm)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, style=y, 
                palette={0: 'blue', 1: 'red'}, markers={0: 'o', 1: '^'}, 
                s=80, edgecolor='black')
plt.title('ロジスティック回帰 の決定境界 [PCA空間]')
plt.xlabel('第1主成分 (PC1)')
plt.ylabel('第2主成分 (PC2)')
plt.legend(labels=['むし', 'ドラゴン'], loc='upper right')
plt.show()

#不正解のポケモンを表示
df_target['予測確率'] = y_pred_prob
df_target['予測クラス'] = y_pred
df_target['予測正解'] = (df_target['is_dragon'] == df_target['予測クラス'])
misclassified = df_target[df_target['予測正解'] == False].copy()
misclassified['実際のタイプ'] = misclassified['is_dragon'].map({1: 'ドラゴン', 0: 'むし'})
misclassified['予測されたタイプ'] = misclassified['予測クラス'].map({1: 'ドラゴン', 0: 'むし'})
cols_to_show = ['名前', 'タイプ1', 'タイプ2', '実際のタイプ', '予測されたタイプ', '予測確率']
print("\n" + "="*70)
print(f"=== 判別を間違えたポケモン（全 {len(misclassified)} 匹） ===")
print("="*70)
if len(misclassified) > 0:
    # 確率が高い（ドラゴンと予測されやすかった）順に並び替え
    misclassified = misclassified.sort_values('予測確率', ascending=False)
    print(misclassified[cols_to_show].to_string(index=False))
else:
    print("全問正解でした！")