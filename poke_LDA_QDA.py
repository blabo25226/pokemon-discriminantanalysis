# 虫タイプとドラゴンタイプの判別分析-線形判別と二次判別

# データの読み取り
from poke_preparation import load_pokemon_data
df, df_target, dfd, df_analysis, num_cols  = load_pokemon_data()

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns


# 多重共線性を回避するためにVIFを用いて変数選択をする
# VIFを計算して、閾値(threshold)以上の変数を機械的に除外する関数
def remove_multicollinearity_vif(X, threshold=10.0):
    # 元データを壊さないようにコピー
    X_vif = X.copy()
    # VIFの計算には定数項(const)が必要なため追加
    if 'const' not in X_vif.columns:
        X_vif = sm.add_constant(X_vif)
    dropped_vars = []
    while True:
        # 現在の変数群でVIFを計算
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_vif.columns
        vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
        # 定数項(const)は除外対象から外す
        vif_data = vif_data[vif_data['feature'] != 'const']
        # 最もVIFが高い変数を見つける
        max_vif = vif_data['VIF'].max()
        max_col = vif_data.loc[vif_data['VIF'].idxmax(), 'feature']
        # 最大VIFが閾値(10)を超えていたら、その変数を除外
        if max_vif > threshold:
            print(f"除外: {max_col} (VIF = {max_vif:.2f})")
            X_vif = X_vif.drop(columns=[max_col])
            dropped_vars.append(max_col)
        else:
            # すべての変数のVIFが閾値を下回ったらループ終了
            break
    # 計算用に足した定数項を最終的なデータから外して返す
    X_final = X_vif.drop(columns=['const']) if 'const' in X_vif.columns else X_vif
    print("\n=== 最終的に残った変数 ===")
    print(X_final.columns.tolist())
    return X_final


#線形判別分析
y = df_target['is_dragon']
X = df_target[num_cols]
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
X_features = remove_multicollinearity_vif(X_scaled, threshold=10.0)
lda = LinearDiscriminantAnalysis()
lda.fit(X_features, y)
y_pred_lda = lda.predict(X_features)
#正解率
acc_lda = accuracy_score(y, y_pred_lda)
print(f"LDA (線形判別分析) 正解率: {acc_lda:.3f} ({acc_lda * 100:.1f}%)")

#二次判別分析
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_features, y)
y_pred_qda = qda.predict(X_features)
#正解率
acc_qda = accuracy_score(y, y_pred_qda)
print(f"QDA (二次判別分析) 正解率: {acc_qda:.3f} ({acc_qda * 100:.1f}%)")

#混同行列
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm_lda = confusion_matrix(y, y_pred_lda)
sns.heatmap(cm_lda, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['むし(予測)', 'ドラゴン(予測)'],
            yticklabels=['むし(正解)', 'ドラゴン(正解)'])
axes[0].set_title('LDA 混同行列')
axes[0].set_ylabel('実際のタイプ')
axes[0].set_xlabel('予測されたタイプ')
cm_qda = confusion_matrix(y, y_pred_qda)
sns.heatmap(cm_qda, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['むし(予測)', 'ドラゴン(予測)'],
            yticklabels=['むし(正解)', 'ドラゴン(正解)'])
axes[1].set_title('QDA 混同行列')
axes[1].set_ylabel('実際のタイプ')
axes[1].set_xlabel('予測されたタイプ')
plt.tight_layout()
plt.show()

#主成分分析を用いた散布図
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_features)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=df_target.index)
df_pca['is_dragon'] = y
df_pca['LDA_correct'] = (y == y_pred_lda)
df_pca['QDA_correct'] = (y == y_pred_qda)
X_scaled_np = X_scaled.values  # numpy配列として取得
lda_2d = LinearDiscriminantAnalysis()
lda_2d.fit(X_pca, y)
qda_2d = QuadraticDiscriminantAnalysis()
qda_2d.fit(X_pca, y)
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),  # 0.05刻みで細かく
                     np.arange(y_min, y_max, 0.05))
mesh_points = np.c_[xx.ravel(), yy.ravel()]
Z_lda = lda_2d.predict(mesh_points).reshape(xx.shape)
Z_qda = qda_2d.predict(mesh_points).reshape(xx.shape)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
def plot_decision_boundary(ax, Z, title):
    # 背景の塗りつぶし (むし=0: 青系, ドラゴン=1: 赤系)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    # 実際のデータポイントを散布図として重ねる
    # (y=0:むし, y=1:ドラゴン)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, style=y, 
                    palette={0: 'blue', 1: 'red'}, markers={0: 'o', 1: '^'}, 
                    s=80, edgecolor='black', ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('第1主成分 (PC1)')
    ax.set_ylabel('第2主成分 (PC2)')
    # 凡例の調整
    handles, labels = ax.get_legend_handles_labels()
    # seabornの仕様でhueのタイトルが入るのを防ぐため、インデックス1以降を取得
    ax.legend(handles=handles, labels=['むし', 'ドラゴン'], loc='upper right')
plot_decision_boundary(axes[0], Z_lda, 'LDA (線形判別分析) の決定境界')
plot_decision_boundary(axes[1], Z_qda, 'QDA (二次判別分析) の決定境界')
plt.show()

# 不正解のポケモンを表示
def print_misclassified(y_pred, model_name):
    misclassified = df_target[y != y_pred].copy()
    misclassified['実際のタイプ'] = misclassified['is_dragon'].map({1: 'ドラゴン', 0: 'むし'})
    misclassified['予測されたタイプ'] = pd.Series(y_pred, index=df_target.index).map({1: 'ドラゴン', 0: 'むし'})
    cols_to_show = ['名前', 'タイプ1', 'タイプ2', '実際のタイプ', '予測されたタイプ']
    print("\n" + "="*70)
    print(f"=== {model_name} で判別を間違えたポケモン（全 {len(misclassified)} 匹） ===")
    print("="*70)
    if len(misclassified) > 0:
        print(misclassified[cols_to_show].to_string(index=False))
    else:
        print("全問正解でした！")
print_misclassified(y_pred_lda, "LDA (線形判別分析)")
print_misclassified(y_pred_qda, "QDA (二次判別分析)")