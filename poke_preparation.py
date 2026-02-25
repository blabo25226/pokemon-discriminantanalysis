# 今回取り扱うデータの加工と概要

def load_pokemon_data():
    import pandas as pd
    url = 'http://blog.game-de.com/pokedata/pokemon-data/'
    dfs = pd.read_html(url, encoding='utf-8')
    df=dfs[0].copy()
    df
    # データの加工
    import numpy as np
    # 謎の空白
    df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
    # '重さ (kg)'カラムから数字部分を抽出し、float型に変換
    df['重さ (kg)'] = df['重さ (kg)'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    # カラム名をより分かりやすい名前に変更
    df = df.rename(columns={
        '計': '合計種族値',
        '高さ (m)':'高さ',
        '重さ (kg)':'重さ',
        '捕 獲':'捕獲率',
        '性別 ♂:♀': '性別',
        '経 験 値':'経験値',
        'な つ き':'初期なつき度'
    })
    def calculate_male_ratio(value):
        val_str = str(value).strip()
        # 1. 性別不明の場合
        if val_str == '不明' or val_str == 'nan':
            return np.nan
        # 2. オスのみ、メスのみの場合
        if val_str == '♂':
            return 1.0
        if val_str == '♀':
            return 0.0
        # 3. "m:w" 表記の場合
        if ':' in val_str:
            try:
                m, f = map(float, val_str.split(':'))
                return m / (m + f)
            except:
                return np.nan
        return np.nan
    df['雄率'] = df['性別'].apply(calculate_male_ratio)
    num_cols = ['HP', '攻撃', '防御', '特攻', '特防', '素早', '合計種族値', '高さ', '重さ','捕獲率','経験値','初期なつき度']

    # この判別分析でのデータの取り出し
    dfd = df.copy()
    dfd['is_bug'] = dfd.apply(lambda x: 1 if (x['タイプ1'] == 'むし') or (x['タイプ2'] == 'むし') else 0, axis=1)
    dfd['is_dragon'] = dfd.apply(lambda x: 1 if (x['タイプ1'] == 'ドラゴン') or (x['タイプ2'] == 'ドラゴン') else 0, axis=1)

    # この分析で取り扱うデータの加工
    df_analysis = dfd.dropna(subset=num_cols + ['is_bug', 'is_dragon']).copy()
    df_target = df_analysis[(df_analysis['is_bug'] == 1) | (df_analysis['is_dragon'] == 1)].copy()

    return df, df_target, dfd, df_analysis, num_cols 


df, df_target, dfd, df_analysis, num_cols  = load_pokemon_data()

# データの概要
print(df.head())
print(df.info())
print(df.describe())
print(f"対象となるポケモンの数: {len(df_target)}匹")
