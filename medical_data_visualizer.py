import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")

# 2
df['overweight'] = ((df["weight"] / ((df["height"]/100)**2)) > 25)*1

# 3
df['cholesterol'] = df['cholesterol'].replace({1: 0, 2: 1, 3: 1})
df["gluc"] = df['gluc'].replace({1: 0, 2: 1, 3: 1})

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'overweight'])


    # 6
    df_cat = pd.melt(df,  id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    

    # 7
    df_counted = df_cat.value_counts().to_frame().reset_index(level=[0,1,2])
    df_counted.columns.values[3] = 'total'
    lines = np.sort(df_cat['variable'].unique())

    # 8
    fig = sns.catplot(data=df_counted, x='variable', y='total', col='cardio', hue='value', kind='bar', order=lines).fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['weight'] <= df['weight'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975))]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(8, 8))

    # 15

    sns.heatmap(corr, mask=mask, fmt=".1f",  square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    # 16
    fig.savefig('heatmap.png')
    return fig
