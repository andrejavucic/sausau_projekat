import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ========== KREIRANJE FOLDER ZA GRAFOVE ==========
os.makedirs('reports/figures', exist_ok=True)

# ========== UCITAVANJE PODATAKA ==========
df = pd.read_csv('data/processed/bank-additional-cleaned.csv')

# ========== POSTAVKE ZA GRAFOVE ==========
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================
# 1. DISTRIBUCIJA CILJNE PROMENLJIVE
# ============================================
plt.figure(figsize=(8, 5))
df['y'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'], edgecolor='black')
plt.title('Distribucija pretplate (y)', fontsize=14, fontweight='bold')
plt.xlabel('Pretplata')
plt.ylabel('Broj klijenata')
plt.xticks(rotation=0)
for i, v in enumerate(df['y'].value_counts().values):
    plt.text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/01_target_distribution.png', dpi=150)
plt.close()

# ============================================
# 2. PRETPLATA PO KATEGORIJSKIM ATRIBUTIMA
# ============================================
# 2a. Pretplata po zanimanju
plt.figure(figsize=(12, 6))
job_order = df['job'].value_counts().index
sns.countplot(data=df, x='job', hue='y', order=job_order, palette=['skyblue', 'salmon'])
plt.title('Pretplata po zanimanju', fontsize=14, fontweight='bold')
plt.xlabel('Zanimanje')
plt.ylabel('Broj klijenata')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Pretplata')
plt.tight_layout()
plt.savefig('reports/figures/02_job_distribution.png', dpi=150)
plt.close()

# 2b. Pretplata po nivou obrazovanja
plt.figure(figsize=(10, 6))
education_order = ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 
                   'high.school', 'professional.course', 'university.degree']
education_order = [e for e in education_order if e in df['education'].unique()]
sns.countplot(data=df, x='education', hue='y', order=education_order, palette=['skyblue', 'salmon'])
plt.title('Pretplata po nivou obrazovanja', fontsize=14, fontweight='bold')
plt.xlabel('Nivo obrazovanja')
plt.ylabel('Broj klijenata')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Pretplata')
plt.tight_layout()
plt.savefig('reports/figures/03_education_distribution.png', dpi=150)
plt.close()

# 2c. Pretplata po mesecu
plt.figure(figsize=(12, 6))
month_order = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month_order = [m for m in month_order if m in df['month'].unique()]
sns.countplot(data=df, x='month', hue='y', order=month_order, palette=['skyblue', 'salmon'])
plt.title('Pretplata po mesecu', fontsize=14, fontweight='bold')
plt.xlabel('Mesec')
plt.ylabel('Broj klijenata')
plt.legend(title='Pretplata')
plt.tight_layout()
plt.savefig('reports/figures/04_month_distribution.png', dpi=150)
plt.close()

# ============================================
# 3. NUMERIČKI ATRIBUTI
# ============================================
# 3a. Starost - boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='y', y='age', palette=['skyblue', 'salmon'])
plt.title('Distribucija starosti u odnosu na pretplatu', fontsize=14, fontweight='bold')
plt.xlabel('Pretplata')
plt.ylabel('Starost (godine)')
plt.tight_layout()
plt.savefig('reports/figures/05_age_boxplot.png', dpi=150)
plt.close()

# 3b. Broj kontakata - boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='y', y='campaign', palette=['skyblue', 'salmon'])
plt.title('Broj kontakata tokom kampanje', fontsize=14, fontweight='bold')
plt.xlabel('Pretplata')
plt.ylabel('Broj kontakata')
plt.tight_layout()
plt.savefig('reports/figures/06_campaign_boxplot.png', dpi=150)
plt.close()

# ============================================
# 4. STOPA PRETPLATE PO GRUPAMA
# ============================================
# 4a. Po starosnim grupama
df['age_group'] = pd.cut(df['age'], bins=[18, 30, 40, 50, 60, 100],
                          labels=['18-30', '30-40', '40-50', '50-60', '60+'])
subscription_rate = df.groupby('age_group', observed=True)['y'].apply(lambda x: (x == 'yes').mean()) * 100

plt.figure(figsize=(8, 6))
subscription_rate.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Stopa pretplate (%) po starosnim grupama', fontsize=14, fontweight='bold')
plt.xlabel('Starosna grupa')
plt.ylabel('Stopa pretplate (%)')
plt.xticks(rotation=0)
for i, v in enumerate(subscription_rate):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center')
plt.tight_layout()
plt.savefig('reports/figures/07_age_group_rate.png', dpi=150)
plt.close()

# 4b. Po zanimanju (top 5)
job_rate = df.groupby('job')['y'].apply(lambda x: (x == 'yes').mean()) * 100
top5_jobs = job_rate.nlargest(5)

plt.figure(figsize=(10, 6))
top5_jobs.plot(kind='barh', color='lightgreen', edgecolor='black')
plt.title('Top 5 zanimanja po stopi pretplate (%)', fontsize=14, fontweight='bold')
plt.xlabel('Stopa pretplate (%)')
plt.ylabel('Zanimanje')
for i, v in enumerate(top5_jobs):
    plt.text(v + 0.5, i, f'{v:.1f}%', va='center')
plt.tight_layout()
plt.savefig('reports/figures/08_top5_jobs_rate.png', dpi=150)
plt.close()

# ============================================
# 5. KORELACIONA MATRICA
# ============================================
numeric_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

corr_df = df[numeric_cols].copy()
corr_df['y'] = (df['y'] == 'yes').astype(int)

plt.figure(figsize=(10, 8))
corr_matrix = corr_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('Korelaciona matrica', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/09_correlation_heatmap.png', dpi=150)
plt.close()