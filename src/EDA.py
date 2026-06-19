import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def pokreni_edu():
    os.makedirs('EDA_figures', exist_ok=True)

    # ========== UCITAVANJE ==========
    df = pd.read_csv('data/raw/bank-additional-full.csv', sep=';')
    df = df.dropna()    
    df = df.drop_duplicates()  

    df['pdays'] = df['pdays'].replace(999, -1)
    df['y_numeric'] = df['y'].map({'yes': 1, 'no': 0})

    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)

    # ========== DODAJ SAMO JEDAN GRAFIK ZA PREGLED SVIH NUMERICKIH PODATAKA ==========
    
    # 1. Histogrami za sve numeričke kolone (vidiš odmah distribuciju)
    numeric_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'nr.employed', 'cons.price.idx', 'cons.conf.idx', 'euribor3m']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue', bins=30)
        axes[i].set_title(f'Distribucija: {col}', fontweight='bold')
        axes[i].set_xlabel(col)
    
    plt.suptitle('Pregled svih numeričkih podataka', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('EDA_figures/00_all_numeric_distributions.png', dpi=150)
    plt.close()

    
    # ============================================
    # 1. DISTRIBUCIJA CILJNE PROMENLJIVE (y je 0/1)
    # ============================================
    plt.figure(figsize=(8, 5))
    counts = df['y'].value_counts().sort_index()
    labels = ['Ne', 'Da']
    colors = ['skyblue', 'salmon']
    bars = plt.bar(labels, counts.values, color=colors, edgecolor='black')
    plt.title('Distribucija pretplate (y)', fontsize=14, fontweight='bold')
    plt.xlabel('Pretplata')
    plt.ylabel('Broj klijenata')
    for bar, v in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                 f'{v:,}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('EDA_figures/01_target_distribution.png', dpi=150)
    plt.close()
    

    # ============================================
    # 2. PRETPLATA PO KATEGORIJSKIM ATRIBUTIMA
    # ============================================
    # 2a. Pretplata po zanimanju
    plt.figure(figsize=(12, 6))
    job_order = df['job'].value_counts().index
    sns.countplot(data=df, x='job', hue='y', order=job_order,
                  palette=['skyblue', 'salmon'])
    plt.title('Pretplata po zanimanju', fontsize=14, fontweight='bold')
    plt.xlabel('Zanimanje')
    plt.ylabel('Broj klijenata')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Pretplata', labels=['Ne', 'Da'])
    plt.tight_layout()
    plt.savefig('EDA_figures/02_job_distribution.png', dpi=150)
    plt.close()

    # 2b. Pretplata po nivou obrazovanja
    plt.figure(figsize=(10, 6))
    education_order = ['unknown', 'illiterate', 'basic.4y', 'basic.6y', 'basic.9y',
                       'high.school', 'professional.course', 'university.degree']
    education_order = [e for e in education_order if e in df['education'].unique()]
    sns.countplot(data=df, x='education', hue='y', order=education_order,
                  palette=['skyblue', 'salmon'])
    plt.title('Pretplata po nivou obrazovanja', fontsize=14, fontweight='bold')
    plt.xlabel('Nivo obrazovanja')
    plt.ylabel('Broj klijenata')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Pretplata', labels=['Ne', 'Da'])
    plt.tight_layout()
    plt.savefig('EDA_figures/03_education_distribution.png', dpi=150)
    plt.close()

    # 2c. Pretplata po mesecu
    plt.figure(figsize=(12, 6))
    month_order = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month_order = [m for m in month_order if m in df['month'].unique()]
    sns.countplot(data=df, x='month', hue='y', order=month_order,
                  palette=['skyblue', 'salmon'])
    plt.title('Pretplata po mesecu', fontsize=14, fontweight='bold')
    plt.xlabel('Mesec')
    plt.ylabel('Broj klijenata')
    plt.legend(title='Pretplata', labels=['Ne', 'Da'])
    plt.tight_layout()
    plt.savefig('EDA_figures/04_month_distribution.png', dpi=150)
    plt.close()

    # ============================================
    # 2d. Pretplata po ishodu prethodne kampanje (POUTCOME)
    # poutcome - ishod prethodne kampanje (da li su bili preplaceni?)

    # ZAKLJUCAK: 
    #   -oni koji su u prethodnoj kampanji bili preplaceni (poucome) -> 65% da ce opet biti
    #   -oni koji nisu bili preplaceni (failure) -> 14% da ce se sada preplatiti
    #   -oni koji nisu ni ucestvovali, nisu bili kontatirani (nonexistent) -> 9%
    # ============================================
    plt.figure(figsize=(10, 6))
    poutcome_order = ['unknown', 'failure', 'success']
    poutcome_order = [p for p in poutcome_order if p in df['poutcome'].unique()]
    sns.countplot(data=df, x='poutcome', hue='y', order=poutcome_order,
                palette=['skyblue', 'salmon'])
    plt.title('Pretplata po ishodu prethodne kampanje', fontsize=14, fontweight='bold')
    plt.xlabel('Ishod prethodne kampanje')
    plt.ylabel('Broj klijenata')
    plt.legend(title='Pretplata', labels=['Ne', 'Da'])
    plt.tight_layout()
    plt.savefig('EDA_figures/10_poutcome_distribution.png', dpi=150)
    plt.close()

    # Stopa pretplate po poutcome
    print()
    poutcome_rate = df.groupby('poutcome')['y'].apply(lambda x: (x == 'yes').mean() * 100)    
    print("\nStopa pretplate (%) po ishodu prethodne kampanje:")
    print(poutcome_rate.sort_values(ascending=False))

    plt.figure(figsize=(8, 6))
    poutcome_rate.sort_values().plot(kind='barh', color='lightgreen', edgecolor='black')
    plt.title('Stopa pretplate (%) po ishodu prethodne kampanje', fontsize=14, fontweight='bold')
    plt.xlabel('Stopa pretplate (%)')
    plt.ylabel('Ishod prethodne kampanje')
    for i, v in enumerate(poutcome_rate.sort_values()):
        plt.text(v + 0.5, i, f'{v:.1f}%', va='center')
    plt.tight_layout()
    plt.savefig('EDA_figures/11_poutcome_rate.png', dpi=150)
    plt.close()
    print()

    # ============================================
    # 3. NUMERICKI ATRIBUTI
    # ============================================
    # 3a. Starost - boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='y', y='age', hue='y',
                palette=['skyblue', 'salmon'], legend=False)
    plt.title('Distribucija starosti u odnosu na pretplatu', fontsize=14, fontweight='bold')
    plt.xlabel('Pretplata')
    plt.ylabel('Starost (godine)')
    plt.xticks([0, 1], ['Ne', 'Da'])
    plt.tight_layout()
    plt.savefig('EDA_figures/05_age_boxplot.png', dpi=150)
    plt.close()

    """
    # 3b. Broj kontakata - boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='y', y='campaign', hue='y',
                palette=['skyblue', 'salmon'], legend=False)
    plt.title('Broj kontakata tokom kampanje', fontsize=14, fontweight='bold')
    plt.xlabel('Pretplata')
    plt.ylabel('Broj kontakata')
    plt.xticks([0, 1], ['Ne', 'Da'])
    plt.tight_layout()
    plt.savefig('EDA_figures/06_campaign_boxplot.png', dpi=150)
    plt.close()
    """

    # ============================================
    # 4. STOPA PRETPLATE PO GRUPAMA
    # ============================================
    # 4a. Po starosnim grupama
    df['age_group'] = pd.cut(df['age'], bins=[18, 30, 40, 50, 60, 100],
                              labels=['18-30', '30-40', '40-50', '50-60', '60+'])
    subscription_rate = df.groupby('age_group', observed=True)['y'].apply(lambda x: (x == 'yes').mean() * 100)

    plt.figure(figsize=(8, 6))
    subscription_rate.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Stopa pretplate (%) po starosnim grupama', fontsize=14, fontweight='bold')
    plt.xlabel('Starosna grupa')
    plt.ylabel('Stopa pretplate (%)')
    plt.xticks(rotation=0)
    for i, v in enumerate(subscription_rate):
        plt.text(i, v + 0.5, f'{v:.1f}%', ha='center')
    plt.tight_layout()
    plt.savefig('EDA_figures/07_age_group_rate.png', dpi=150)
    plt.close()

    # 4b. Po zanimanju (top 5)
    job_rate = df.groupby('job')['y'].apply(lambda x: (x == 'yes').mean() * 100)
    top5_jobs = job_rate.nlargest(5)

    plt.figure(figsize=(10, 6))
    top5_jobs.plot(kind='barh', color='lightgreen', edgecolor='black')
    plt.title('Top 5 zanimanja po stopi pretplate (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Stopa pretplate (%)')
    plt.ylabel('Zanimanje')
    for i, v in enumerate(top5_jobs):
        plt.text(v + 0.5, i, f'{v:.1f}%', va='center')
    plt.tight_layout()
    plt.savefig('EDA_figures/08_top5_jobs_rate.png', dpi=150)
    plt.close()

    # ============================================
    # 5. KORELACIONA MATRICA
    # ============================================
    #  1 -> jaka direktna zavisnost
    # -1 -> jaka obrnuta zavisnost
    #  0 -> nemaju uticaja jedna na drugu
    numeric_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                    'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

    corr_df = df[numeric_cols + ['y_numeric']].copy()

    plt.figure(figsize=(10, 8))
    corr_matrix = corr_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
    plt.title('Korelaciona matrica', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('EDA_figures/correlation_heatmap.png', dpi=150)
    plt.close()



    # Stopa pretplate po danima u nedelji
    plt.figure(figsize=(10, 6))
    day_order = ['mon', 'tue', 'wed', 'thu', 'fri']
    day_order = [d for d in day_order if d in df['day_of_week'].unique()]
    sns.countplot(data=df, x='day_of_week', hue='y', order=day_order,
                palette=['skyblue', 'salmon'])
    plt.title('Pretplata po danima u nedelji', fontsize=14, fontweight='bold')
    plt.xlabel('Dan u nedelji')
    plt.ylabel('Broj klijenata')
    plt.legend(title='Pretplata', labels=['Ne', 'Da'])
    plt.tight_layout()
    plt.savefig('EDA_figures/12_day_of_week_distribution.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    pokreni_edu()