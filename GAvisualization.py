import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
import matplotlib.cm as cm
import os

# Configuration des styles et couleurs pour des graphiques impressionnants
plt.style.use('seaborn-v0_8-whitegrid')
colors = sns.color_palette("viridis", 6)
class_colors = {
    1: colors[0],
    2: colors[1],
    3: colors[2],
    4: colors[3],
    5: colors[4],
    6: colors[5]
}

# Fonction pour charger les données des fichiers de résultats TS
def load_tabu_search_results(file_paths):
    all_data = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Le fichier {file_path} n'existe pas.")
            continue
            
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        start_line = 0
        for i, line in enumerate(lines):
            if '===' in line:
                start_line = i + 1
                break
                
        for line in lines[start_line:]:
            parts = line.strip().split()
            if len(parts) >= 8:
                try:
                    jobs = int(parts[0])
                    molds = int(parts[1])
                    class_id = int(parts[2])
                    inst = int(parts[3])
                    lb = int(parts[4])
                    time = float(parts[5])
                    ts = int(parts[6])
                    gap = float(parts[7].strip('%'))
                    
                    # Déterminer si la solution est optimale 
                    is_optimal = gap <= 1.0
                    
                    all_data.append({
                        'Jobs': jobs,
                        'Molds': molds,
                        'Class': class_id,
                        'Instance': inst,
                        'LB': lb,
                        'Time': time,
                        'GA': ts,
                        'Gap': gap,
                        'IsOptimal': is_optimal
                    })
                except (ValueError, IndexError) as e:
                    print(f"Erreur lors de la lecture de la ligne: {line}")
                    print(f"Erreur: {e}")
    
    return pd.DataFrame(all_data)

# Charger les données de Tabu Search
file_paths = [
    'data/GA_results_whole.txt'
]

df_ts = load_tabu_search_results(file_paths)

if df_ts.empty:
    print("Aucune donnée n'a été chargée. Vérifiez les chemins des fichiers.")
else:
    print(f"Données chargées avec succès: {len(df_ts)} entrées.")
    
    # 1. Graphique du nombre de solutions optimales par classe
    optimal_by_class = df_ts.groupby('Class')['IsOptimal'].sum().reset_index()
    total_by_class = df_ts.groupby('Class').size().reset_index(name='Total')
    merged_data = pd.merge(optimal_by_class, total_by_class, on='Class')
    merged_data['OptimalPercentage'] = 100 * merged_data['IsOptimal'] / merged_data['Total']
    
    plt.figure(figsize=(14, 8))
    
    # Barres pour le nombre total d'instances par classe
    bars_total = plt.bar(merged_data['Class'], merged_data['Total'], alpha=0.3, color='lightgray')
    
    # Barres pour le nombre de solutions optimales
    bars_optimal = plt.bar(merged_data['Class'], merged_data['IsOptimal'], 
                        color=[class_colors[c] for c in merged_data['Class']])
    
    # Ajouter des labels au-dessus des barres
    for i, (opt, tot) in enumerate(zip(merged_data['IsOptimal'], merged_data['Total'])):
        plt.text(i+1, opt + 1, f"{opt} / {tot}\n({merged_data['OptimalPercentage'].iloc[i]:.1f}%)", 
                 ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Classe', fontsize=14)
    plt.ylabel('Nombre d\'instances', fontsize=14)
    plt.title('Nombre de solutions optimales  par classe', fontsize=16, pad=20)
    plt.xticks(merged_data['Class'], [f'C{c}' for c in merged_data['Class']])
    plt.legend(['Total d\'instances', 'Solutions optimales'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('GA_optimal_solutions_by_class.png', dpi=300)
    plt.show()

    # 2. Graphique du nombre de solutions optimales par taille de problème
    df_ts['ProblemSize'] = df_ts['Jobs'].astype(str) + ' jobs, ' + df_ts['Molds'].astype(str) + ' moules'
    
    optimal_by_size = df_ts.groupby(['ProblemSize', 'Jobs', 'Molds'])['IsOptimal'].agg(['sum', 'count']).reset_index()
    optimal_by_size['OptimalPercentage'] = 100 * optimal_by_size['sum'] / optimal_by_size['count']
    
    # Trier par nombre de jobs puis par nombre de moules
    optimal_by_size['SortKey'] = optimal_by_size['Jobs'] * 100 + optimal_by_size['Molds']
    optimal_by_size = optimal_by_size.sort_values('SortKey')
    
    plt.figure(figsize=(16, 8))
    
    x = np.arange(len(optimal_by_size))
    width = 0.35
    
    plt.bar(x - width/2, optimal_by_size['count'], width, color='lightgray', alpha=0.6, label='Total')
    plt.bar(x + width/2, optimal_by_size['sum'], width, color='#2ca02c', label='Optimales')
    
    for i, row in enumerate(optimal_by_size.itertuples()):
        plt.text(i - width/2, row.count + 1, f"{row.count}", ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, row.sum + 1, f"{row.sum}\n({row.OptimalPercentage:.1f}%)", 
                 ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Taille du problème', fontsize=14)
    plt.ylabel('Nombre d\'instances', fontsize=14)
    plt.title('Nombre de solutions optimales  par taille de problème', fontsize=16)
    plt.xticks(x, optimal_by_size['ProblemSize'], rotation=45, ha='right')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.tight_layout()
    plt.savefig('GA_optimal_solutions_by_size.png', dpi=300)
    plt.show()
    
    # 3. Heatmap des solutions optimales par classe et taille de problème
    pivot_df = df_ts.pivot_table(index=['Jobs', 'Molds'], 
                              columns='Class', 
                              values='IsOptimal', 
                              aggfunc='sum',
                              fill_value=0)
    
    # Créer un label combiné jobs/moules
    row_labels = [f"{j} jobs, {m} moules" for j, m in pivot_df.index]
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='g', linewidths=.5)
    plt.title('Nombre de solutions optimales par classe et taille de problème', fontsize=16)
    plt.ylabel('Taille du problème', fontsize=14)
    plt.xlabel('Classe', fontsize=14)
    
    # Personnaliser les labels des axes
    ax.set_yticklabels(row_labels)
    ax.set_xticklabels([f'C{c}' for c in pivot_df.columns])
    
    plt.tight_layout()
    plt.savefig('GA_optimal_solutions_heatmap.png', dpi=300)
    plt.show()
    
    # 4. Distribution des gaps avec mise en évidence des solutions optimales
    plt.figure(figsize=(14, 8))
    
    # Définir les intervalles pour l'histogramme
    bins = np.arange(0, df_ts['Gap'].max() + 5, 1)
    
    # Définir les couleurs pour les barres
    colors = []
    for b in bins[:-1]:
        if b <= 1:
            colors.append('#2ca02c')  # Vert pour les solutions optimales
        else:
            colors.append('#d62728')  # Rouge pour les solutions non-optimales
    
    # Créer l'histogramme
    n, bins, patches = plt.hist(df_ts['Gap'], bins=bins, edgecolor='black', linewidth=1.2)
    
    # Appliquer les couleurs aux barres
    for i, patch in enumerate(patches):
        patch.set_facecolor(colors[i])
    
    # Ajouter une ligne verticale pour le seuil d'optimalité
    plt.axvline(x=1, color='black', linestyle='--', linewidth=2, label='Seuil d\'optimalité ')
    
    # Calculer le pourcentage de solutions optimales
    optimal_count = sum(df_ts['IsOptimal'])
    total_count = len(df_ts)
    optimal_percentage = 100 * optimal_count / total_count
    
    # Ajouter une annotation
    plt.text(2, max(n) * 0.9, f"Solutions optimales: {optimal_count}/{total_count} ({optimal_percentage:.1f}%)", 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Gap (%)', fontsize=14)
    plt.ylabel('Nombre d\'instances', fontsize=14)
    plt.title('Distribution des gaps de GENETIC ALGORITHM avec mise en évidence des solutions optimales ', fontsize=16)
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('GA_gap_distribution_optimal.png', dpi=300)
    plt.show()
    
    # 5. Graphique avancé: carte de chaleur circulaire des solutions optimales par classe
    from matplotlib.pyplot import cm
    import matplotlib.colors as mcolors
    
    # Calculer le nombre de solutions optimales et total par classe
    optimal_counts = df_ts.groupby('Class')['IsOptimal'].sum()
    total_counts = df_ts.groupby('Class').size()
    optimal_percentages = 100 * optimal_counts / total_counts
    
    # Préparer les données pour la visualisation circulaire
    N = len(optimal_counts)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    optimal_values = optimal_counts.values.tolist()
    total_values = total_counts.values.tolist()
    optimal_pct_values = optimal_percentages.values.tolist()
    
    # Fermer le graphique en répétant la première valeur
    angles.append(angles[0])
    optimal_values.append(optimal_values[0])
    total_values.append(total_values[0])
    optimal_pct_values.append(optimal_pct_values[0])
    
    # Créer le graphique radar
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Remplir l'arrière-plan avec des anneaux représentant les pourcentages
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=100)
    
    for i in range(N):
        ax.fill_between([angles[i], angles[i+1]], 0, [total_values[i], total_values[i+1]], 
                        color='lightgray', alpha=0.3)
        ax.fill_between([angles[i], angles[i+1]], 0, [optimal_values[i], optimal_values[i+1]], 
                        color=cmap(norm(optimal_pct_values[i])), alpha=0.7)
    
    # Ajouter les pourcentages sur le graphique
    for i in range(N):
        angle = angles[i]
        if optimal_values[i] > 0:
            x = (optimal_values[i] + 2) * np.cos(angle)
            y = (optimal_values[i] + 2) * np.sin(angle)
            ax.text(angle, optimal_values[i] + 1, f"{optimal_pct_values[i]:.1f}%", 
                    ha='center', va='center', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # Configurer les axes et les labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'Classe {c}' for c in optimal_counts.index])
    ax.set_yticks(np.arange(0, max(total_values) + 10, 10))
    
    # Étiquette des axes et titre
    plt.title('Distribution des solutions optimales  par classe', fontsize=16, y=1.08)
    
    # Ajouter une légende de couleur pour les pourcentages
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Pourcentage de solutions optimales (%)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('GA_optimal_solutions_radar.png', dpi=300)
    plt.show()
    
    # 6. Analyse des temps d'exécution pour les solutions optimales vs non-optimales
    plt.figure(figsize=(12, 6))
    
    # Créer un violinplot pour comparer les temps d'exécution
    sns.violinplot(x='IsOptimal', y='Time', data=df_ts, palette=['#d62728', '#2ca02c'])
    
    # Ajouter des points pour montrer la distribution
    sns.stripplot(x='IsOptimal', y='Time', data=df_ts, color='black', alpha=0.4, jitter=True, size=4)
    
    plt.xlabel('Solution optimale ', fontsize=14)
    plt.ylabel('Temps d\'exécution (secondes)', fontsize=14)
    plt.title('Comparaison des temps d\'exécution: solutions optimales vs non-optimales', fontsize=16)
    plt.xticks([0, 1], ['Non optimale', 'Optimale'])
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    
    # Calculer et afficher les temps moyens
    mean_time_optimal = df_ts[df_ts['IsOptimal']]['Time'].mean()
    mean_time_nonoptimal = df_ts[~df_ts['IsOptimal']]['Time'].mean()
    
    plt.text(0, df_ts['Time'].max() * 0.9, f"Temps moyen: {mean_time_nonoptimal:.6f}s", 
             ha='center', va='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.text(1, df_ts['Time'].max() * 0.9, f"Temps moyen: {mean_time_optimal:.6f}s", 
             ha='center', va='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('ts_time_optimal_vs_nonoptimal.png', dpi=300)
    plt.show()
    
    # 7. Graphique d'évolution des solutions optimales en fonction de la classe et la taille du problème
    plt.figure(figsize=(16, 10))
    
    # Grouper par taille de problème et classe
    evolution_data = df_ts.groupby(['Jobs', 'Molds', 'Class'])['IsOptimal'].agg(['sum', 'count']).reset_index()
    evolution_data['OptimalPercentage'] = 100 * evolution_data['sum'] / evolution_data['count']
    
    # Créer une colonne combinée pour l'affichage
    evolution_data['ProblemSize'] = evolution_data['Jobs'].astype(str) + 'j-' + evolution_data['Molds'].astype(str) + 'm'
    
    # Trier par taille de problème
    evolution_data['SortKey'] = evolution_data['Jobs'] * 100 + evolution_data['Molds']
    evolution_data = evolution_data.sort_values('SortKey')
    
    # Créer un pivot pour le graphique
    pivot_evolution = evolution_data.pivot_table(index='ProblemSize', 
                                            columns='Class', 
                                            values='OptimalPercentage',
                                            fill_value=0)
    
    # Assurer que l'ordre des tailles de problème est correct
    problem_size_order = evolution_data.drop_duplicates('ProblemSize').sort_values('SortKey')['ProblemSize'].tolist()
    pivot_evolution = pivot_evolution.reindex(problem_size_order)
    
    # Tracer le graphique linéaire
    for col in pivot_evolution.columns:
        plt.plot(pivot_evolution.index, pivot_evolution[col], marker='o', linewidth=3, markersize=10,
                 color=class_colors[col], label=f'Classe {col}')
    
    plt.xlabel('Taille du problème (jobs-moules)', fontsize=14)
    plt.ylabel('Pourcentage de solutions optimales (%)', fontsize=14)
    plt.title('Évolution du pourcentage de solutions optimales par classe et taille de problème', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Classe', fontsize=12)
    plt.xticks(rotation=45)
    
    # Ajouter des étiquettes de valeur pour chaque point
    for col in pivot_evolution.columns:
        for i, val in enumerate(pivot_evolution[col]):
            plt.text(i, val + 2, f"{val:.1f}%", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('ts_optimal_solutions_evolution.png', dpi=300)
    plt.show()
    
    # 8. Graphique de résumé: Grille de performance de Tabu Search
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. Résumé global en haut à gauche
    ax1 = fig.add_subplot(gs[0, 0])
    optimal_count = sum(df_ts['IsOptimal'])
    total_count = len(df_ts)
    optimal_percentage = 100 * optimal_count / total_count
    
    sizes = [optimal_count, total_count - optimal_count]
    labels = [f'Optimales: {optimal_count} ({optimal_percentage:.1f}%)', 
              f'Non-optimales: {total_count - optimal_count} ({100 - optimal_percentage:.1f}%)']
    colors = ['#2ca02c', '#d62728']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
           wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax1.set_title('Résumé global des solutions', fontsize=14)
    
    # 2. Distribution des gaps en haut au milieu
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(df_ts['Gap'], bins=20, kde=True, ax=ax2, color='#1f77b4')
    ax2.axvline(x=1, color='red', linestyle='--', linewidth=2)
    ax2.text(1.5, ax2.get_ylim()[1] * 0.9, 'Seuil d\'optimalité ', 
            rotation=0, color='red', fontsize=10)
    ax2.set_xlabel('Gap (%)')
    ax2.set_title('Distribution des gaps', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Top N tailles de problème avec le plus de solutions optimales en haut à droite
    ax3 = fig.add_subplot(gs[0, 2])
    top_problems = df_ts.groupby(['ProblemSize'])['IsOptimal'].agg(['sum', 'count']).reset_index()
    top_problems['OptimalPercentage'] = 100 * top_problems['sum'] / top_problems['count']
    top_problems = top_problems.sort_values('OptimalPercentage', ascending=False).head(5)
    
    bars = ax3.barh(top_problems['ProblemSize'], top_problems['OptimalPercentage'], color='#ff7f0e')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f"{width:.1f}%", ha='left', va='center', fontsize=10)
    
    ax3.set_xlabel('Pourcentage de solutions optimales (%)')
    ax3.set_title('Top 5 des tailles de problème', fontsize=14)
    ax3.grid(True, linestyle='--', axis='x', alpha=0.7)
    
    # 4. Performance par classe en bas à gauche et au milieu
    ax4 = fig.add_subplot(gs[1, :2])
    
    class_performance = df_ts.groupby('Class')['IsOptimal'].agg(['sum', 'count']).reset_index()
    class_performance['OptimalPercentage'] = 100 * class_performance['sum'] / class_performance['count']
    
    x = np.arange(len(class_performance))
    width = 0.35
    
    bar1 = ax4.bar(x - width/2, class_performance['count'], width, label='Total', color='lightgray')
    bar2 = ax4.bar(x + width/2, class_performance['sum'], width, label='Optimales', 
                 color=[class_colors[c] for c in class_performance['Class']])
    
    for i, (bar, pct) in enumerate(zip(bar2, class_performance['OptimalPercentage'])):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 1, 
                f"{height} ({pct:.1f}%)", ha='center', va='bottom', fontsize=10)
    
    ax4.set_xlabel('Classe')
    ax4.set_ylabel('Nombre d\'instances')
    ax4.set_title('Performance par classe', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'C{c}' for c in class_performance['Class']])
    ax4.legend()
    ax4.grid(True, linestyle='--', axis='y', alpha=0.7)
    
    # 5. Temps d'exécution en bas à droite
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Groupe par classe pour obtenir les temps moyens
    time_by_class = df_ts.groupby('Class')['Time'].mean().reset_index()
    
    ax5.bar(time_by_class['Class'], time_by_class['Time'] * 1000, 
           color=[class_colors[c] for c in time_by_class['Class']])
    
    for i, val in enumerate(time_by_class['Time']):
        ax5.text(i+1, val*1000 + 0.01, f"{val*1000:.2f}", ha='center', va='bottom', fontsize=10)
    
    ax5.set_xlabel('Classe')
    ax5.set_ylabel('Temps moyen (ms)')
    ax5.set_title('Temps d\'exécution moyen par classe', fontsize=14)
    ax5.set_xticks(time_by_class['Class'])
    ax5.set_xticklabels([f'C{c}' for c in time_by_class['Class']])
    ax5.grid(True, linestyle='--', axis='y', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('ts_performance_summary.png', dpi=300)
    plt.show()
    
    # Afficher un résumé des résultats
    print("\n=== RÉSUMÉ DES RÉSULTATS DE GA ===")
    print(f"Nombre total d'instances: {len(df_ts)}")
    print(f"Solutions optimales : {optimal_count} ({optimal_percentage:.2f}%)")
    print(f"Temps d'exécution moyen: {df_ts['Time'].mean()*1000:.2f} ms")
    print("\nRépartition par classe:")
    
    for class_id in sorted(df_ts['Class'].unique()):
        class_data = df_ts[df_ts['Class'] == class_id]
        class_optimal = sum(class_data['IsOptimal'])
        class_total = len(class_data)
        class_pct = 100 * class_optimal / class_total
        print(f"  Classe {class_id}: {class_optimal}/{class_total} solutions optimales ({class_pct:.2f}%)")