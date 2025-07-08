import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time
from collections import defaultdict

execution_times = defaultdict(list)

def measure_time(func):
    # Decorator que registra o tempo de execução dos métodos
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        execution_times[func.__name__].append(elapsed)
        print(f"[Timing] {func.__name__} levou {elapsed:.6f} s")
        return result
    return wrapper

# FUNÇÕES DE VISUALIZAÇÃO E COMPARAÇÃO (sem alterações significativas)

def plot_rotas(units_df, emergencies_df, routes, total_dist, method_name="Rotas atribuídas", elapsed=None):
    """Visualiza as rotas atribuídas graficamente."""
    plt.figure(figsize=(10, 8))
    cores = plt.cm.get_cmap('tab10', len(units_df))

    for i, (unit_id, em_ids) in enumerate(routes.items()):
        pos_atual = units_df.loc[units_df['unit_id'] == unit_id, ['x', 'y']].values
        if len(pos_atual) == 0:
            continue
        x_unit, y_unit = pos_atual[0]
        rota_x = [x_unit]
        rota_y = [y_unit]

        for em_id in em_ids:
            em = emergencies_df.loc[emergencies_df['emergency_id'] == em_id, ['x', 'y']]
            if not em.empty:
                rota_x.append(em.iloc[0]['x'])
                rota_y.append(em.iloc[0]['y'])

        plt.plot(rota_x, rota_y, marker='o', color=cores(i), label=f'Unidade {unit_id}')
        plt.scatter(x_unit, y_unit, color=cores(i), marker='s', s=100, edgecolors='black')

    title = method_name
    if elapsed is not None:
        title += f"\nDistância Total: {total_dist:.2f} (Tempo: {elapsed:.3f}s)"
    plt.title(title)

    plt.scatter(emergencies_df['x'], emergencies_df['y'], color='lightgray', marker='x', label='Emergências')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('rotas_plot.png')
    plt.show()

def comparar_solucoes(units_df, emergencies_df):
    """Gera e compara soluções usando diferentes heurísticas"""
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    def plot_subplot(ax, title, routes):
        if not routes: # Se não houver rotas, não plota nada
            ax.set_title(f"{title}\n(sem solução)")
            ax.axis('off')
            return
            
        cores = plt.cm.get_cmap('tab10', len(routes))
        for i, (unit_id, em_ids) in enumerate(routes.items()):
            pos_atual = units_df.loc[units_df['unit_id'] == unit_id, ['x', 'y']].values
            if len(pos_atual) == 0:
                continue
            x_unit, y_unit = pos_atual[0]
            rota_x = [x_unit]
            rota_y = [y_unit]
            for em_id in em_ids:
                em = emergencies_df[emergencies_df['emergency_id'] == em_id]
                if not em.empty:
                    rota_x.append(em.iloc[0]['x'])
                    rota_y.append(em.iloc[0]['y'])
            ax.plot(rota_x, rota_y, marker='o', color=cores(i), label=f'{unit_id}')
            ax.scatter(x_unit, y_unit, color=cores(i), marker='s', s=80, edgecolors='black')
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.legend()

    # Solução Greedy
    t0 = time.perf_counter()
    greedy = construct_initial_solution(units_df, emergencies_df)
    tg = time.perf_counter() - t0

    dist_greedy = calcular_distancia_total(units_df, emergencies_df, greedy)
    print(f"[Timing] greedy levou {tg:.6f}s")
    plot_subplot(axs[0, 0], f"Gulosa\nDist: {dist_greedy:.2f}  Tempo: {tg:.3f}s", greedy)

    # Solução GRASP (sem busca local)
    t0 = time.perf_counter()
    grasp = construct_grasp_solution(units_df, emergencies_df, alpha=0.3)
    tg = time.perf_counter() - t0

    dist_grasp = calcular_distancia_total(units_df, emergencies_df, grasp)
    print(f"[Timing] grasp levou {tg:.6f}s")
    plot_subplot(axs[0, 1], f"GRASP\nDist: {dist_grasp:.2f}  Tempo: {tg:.3f}s", grasp)

    # Solução Nearest Neighbor
    t0 = time.perf_counter()
    nn = construct_nearest_neighbor_solution(units_df, emergencies_df)
    tn = time.perf_counter() - t0

    dist_nn = calcular_distancia_total(units_df, emergencies_df, nn)
    print(f"[Timing] nn levou {tn:.6f}s")
    plot_subplot(axs[1, 0], f"NN\nDist: {dist_nn:.2f}  Tempo: {tg:.3f}s", nn)
    
    # {NOVO} Solução GRASP com Busca Local
    t0 = time.perf_counter()
    grasp_ls = grasp_com_busca_local(units_df, emergencies_df, max_iterations=5, alpha=0.3)
    tg = time.perf_counter() - t0

    dist_grasp_ls = calcular_distancia_total(units_df, emergencies_df, grasp_ls)
    print(f"[Timing] GRASP + Busca Local levou {tg:.6f}s")
    plot_subplot(axs[1, 1], f"GRASP + Busca Local\nDist: {dist_grasp_ls:.2f}  Tempo: {tg:.3f}s", grasp_ls)

    plt.suptitle("Comparação de Heurísticas", fontsize=16)
    plt.savefig("comparacao_solucoes.png")
    plt.show()

    print("\nDistâncias totais para comparação:")
    print(f"  Gulosa: {dist_greedy:.2f}")
    print(f"  GRASP:  {dist_grasp:.2f}")
    print(f"  NN:     {dist_nn:.2f}")
    print(f"  GRASP + Busca Local: {dist_grasp_ls:.2f}")

def comparar_solucoes_grasp(units_df, emergencies_df, alpha=0.3):
    """
    Gera uma única solução GRASP e a compara com sua versão refinada pela Busca Local,
    exibindo os dois resultados lado a lado e salvando ambos em arquivos CSV.
    """
    # 1. Ajuste para criar 1 linha e 2 colunas, para 2 gráficos lado a lado.
    #    O figsize foi ajustado para um formato mais horizontal.
    fig, axs = plt.subplots(1, 2, figsize=(20, 9))
    plt.subplots_adjust(wspace=0.25) # Adiciona um espaço entre os gráficos

    # Função interna para desenhar o gráfico (nenhuma mudança aqui)
    def plot_subplot(ax, title, routes):
        if not routes:
            ax.set_title(f"{title}\n(sem solução)")
            ax.axis('off')
            return
            
        cores = plt.cm.get_cmap('tab10', len(routes))
        for i, (unit_id, em_ids) in enumerate(routes.items()):
            pos_atual = units_df.loc[units_df['unit_id'] == unit_id, ['x', 'y']].values
            if len(pos_atual) == 0:
                continue
            x_unit, y_unit = pos_atual[0]
            rota_x = [x_unit]
            rota_y = [y_unit]
            for em_id in em_ids:
                em = emergencies_df[emergencies_df['emergency_id'] == em_id]
                if not em.empty:
                    rota_x.append(em.iloc[0]['x'])
                    rota_y.append(em.iloc[0]['y'])
            ax.plot(rota_x, rota_y, marker='o', color=cores(i), label=f'Unidade {unit_id}')
            ax.scatter(x_unit, y_unit, color=cores(i), marker='s', s=80, edgecolors='black')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Coordenada X")
        ax.set_ylabel("Coordenada Y")
        ax.grid(True)
        ax.legend()

    print("\n--- Iniciando Comparação: Efeito da Busca Local ---")

    # 2. Gera UMA solução inicial usando a construção GRASP
    print("FASE 1: Construindo solução com GRASP...")
    grasp_inicial = construct_grasp_solution(units_df, emergencies_df, alpha=alpha)
    dist_inicial = calcular_distancia_total(units_df, emergencies_df, grasp_inicial)
    print(f"  > Distância da solução construída: {dist_inicial:.2f}")

    # 3. Aplica a Busca Local nessa solução inicial para refiná-la
    print("FASE 2: Refinando a mesma solução com Busca Local...")
    grasp_refinado = busca_local(grasp_inicial, units_df, emergencies_df)
    dist_refinada = calcular_distancia_total(units_df, emergencies_df, grasp_refinado)
    print(f"  > Distância da solução refinada: {dist_refinada:.2f}")
    
    # Gráfico da Esquerda: Solução GRASP antes da Busca Local
    plot_subplot(axs[0], f"Antes da Busca Local (Apenas Construção GRASP)\nDistância: {dist_inicial:.2f}", grasp_inicial)

    # Gráfico da Direita: Solução GRASP depois da Busca Local
    plot_subplot(axs[1], f"Após a Busca Local (Solução Refinada)\nDistância: {dist_refinada:.2f}", grasp_refinado)

    plt.suptitle("Comparativo do Efeito da Busca Local", fontsize=18, weight='bold')
    plt.savefig("comparacao_grasp_vs_grasp_ls.png")
    plt.show()

    # Imprime a melhora da solucao
    print("\n--- Resumo da Comparação ---")
    print(f"Distância inicial (Apenas Construção): {dist_inicial:.2f}")
    print(f"Distância final (Após Refinamento): {dist_refinada:.2f}")
    melhora = dist_inicial - dist_refinada
    if dist_inicial > 0 and melhora > 0:
        percentual = (melhora / dist_inicial) * 100
        print(f"Melhora de {melhora:.2f} na distância ({percentual:.2f}%)")
    else:
        print("A Busca Local não encontrou uma solução melhor nesta execução.")

    print("\nSalvando resultados em CSV...")
    salvar_rotas_csv(grasp_inicial, 'rotas_grasp_inicial.csv')
    salvar_rotas_csv(grasp_refinado, 'rotas_grasp_refinado_com_bl.csv')

# FUNÇÕES DE CÁLCULO E CONSTRUÇÃO DE SOLUÇÃO

def calcular_distancia_total(units_df, emergencies_df, routes):
    """Calcula a distância total percorrida por todas unidades."""
    total = 0.0
    for unit_id, em_ids in routes.items():
        unidade = units_df[units_df['unit_id'] == unit_id]
        if unidade.empty:
            continue
        pos_atual = unidade[['x', 'y']].iloc[0].values
        for em_id in em_ids:
            emergencia = emergencies_df[emergencies_df['emergency_id'] == em_id]
            if emergencia.empty:
                continue
            pos_em = emergencia[['x', 'y']].iloc[0].values
            dist = np.linalg.norm(pos_em - pos_atual)
            total += dist
            pos_atual = pos_em
    return total

@measure_time
def construct_initial_solution(units_df, emergencies_df):
    """Heurística gulosa: atribui cada emergência à unidade mais próxima no momento."""
    pending = emergencies_df.copy()
    routes = {unit: [] for unit in units_df['unit_id']}
    current_positions = {row['unit_id']: (row['x'], row['y']) for _, row in units_df.iterrows()}

    while not pending.empty:
        best_dist = float('inf')
        best_unit, best_idx = None, None

        for unit_id, pos in current_positions.items():
            dx = pending['x'] - pos[0]
            dy = pending['y'] - pos[1]
            dists = np.hypot(dx, dy)
            if dists.isnull().all():
                continue
            idx_min = dists.idxmin()
            dist_min = dists.loc[idx_min]
            if dist_min < best_dist:
                best_dist = dist_min
                best_unit = unit_id
                best_idx = idx_min
        
        if best_idx is None:
            break

        chosen = pending.loc[best_idx]
        routes[best_unit].append(chosen['emergency_id'])
        current_positions[best_unit] = (chosen['x'], chosen['y'])
        pending = pending.drop(best_idx)

    return routes

@measure_time
def construct_grasp_solution(units_df, emergencies_df, alpha=0.3):
    """Fase de construção GRASP: utiliza lista restrita de candidatos (RCL) baseada em alpha."""
    pending = emergencies_df.copy()
    routes = {unit: [] for unit in units_df['unit_id']}
    current_positions = {row['unit_id']: (row['x'], row['y']) for _, row in units_df.iterrows()}

    while not pending.empty:
        candidatos = []
        for unit_id, pos in current_positions.items():
            dx = pending['x'] - pos[0]
            dy = pending['y'] - pos[1]
            dists = np.hypot(dx, dy)
            for idx, dist in dists.items():
                if not np.isnan(dist):
                    candidatos.append((unit_id, idx, dist))
        
        if not candidatos:
            break

        candidatos.sort(key=lambda x: x[2])
        
        # Limita o tamanho da RCL para não exceder o número de candidatos
        limite = min(int(len(candidatos) * alpha) + 1, len(candidatos))
        rcl = candidatos[:limite]
        chosen_unit, chosen_idx, _ = random.choice(rcl)

        chosen = pending.loc[chosen_idx]
        routes[chosen_unit].append(chosen['emergency_id'])
        current_positions[chosen_unit] = (chosen['x'], chosen['y'])
        pending = pending.drop(chosen_idx)

    return routes

@measure_time
def construct_nearest_neighbor_solution(units_df, emergencies_df):
    """Heurística Nearest Neighbor para cada unidade até esgotar emergências."""
    pending = emergencies_df.copy()
    routes = {unit: [] for unit in units_df['unit_id']}
    current_positions = {row['unit_id']: (row['x'], row['y']) for _, row in units_df.iterrows()}

    unit_ids = list(current_positions.keys())
    unit_idx = 0
    while not pending.empty:
        # Passa ciclicamente pelas unidades
        unit_id = unit_ids[unit_idx % len(unit_ids)]
        
        pos = current_positions[unit_id]
        dx = pending['x'] - pos[0]
        dy = pending['y'] - pos[1]
        dists = np.hypot(dx, dy)
        
        if dists.isnull().all():
            break

        idx_min = dists.idxmin()
        chosen = pending.loc[idx_min]
        routes[unit_id].append(chosen['emergency_id'])
        current_positions[unit_id] = (chosen['x'], chosen['y'])
        pending = pending.drop(idx_min)
        
        unit_idx += 1

    return routes

# FASE DE BUSCA LOCAL E FUNÇÃO PRINCIPAL DO GRASP

def busca_local(solucao_inicial, units_df, emergencies_df):
    """
    Aplica Busca Local para refinar uma solução.
    Usa o movimento de 'realocação' de uma emergência entre duas rotas.
    Adota a estratégia de 'primeira melhora' (first improvement).
    """
    solucao_atual = copy.deepcopy(solucao_inicial)
    melhor_distancia = calcular_distancia_total(units_df, emergencies_df, solucao_atual)
    
    houve_melhora = True
    while houve_melhora:
        houve_melhora = False
        # Itera sobre cada unidade e sua rota
        for id_unidade_origem in list(solucao_atual.keys()):
            # Itera sobre cada emergência na rota (de trás para frente para evitar problemas ao remover)
            for i in range(len(solucao_atual[id_unidade_origem]) - 1, -1, -1):
                emergencia_a_mover = solucao_atual[id_unidade_origem][i]
                
                # Tenta mover a emergência para a rota de outra unidade
                for id_unidade_destino in list(solucao_atual.keys()):
                    if id_unidade_origem == id_unidade_destino:
                        continue # Pula se for a mesma rota
                    
                    # Cria uma solução candidata temporária aplicando o movimento
                    solucao_candidata = copy.deepcopy(solucao_atual)
                    emergencia_removida = solucao_candidata[id_unidade_origem].pop(i)
                    solucao_candidata[id_unidade_destino].append(emergencia_removida) # Insere no final por simplicidade

                    # Calcula o custo da solução candidata
                    distancia_candidata = calcular_distancia_total(units_df, emergencies_df, solucao_candidata)
                    
                    # Se a nova solução for melhor, aceita-a e reinicia a busca
                    if distancia_candidata < melhor_distancia:
                        solucao_atual = solucao_candidata
                        melhor_distancia = distancia_candidata
                        houve_melhora = True
                        # Sai dos loops para reiniciar a busca a partir da nova solução melhorada
                        break
                if houve_melhora:
                    break
            if houve_melhora:
                break
    
    return solucao_atual

def busca_local_2opt(solucao_inicial, units_df, emergencies_df):
    """
    Busca Local com movimento 2-opt dentro de cada rota individual.
    """
    solucao = copy.deepcopy(solucao_inicial)
    melhorou = True

    while melhorou:
        melhorou = False
        for unidade, rota in solucao.items():
            rota_melhor = rota.copy()
            melhor_dist = rota_dist(units_df, emergencies_df, unidade, rota_melhor)
            n = len(rota_melhor)
            for i in range(n - 1):
                for j in range(i + 2, n):
                    nova_rota = rota_melhor[:i+1] + rota_melhor[i+1:j+1][::-1] + rota_melhor[j+1:]
                    nova_dist = rota_dist(units_df, emergencies_df, unidade, nova_rota)
                    if nova_dist < melhor_dist:
                        rota_melhor = nova_rota
                        melhor_dist = nova_dist
                        melhorou = True
            solucao[unidade] = rota_melhor
    return solucao

def perturbar_solucao(routes):
    import random
    from copy import deepcopy

    nova_solucao = deepcopy(routes)
    unidades = list(nova_solucao.keys())

    for _ in range(10):  # tenta no máximo 10 vezes encontrar duas unidades com emergências
        u1, u2 = random.sample(unidades, 2)
        if nova_solucao[u1] and nova_solucao[u2]:
            i1 = random.randrange(len(nova_solucao[u1]))
            i2 = random.randrange(len(nova_solucao[u2]))
            # swap entre emergências
            nova_solucao[u1][i1], nova_solucao[u2][i2] = nova_solucao[u2][i2], nova_solucao[u1][i1]
            break

    return nova_solucao

def rota_dist(units_df, emergencies_df, unidade, rota):
    """Calcula a distância total de uma rota individual"""
    unidade_pos = units_df[units_df['unit_id'] == unidade][['x', 'y']].iloc[0].values
    total = 0
    pos_atual = unidade_pos
    for em_id in rota:
        pos_em = emergencies_df[emergencies_df['emergency_id'] == em_id][['x', 'y']].iloc[0].values
        total += np.linalg.norm(pos_em - pos_atual)
        pos_atual = pos_em
    return total

@measure_time
def grasp_com_busca_local(units_df, emergencies_df, max_iterations=10, alpha=0.3):
    """
    Executa o algoritmo GRASP completo: construção + busca local.
    Retorna a melhor solução encontrada após um número de iterações.
    """
    melhor_solucao_geral = None
    melhor_distancia_geral = float('inf')

    print(f"\nExecutando GRASP com Busca Local ({max_iterations} iterações, alpha={alpha})...")
    
    for i in range(max_iterations):
        # 1. Fase de Construção
        solucao_construida = construct_grasp_solution(units_df, emergencies_df, alpha)
        
        # 2. Fase de Busca Local
        solucao_refinada = busca_local(solucao_construida, units_df, emergencies_df)
        
        distancia_atual = calcular_distancia_total(units_df, emergencies_df, solucao_refinada)
        
        print(f"  Iteração {i+1}/{max_iterations}: Distância = {distancia_atual:.2f}")

        # 3. Atualiza a melhor solução encontrada
        if distancia_atual < melhor_distancia_geral:
            melhor_distancia_geral = distancia_atual
            melhor_solucao_geral = solucao_refinada
            print(f"    -> Nova melhor solução encontrada!")

    print(f"\nGRASP com Busca Local concluído. Melhor distância: {melhor_distancia_geral:.2f}")
    return melhor_solucao_geral

@measure_time
def gulosa_com_busca_local(units_df, emergencies_df):
    """
    Executa a heurística Gulosa + Busca Local:
    1) Constrói uma solução inicial usando a heurística gulosa.
    2) Refina essa solução aplicando a Busca Local.
    Retorna a solução refinada.
    """
    print("\nExecutando Gulosa com Busca Local...")

    # 1. Fase de construção gulosa
    solucao_gulosa = construct_initial_solution(units_df, emergencies_df)
    dist_inicial = calcular_distancia_total(units_df, emergencies_df, solucao_gulosa)
    print(f"  Distância inicial (Gulosa): {dist_inicial:.2f}")

    # 2. Fase de Busca Local
    solucao_refinada = busca_local(solucao_gulosa, units_df, emergencies_df)
    dist_refinada = calcular_distancia_total(units_df, emergencies_df, solucao_refinada)
    print(f"  Distância após Busca Local: {dist_refinada:.2f}")

    # 3. Mostra melhora, se houver
    melhora = dist_inicial - dist_refinada
    if melhora > 0:
        percentual = (melhora / dist_inicial) * 100
        print(f"  Melhora de {melhora:.2f} ({percentual:.2f}%)")
    else:
        print("  Nenhuma melhoria encontrada pela Busca Local.")

    return solucao_refinada

@measure_time
def nn_com_busca_local(units_df, emergencies_df):
    """
    Executa a heurística Nearest Neighbor + Busca Local:
    1) Constrói uma solução inicial usando Nearest Neighbor.
    2) Refina essa solução aplicando a Busca Local.
    Retorna a solução refinada.
    """
    print("\nExecutando Nearest Neighbor com Busca Local...")

    # 1. Fase de construção Nearest Neighbor
    solucao_nn = construct_nearest_neighbor_solution(units_df, emergencies_df)
    dist_inicial = calcular_distancia_total(units_df, emergencies_df, solucao_nn)
    print(f"  Distância inicial (NN): {dist_inicial:.2f}")

    # 2. Fase de Busca Local
    solucao_refinada = busca_local(solucao_nn, units_df, emergencies_df)
    dist_refinada = calcular_distancia_total(units_df, emergencies_df, solucao_refinada)
    print(f"  Distância após Busca Local: {dist_refinada:.2f}")

    # 3. Mostra melhora, se houver
    melhora = dist_inicial - dist_refinada
    if melhora > 0:
        percentual = (melhora / dist_inicial) * 100
        print(f"  Melhora de {melhora:.2f} ({percentual:.2f}%)")
    else:
        print("  Nenhuma melhoria encontrada pela Busca Local.")

    return solucao_refinada

def comparar_solucoes_greedy(units_df, emergencies_df):
    """
    Gera uma solução Gulosa pura e a compara com sua versão refinada pela Busca Local.
    Exibe os dois resultados lado a lado e salva ambos em arquivos CSV.
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 9))
    plt.subplots_adjust(wspace=0.25)

    def plot_subplot(ax, title, routes):
        if not routes:
            ax.set_title(f"{title}\n(sem solução)")
            ax.axis('off')
            return
        
        cores = plt.cm.get_cmap('tab10', len(routes))
        for i, (unit_id, em_ids) in enumerate(routes.items()):
            pos_atual = units_df.loc[units_df['unit_id'] == unit_id, ['x', 'y']].values
            if len(pos_atual) == 0:
                continue
            x_unit, y_unit = pos_atual[0]
            rota_x = [x_unit]
            rota_y = [y_unit]
            for em_id in em_ids:
                em = emergencies_df[emergencies_df['emergency_id'] == em_id]
                if not em.empty:
                    rota_x.append(em.iloc[0]['x'])
                    rota_y.append(em.iloc[0]['y'])
            ax.plot(rota_x, rota_y, marker='o', color=cores(i), label=f'Unidade {unit_id}')
            ax.scatter(x_unit, y_unit, color=cores(i), marker='s', s=80, edgecolors='black')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Coordenada X")
        ax.set_ylabel("Coordenada Y")
        ax.grid(True)
        ax.legend()

    print("\n--- Iniciando Comparação: Gulosa vs Gulosa + Busca Local ---")

    # Solução Gulosa
    print("FASE 1: Construindo solução Gulosa...")
    greedy_inicial = construct_initial_solution(units_df, emergencies_df)
    dist_inicial = calcular_distancia_total(units_df, emergencies_df, greedy_inicial)
    print(f"  > Distância da solução Gulosa: {dist_inicial:.2f}")

    # Gulosa + Busca Local
    print("FASE 2: Refinando a mesma solução com Busca Local...")
    greedy_refinado = busca_local_2opt(greedy_inicial, units_df, emergencies_df)
    dist_refinada = calcular_distancia_total(units_df, emergencies_df, greedy_refinado)
    print(f"  > Distância após Busca Local: {dist_refinada:.2f}")

    # Plot
    plot_subplot(axs[0], f"Gulosa (Distância: {dist_inicial:.2f})", greedy_inicial)
    plot_subplot(axs[1], f"Gulosa + Busca Local (Distância: {dist_refinada:.2f})", greedy_refinado)

    plt.suptitle("Comparativo: Gulosa vs Gulosa + Busca Local", fontsize=18, weight='bold')
    plt.savefig("comparacao_greedy_vs_greedy_bl.png")
    plt.show()

    # Resumo
    print("\n--- Resumo da Comparação ---")
    print(f"Distância inicial (Gulosa): {dist_inicial:.2f}")
    print(f"Distância final (Após Busca Local): {dist_refinada:.2f}")
    melhora = dist_inicial - dist_refinada
    if dist_inicial > 0 and melhora > 0:
        percentual = (melhora / dist_inicial) * 100
        print(f"Melhora de {melhora:.2f} na distância ({percentual:.2f}%)")
    else:
        print("A Busca Local não encontrou uma solução melhor nesta execução.")

    print("\nSalvando resultados em CSV...")
    salvar_rotas_csv(greedy_inicial, 'rotas_greedy_inicial.csv')
    salvar_rotas_csv(greedy_refinado, 'rotas_greedy_refinado_com_bl.csv')

def comparar_solucoes_nn(units_df, emergencies_df):
    """
    Gera uma única solução Nearest Neighbor e a compara com sua versão refinada pela Busca Local,
    exibindo os dois resultados lado a lado e salvando ambos em arquivos CSV.
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 9))
    plt.subplots_adjust(wspace=0.25)

    def plot_subplot(ax, title, routes):
        if not routes:
            ax.set_title(f"{title}\n(sem solução)")
            ax.axis('off')
            return
        
        cores = plt.cm.get_cmap('tab10', len(routes))
        for i, (unit_id, em_ids) in enumerate(routes.items()):
            pos_atual = units_df.loc[units_df['unit_id'] == unit_id, ['x', 'y']].values
            if len(pos_atual) == 0:
                continue
            x_unit, y_unit = pos_atual[0]
            rota_x = [x_unit]
            rota_y = [y_unit]
            for em_id in em_ids:
                em = emergencies_df[emergencies_df['emergency_id'] == em_id]
                if not em.empty:
                    rota_x.append(em.iloc[0]['x'])
                    rota_y.append(em.iloc[0]['y'])
            ax.plot(rota_x, rota_y, marker='o', color=cores(i), label=f'Unidade {unit_id}')
            ax.scatter(x_unit, y_unit, color=cores(i), marker='s', s=80, edgecolors='black')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Coordenada X")
        ax.set_ylabel("Coordenada Y")
        ax.grid(True)
        ax.legend()

    print("\n--- Iniciando Comparação: Nearest Neighbor vs Nearest Neighbor + Busca Local ---")

    # Solução Nearest Neighbor
    print("FASE 1: Construindo solução Nearest Neighbor...")
    nn_inicial = construct_nearest_neighbor_solution(units_df, emergencies_df)
    dist_inicial = calcular_distancia_total(units_df, emergencies_df, nn_inicial)
    print(f"  > Distância da solução NN: {dist_inicial:.2f}")

    # Nearest Neighbor + Busca Local
    print("FASE 2: Refinando a mesma solução com Busca Local...")
    nn_refinado = busca_local(nn_inicial, units_df, emergencies_df)
    dist_refinada = calcular_distancia_total(units_df, emergencies_df, nn_refinado)
    print(f"  > Distância após Busca Local: {dist_refinada:.2f}")

    # Plot
    plot_subplot(axs[0], f"Nearest Neighbor (Distância: {dist_inicial:.2f})", nn_inicial)
    plot_subplot(axs[1], f"Nearest Neighbor + Busca Local (Distância: {dist_refinada:.2f})", nn_refinado)

    plt.suptitle("Comparativo: Nearest Neighbor vs Nearest Neighbor + Busca Local", fontsize=18, weight='bold')
    plt.savefig("comparacao_nn_vs_nn_bl.png")
    plt.show()

    # Resumo
    print("\n--- Resumo da Comparação ---")
    print(f"Distância inicial (NN): {dist_inicial:.2f}")
    print(f"Distância final (Após Busca Local): {dist_refinada:.2f}")
    melhora = dist_inicial - dist_refinada
    if dist_inicial > 0 and melhora > 0:
        percentual = (melhora / dist_inicial) * 100
        print(f"Melhora de {melhora:.2f} na distância ({percentual:.2f}%)")
    else:
        print("A Busca Local não encontrou uma solução melhor nesta execução.")

    print("\nSalvando resultados em CSV...")
    salvar_rotas_csv(nn_inicial, 'rotas_nn_inicial.csv')
    salvar_rotas_csv(nn_refinado, 'rotas_nn_refinado_com_bl.csv')

def ils(units_df, emergencies_df,
        construcao_func,
        busca_local_func,
        perturbar_func,
        max_iter=30):
    
    from copy import deepcopy

    solucao_inicial = construcao_func(units_df, emergencies_df)
    solucao_local = busca_local_func(solucao_inicial, units_df, emergencies_df)
    melhor_solucao = deepcopy(solucao_local)
    melhor_custo = calcular_distancia_total(units_df, emergencies_df, melhor_solucao)

    historico = [melhor_custo]

    for i in range(max_iter):
        solucao_perturbada = perturbar_func(melhor_solucao)
        solucao_aprimorada = busca_local_func(solucao_perturbada, units_df, emergencies_df)
        custo_aprimorado = calcular_distancia_total(units_df, emergencies_df, solucao_aprimorada)

        if custo_aprimorado < melhor_custo:
            melhor_solucao = deepcopy(solucao_aprimorada)
            melhor_custo = custo_aprimorado
            print(f"[{i+1}] ✅ Melhorou: {melhor_custo:.2f}")
        else:
            print(f"[{i+1}] 🔁 Sem melhora")

        historico.append(melhor_custo)

    return melhor_solucao, historico  # ✅ Corrigido aqui!



# FUNÇÕES DE ENTRADA DE DADOS E EXECUÇÃO PRINCIPAL

def salvar_rotas_csv(routes, caminho='rotas_resultado.csv', elapsed=None):
    #Salva as rotas em um arquivo CSV
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write("unidade_id,emergencias_atendidas\n")
        for unit, ems in routes.items():
            # Formata a lista de emergências como uma string separada por ponto e vírgula
            ems_str = ";".join(map(str, ems))
            if elapsed is not None:
                f.write(f'"{unit}","{ems_str}",{elapsed:.6f}\n')
            else:
                f.write(f'"{unit}","{ems_str}"\n')
    print(f"\nRotas salvas em formato CSV: {caminho}")

def input_units():
    n = int(input("Quantas unidades? "))
    units = []
    for i in range(1, n+1):
        uid = input(f"  ID da unidade #{i}: ")
        x   = float(input(f"  Coordenada X da unidade {uid}: "))
        y   = float(input(f"  Coordenada Y da unidade {uid}: "))
        units.append({'unit_id': uid, 'x': x, 'y': y})
    return pd.DataFrame(units)

def input_emergencies():
    m = int(input("Quantas emergências? "))
    ems = []
    for j in range(1, m+1):
        eid = input(f"  ID da emergência #{j}: ")
        x   = float(input(f"  Coordenada X da emergência {eid}: "))
        y   = float(input(f"  Coordenada Y da emergência {eid}: "))
        ems.append({'emergency_id': eid, 'x': x, 'y': y})
    return pd.DataFrame(ems)

if __name__ == "__main__":
    # Pergunta sobre o modo de entrada dos dados
    modo = input("Deseja ler de CSV (c) ou digitar interativamente (i)? [c/i]: ").strip().lower()
    
    if modo == 'c':
        try:
            # Tenta carregar os arquivos CSV fornecidos
            units_df = pd.read_csv('unidades.csv')
            emergencies_df = pd.read_csv('emergencias.csv')
            
            # Converte colunas para numérico, tratando erros
            units_df['x'] = pd.to_numeric(units_df['x'], errors='coerce')
            units_df['y'] = pd.to_numeric(units_df['y'], errors='coerce')
            emergencies_df['x'] = pd.to_numeric(emergencies_df['x'], errors='coerce')
            emergencies_df['y'] = pd.to_numeric(emergencies_df['y'], errors='coerce')

            # Remove linhas com coordenadas inválidas
            units_df = units_df.dropna(subset=['x', 'y'])
            emergencies_df = emergencies_df.dropna(subset=['x', 'y'])
        except FileNotFoundError:
            print("Erro: Arquivos 'units_top10.csv' ou 'Emergencies_CSV_FormatRight.csv' não encontrados.")
            exit()
    else:
        units_df = input_units()
        emergencies_df = input_emergencies()

    # Pergunta sobre o método a ser executado
    print("\nEscolha o método:")
    print("  1 - greedy            -> Heurística Gulosa")
    print("  2 - greedy_bl         -> Gulosa + Busca Local")
    print("  3 - greedy_e_greedy_bl  -> Comparar Gulosa e Gulosa + BL")
    print("  4 - grasp             -> Construção do GRASP")
    print("  5 - grasp_bl          -> GRASP")
    print("  6 - grasp_e_grasp_bl  -> Comparar GRASP e GRASP + BL")
    print("  7 - nn                -> Nearest Neighbor")
    print("  8 - nn_bl             -> Nearest Neighbor + Busca Local")
    print("  9 - nn_e_nn_bl        -> Comparar Nearest Neighbor e Nearest Neighbor + BL")
    print("  10 - ils_greedy       -> ILS com Gulosa")
    print("  11 - ils_grasp        -> ILS com GRASP")
    print("  12 - ils_nn           -> ILS com Nearest Neighbor")
    print("  13 - todos            -> Comparar todas heurísticas")

    opcao = input("Digite o número da opção desejada: ").strip()

    opcoes_map = {
				"1": "greedy",
				"2": "greedy_bl",
                "3": "greedy_e_greedy_bl",
				"4": "grasp",
				"5": "grasp_bl",
				"6": "grasp_e_grasp_bl",
				"7": "nn",
                "8": "nn_bl",
                "9": "nn_e_nn_bl",
                "10": "ils_greedy",
                "11": "ils_grasp",
				"12": "ils_nn",
                "13": "todos",
                }

    metodo = opcoes_map.get(opcao, "greedy")  # Se digitar errado, vai usar 'greedy'
    print(f"\nMétodo selecionado: {metodo}")

    routes = None
    elapsed = None
    method_label = None
    
    if metodo == 'greedy':
        method_label = "Gulosa"
        t0 = time.perf_counter()
        routes = construct_initial_solution(units_df, emergencies_df)
        elapsed = time.perf_counter() - t0

    elif metodo == 'grasp':
        method_label = "GRASP"
        t0 = time.perf_counter()
        routes = construct_grasp_solution(units_df, emergencies_df, alpha=0.3)
        elapsed = time.perf_counter() - t0
        
    elif metodo == 'nn':
        method_label = "Nearest Neighbor"
        t0 = time.perf_counter()
        routes = construct_nearest_neighbor_solution(units_df, emergencies_df)
        elapsed = time.perf_counter() - t0

    elif metodo == 'grasp_bl':
        method_label = "Grasp + Busca Local"
        max_iter = int(input("  Número de iterações para o GRASP: "))
        alpha_val = float(input("  Valor de alpha para o GRASP (ex: 0.3): "))
        t0 = time.perf_counter()
        routes = grasp_com_busca_local(units_df, emergencies_df, max_iterations=max_iter, alpha=alpha_val)
        elapsed = time.perf_counter() - t0

    elif metodo == 'greedy_bl':
        method_label = "Greedy + Busca Local"
        t0 = time.perf_counter()
        routes = gulosa_com_busca_local(units_df, emergencies_df)
        elapsed = time.perf_counter() - t0

    elif metodo == 'grasp_e_grasp_bl':
        method_label = "GRASP X GRASP + BL"
        t0 = time.perf_counter()
        routes = comparar_solucoes_grasp(units_df, emergencies_df)
        elapsed = time.perf_counter() - t0
    elif metodo == 'greedy_e_greedy_bl':
        method_label = "Gulosa X Gulosa + BL"
        t0 = time.perf_counter()
        comparar_solucoes_greedy(units_df, emergencies_df)
        elapsed = time.perf_counter() - t0
    elif metodo == 'nn_bl':
        method_label = "Nearest Neighbor + Busca Local"
        t0 = time.perf_counter()
        routes = nn_com_busca_local(units_df, emergencies_df)
        elapsed = time.perf_counter() - t0
    elif metodo == 'nn_e_nn_bl':
        method_label = "Nearest Neighbor X Nearest Neighbor + BL"
        t0 = time.perf_counter()
        comparar_solucoes_nn(units_df, emergencies_df)
        elapsed = time.perf_counter() - t0
    elif metodo == 'todos':
        comparar_solucoes(units_df, emergencies_df)
    elif metodo == "ils_grasp":
        method_label = "ILS (com GRASP)"
        t0 = time.perf_counter()
        routes, _ = ils(
            units_df, emergencies_df,
            construcao_func=lambda u, e: construct_grasp_solution(u, e, alpha=0.3),
            busca_local_func=busca_local,
            perturbar_func=perturbar_solucao,
            max_iter=30
        )
        elapsed = time.perf_counter() - t0

    elif metodo == "ils_greedy":
        method_label = "ILS (com Greedy)"
        t0 = time.perf_counter()
        routes, _ = ils(
            units_df, emergencies_df,
            construcao_func=construct_initial_solution,
            busca_local_func=busca_local,
            perturbar_func=perturbar_solucao,
            max_iter=30
        )
        elapsed = time.perf_counter() - t0
    elif metodo == "ils_nn":
        method_label = "ILS (com Nearest Neighbor)"
        t0 = time.perf_counter()
        routes, _ = ils(
            units_df, emergencies_df,
            construcao_func=construct_nearest_neighbor_solution,
            busca_local_func=busca_local,
            perturbar_func=perturbar_solucao,
            max_iter=30
        )
        elapsed = time.perf_counter() - t0
    else:
        print("Método inválido. Usando 'greedy' por padrão.")
        routes = construct_initial_solution(units_df, emergencies_df)

    # Se um método que retorna uma única rota foi escolhido, exibe os resultados

    if routes is not None:
        print("\nRotas atribuídas:")
        for unit, route in routes.items():
            print(f"  Unidade {unit}: emergências {route}")
        
        salvar_rotas_csv(routes, f'rotas_{metodo}.csv', elapsed=elapsed)
        
        print(f"[Timing] {metodo} levou {elapsed:.6f}s")
        total_dist = calcular_distancia_total(units_df, emergencies_df, routes)
        print(f"\nDistância total ({method_label}): {total_dist:.2f}")

        # chama o plot passando o tempo e o rótulo
        plot_rotas(
            units_df,
            emergencies_df,
            routes,
            total_dist,
            method_name=method_label,
            elapsed=elapsed
        )

    print("\nProcesso concluído.")