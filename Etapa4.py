import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Plota as rotas atribuídas graficamente
# Cada rota de unidade é desenhada com uma cor distinta

def plot_rotas(units_df, emergencies_df, routes):
    plt.figure(figsize=(10, 8))
    cores = plt.cm.get_cmap('tab10', len(units_df))

    for i, (unit_id, em_ids) in enumerate(routes.items()):
        # Posição inicial da unidade
        pos_atual = units_df.loc[units_df['unit_id'] == unit_id, ['x', 'y']].values
        if len(pos_atual) == 0:
            continue
        x_unit, y_unit = pos_atual[0]
        rota_x = [x_unit]
        rota_y = [y_unit]

        # Adiciona posições das emergências
        for em_id in em_ids:
            em = emergencies_df.loc[emergencies_df['emergency_id'] == em_id, ['x', 'y']]
            if not em.empty:
                rota_x.append(em.iloc[0]['x'])
                rota_y.append(em.iloc[0]['y'])

        plt.plot(rota_x, rota_y, marker='o', color=cores(i), label=f'Unidade {unit_id}')
        plt.scatter(x_unit, y_unit, color=cores(i), marker='s', s=100, edgecolors='black')

    # Mostra todas as emergências
    plt.scatter(emergencies_df['x'], emergencies_df['y'], color='lightgray', marker='x', label='Emergências')
    plt.title('Rotas atribuídas às unidades')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('rotas_plot.png')
    plt.show()

# Gera e compara soluções usando diferentes heurísticas

def comparar_solucoes(units_df, emergencies_df):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    def plot_subplot(ax, title, routes):
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

    # Gera solução Greedy
    greedy = construct_initial_solution(units_df, emergencies_df)
    dist_greedy = calcular_distancia_total(units_df, emergencies_df, greedy)
    plot_subplot(axs[0, 0], f"Greedy ({dist_greedy:.2f})", greedy)

    # Gera solução GRASP com alpha = 0.3
    grasp = construct_grasp_solution(units_df, emergencies_df, alpha=0.3)
    dist_grasp = calcular_distancia_total(units_df, emergencies_df, grasp)
    plot_subplot(axs[0, 1], f"GRASP ({dist_grasp:.2f})", grasp)

    # Gera solução Nearest Neighbor por unidade
    nn = construct_nearest_neighbor_solution(units_df, emergencies_df)
    dist_nn = calcular_distancia_total(units_df, emergencies_df, nn)
    plot_subplot(axs[1, 0], f"Nearest Neighbor ({dist_nn:.2f})", nn)

    axs[1, 1].axis('off')
    plt.suptitle("Comparação de Heurísticas", fontsize=16)
    plt.savefig("comparacao_solucoes.png")
    plt.show()

    print(f"\nDistâncias totais:")
    print(f"  Greedy: {dist_greedy:.2f}")
    print(f"  GRASP:  {dist_grasp:.2f}")
    print(f"  NN:     {dist_nn:.2f}")

# Heurística gulosa: atribui cada emergência à unidade mais próxima no momento

def construct_initial_solution(units_df, emergencies_df):
    pending = emergencies_df.copy()
    routes = {unit: [] for unit in units_df['unit_id']}
    current_positions = {row['unit_id']: (row['x'], row['y']) for _, row in units_df.iterrows()}

    # Enquanto houver emergências pendentes
    while not pending.empty:
        best_dist = float('inf')
        best_unit, best_idx = None, None

        # Procura a melhor combinação unidade + emergência (menor distância)
        for unit_id, pos in current_positions.items():
            dx = pending['x'] - pos[0]
            dy = pending['y'] - pos[1]
            dists = np.hypot(dx, dy)

            # Ignora distâncias NaN (caso não haja emergências pendentes)
            if dists.isnull().all():
                continue

            # Encontra a emergência mais próxima para a unidade atual
            idx_min = dists.idxmin()
            dist_min = dists.loc[idx_min]

            # Atualiza se for a melhor distância encontrada até agora
            if dist_min < best_dist:
                best_dist = dist_min
                best_unit = unit_id
                best_idx = idx_min

        # Se não houver mais emergências pendentes, encerra o loop
        if best_idx is None:
            break

        # Atribui a emergência escolhida à unidade selecionada
        chosen = pending.loc[best_idx]

        # Adiciona a emergência à rota da unidade
        routes[best_unit].append(chosen['emergency_id'])

        # Atualiza a posição atual da unidade
        current_positions[best_unit] = (chosen['x'], chosen['y'])

        # Remove a emergência da lista pendente
        pending = pending.drop(best_idx)

    return routes

# Heurística GRASP: utiliza lista restrita de candidatos baseada em alpha

def construct_grasp_solution(units_df, emergencies_df, alpha=0.3):
    pending = emergencies_df.copy()
    routes = {unit: [] for unit in units_df['unit_id']}
    current_positions = {row['unit_id']: (row['x'], row['y']) for _, row in units_df.iterrows()}

    # Enquanto houver emergências pendentes
    while not pending.empty:
        candidatos = []
        # Calcula todas as distâncias possíveis (unidade x emergência)
        for unit_id, pos in current_positions.items():
            dx = pending['x'] - pos[0]
            dy = pending['y'] - pos[1]
            dists = np.hypot(dx, dy)
            for idx, dist in dists.items():
                if not np.isnan(dist):
                    candidatos.append((unit_id, idx, dist))

        # Se não houver candidatos, encerra o loop
        if not candidatos:
            break

        # Ordena candidatos pela distância e seleciona um aleatório da lista restrita
        candidatos.sort(key=lambda x: x[2])

        # Define o tamanho da lista restrita (RCL) baseado em alpha
        limite = int(len(candidatos) * alpha) + 1
        rcl = candidatos[:limite]
        chosen_unit, chosen_idx, _ = random.choice(rcl)

        # Atribui a emergência escolhida à unidade selecionada
        chosen = pending.loc[chosen_idx]
        routes[chosen_unit].append(chosen['emergency_id'])
        current_positions[chosen_unit] = (chosen['x'], chosen['y'])
        pending = pending.drop(chosen_idx)

    return routes

# Heurística Nearest Neighbor para cada unidade até esgotar emergências

def construct_nearest_neighbor_solution(units_df, emergencies_df):
    pending = emergencies_df.copy()
    routes = {unit: [] for unit in units_df['unit_id']}
    current_positions = {row['unit_id']: (row['x'], row['y']) for _, row in units_df.iterrows()}

    while not pending.empty:
        for unit_id in current_positions:
            if pending.empty:
                break

            pos = current_positions[unit_id]
            dx = pending['x'] - pos[0]
            dy = pending['y'] - pos[1]
            dists = np.hypot(dx, dy)
            if dists.isnull().all():
                continue

            idx_min = dists.idxmin()
            chosen = pending.loc[idx_min]
            routes[unit_id].append(chosen['emergency_id'])
            current_positions[unit_id] = (chosen['x'], chosen['y'])
            pending = pending.drop(idx_min)

    return routes

# Calcula a distância total percorrida por todas unidades

def calcular_distancia_total(units_df, emergencies_df, routes):
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

# Exporta as rotas para um CSV

def salvar_rotas_csv(routes, caminho='rotas_resultado.csv'):
    """
    Salva as rotas no mesmo formato exibido no terminal, uma linha por unidade.
    Exemplo: Unidade U1: emergências ['E1', 'E3']
    """
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write("Rotas atribuídas:\n")
        for unit, ems in routes.items():
            f.write(f"Unidade {unit}: emergências {ems}\n")
    print(f"\nRotas (formato de log) salvas em: {caminho}")

# Entrada manual de unidades

def input_units():
    n = int(input("Quantas unidades? "))
    units = []
    for i in range(1, n+1):
        uid = input(f"  ID da unidade #{i}: ")
        x   = float(input(f"  Coordenada X da unidade {uid}: "))
        y   = float(input(f"  Coordenada Y da unidade {uid}: "))
        units.append({'unit_id': uid, 'x': x, 'y': y})
    return pd.DataFrame(units)

# Entrada manual de emergências

def input_emergencies():
    m = int(input("Quantas emergências? "))
    ems = []
    for j in range(1, m+1):
        eid = input(f"  ID da emergência #{j}: ")
        x   = float(input(f"  Coordenada X da emergência {eid}: "))
        y   = float(input(f"  Coordenada Y da emergência {eid}: "))
        ems.append({'emergency_id': eid, 'x': x, 'y': y})
    return pd.DataFrame(ems)

# Execução principal

if __name__ == "__main__":
    modo = input("Deseja ler de CSV (c) ou digitar interativamente (i)? [c/i]: ").strip().lower()
    metodo = input("Escolha o método (greedy/grasp/nn/todos): ").strip().lower()

    if modo == 'c':
        units_df = pd.read_csv('units_top10.csv')
        #units_df = pd.read_csv('Units_CSV_Format.csv')
        emergencies_df = pd.read_csv('Emergencies_CSV_FormatRight.csv')
        units_df['x'] = pd.to_numeric(units_df['x'], errors='coerce')
        units_df['y'] = pd.to_numeric(units_df['y'], errors='coerce')
        emergencies_df['x'] = pd.to_numeric(emergencies_df['x'], errors='coerce')
        emergencies_df['y'] = pd.to_numeric(emergencies_df['y'], errors='coerce')
        units_df = units_df.dropna(subset=['x', 'y'])
        emergencies_df = emergencies_df.dropna(subset=['x', 'y'])
    else:
        units_df = input_units()
        emergencies_df = input_emergencies()

    if metodo == 'greedy':
        routes = construct_initial_solution(units_df, emergencies_df)
    elif metodo == 'grasp':
        routes = construct_grasp_solution(units_df, emergencies_df)
    elif metodo == 'nn':
        routes = construct_nearest_neighbor_solution(units_df, emergencies_df)
    elif metodo == 'todos':
        comparar_solucoes(units_df, emergencies_df)
    
        # Salva CSVs individuais de cada solução
        salvar_rotas_csv(construct_initial_solution(units_df, emergencies_df), 'rotas_greedy.csv')
        salvar_rotas_csv(construct_grasp_solution(units_df, emergencies_df), 'rotas_grasp.csv')
        salvar_rotas_csv(construct_nearest_neighbor_solution(units_df, emergencies_df), 'rotas_nn.csv')
        
        routes = None  # Não plota novamente fora da função
    else:
        print("Método inválido. Usando 'greedy' por padrão.")
        routes = construct_initial_solution(units_df, emergencies_df)

    if routes:
        print("\nRotas atribuídas:")
        for unit, route in routes.items():
            print(f"  Unidade {unit}: emergências {route}")
        salvar_rotas_csv(routes)
        total_distancia = calcular_distancia_total(units_df, emergencies_df, routes)
        print(f"\nDistância total percorrida ({metodo}): {total_distancia:.2f}")
        plot_rotas(units_df, emergencies_df, routes)

    print("\nProcesso concluído.")
