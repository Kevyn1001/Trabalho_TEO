## Trabalho Tópicos Especiais em Otimização 

# Otimização de Atendimento a Emergências com Heurísticas

Este projeto implementa e compara diferentes heurísticas para o problema de roteamento de unidades de atendimento a emergências para ocorrências. O objetivo é minimizar a distância total percorrida pelas unidades ao atender uma série de emergências.

## Funcionalidades

O sistema permite:
* Carregar dados de unidades e emergências via arquivos CSV ou inserção manual.
* Aplicar e comparar quatro heurísticas distintas para atribuir emergências a unidades e definir suas rotas:
    * **Greedy**: Atribui cada emergência à unidade mais próxima no momento.
    * **Nearest Neighbor (NN)**: Cada unidade atende a emergência mais próxima sequencialmente até que todas as emergências sejam atribuídas.
    * **GRASP (Greedy Randomized Adaptive Search Procedure)**: Uma meta-heurística que combina uma fase construtiva gulosa-randomizada com uma busca local.
    * **GRASP com Busca Local (2-opt)**: Uma versão aprimorada do GRASP que aplica uma Busca Local (o algoritmo 2-opt) a cada rota gerada para otimizá-las ainda mais.
* Calcular a distância total percorrida por todas as unidades para uma dada solução.
* Visualizar graficamente as rotas atribuídas, com cada unidade e suas emergências em cores distintas.
* Gerar arquivos CSV com os detalhes das rotas encontradas por cada heurística.

## Heurísticas Implementadas

### 1. Greedy
Uma abordagem simples onde, a cada passo, a emergência ainda não atendida é atribuída à unidade que estiver mais próxima dela no momento. A unidade, após atender a emergência, tem sua "posição atual" atualizada para a localização da emergência recém-atendida.

### 2. Nearest Neighbor (NN)
Similar ao Greedy, mas a atribuição é feita focando em cada unidade individualmente. Cada unidade busca a próxima emergência mais próxima para atender, e esse processo se repete até que todas as emergências sejam atribuídas.

### 3. GRASP (Greedy Randomized Adaptive Search Procedure)
O GRASP opera em duas fases principais:
* **Fase de Construção**: De forma iterativa, constrói uma solução inicial viável. Para cada atribuição, é criada uma Lista Restrita de Candidatos (RCL) com base na proximidade. Um candidato é escolhido aleatoriamente dessa RCL, introduzindo um elemento de aleatoriedade que permite explorar diferentes "bons" caminhos. O parâmetro `alpha` controla o grau de aleatoriedade na formação da RCL.
* **Fase de Busca Local**: (Opcional, mas presente na versão `grasp_bl`) Melhora a solução construída. A solução é explorada em sua vizinhança para encontrar uma solução melhor.

### 4. GRASP com Busca Local (2-opt)
Esta versão estende o GRASP tradicional ao incorporar uma etapa de Busca Local usando o algoritmo 2-opt. Após a fase de construção gulosa-randomizada do GRASP, cada rota individual de cada unidade é submetida a um processo de otimização 2-opt.

#### Algoritmo 2-opt
O 2-opt é uma heurística de busca local comumente utilizada para problemas de roteamento (como o Problema do Caixeiro Viajante - TSP). Ele funciona da seguinte maneira:
1.  **Iteração**: O algoritmo itera até que nenhuma melhoria significativa possa ser feita.
2.  **Troca**: Em cada iteração, ele considera a remoção de duas arestas não adjacentes de uma rota.
3.  **Reconexão**: As duas arestas são então reconectadas de uma forma diferente, invertendo o segmento de rota entre os dois pontos de corte.
4.  **Avaliação**: Se a nova rota resultante tiver uma distância total menor que a rota anterior, ela é aceita, e o processo de busca recomeça a partir desta nova rota melhorada. Caso contrário, a alteração é descartada.

## Como Executar

### Pré-requisitos
Certifique-se de ter as seguintes bibliotecas Python instaladas:
* `pandas`
* `numpy`
* `matplotlib`

Você pode instalá-las usando pip:
```bash
pip install pandas numpy matplotlib
```

## Estrutura de Arquivos
Certifique-se de que os arquivos CSV de entrada estejam na mesma pasta do script Python:
* `Units_CSV_Format.csv`: Contém os dados das unidades (`unit_id`, `x`, `y`).
* `Emergencies_CSV_FormatRight.csv`: Contém os dados das emergências (`emergency_id`, `x`, `y`).

## Execução do Script
Para rodar o programa, execute o script Python no seu terminal:
```bash
python seu_script_principal.py
```

O programa solicitará as seguintes entradas:

1.  **Modo de Entrada**:
    * `c` para ler dados de arquivos CSV.
    * `i` para inserir dados manualmente via terminal.
    * Exemplo: `Deseja ler de CSV (c) ou digitar interativamente (i)? [c/i]: c`

2.  **Método de Otimização**:
    * `greedy`: Executa a heurística gulosa.
    * `grasp`: Executa o GRASP sem busca local.
    * `grasp_bl`: Executa o GRASP com busca local (2-opt).
    * `nn`: Executa a heurística Nearest Neighbor.
    * `todos`: Executa e compara todas as heurísticas, gerando um gráfico comparativo e salvando CSVs para cada solução.
    * Exemplo: `Escolha o método (greedy/grasp/grasp_bl/nn/todos): grasp_bl`

## Saídas

* **No Terminal**: As rotas atribuídas para cada unidade e a distância total percorrida pela solução escolhida.
* **Arquivos CSV**:
    * `rotas_resultado.csv`: Se uma única heurística for escolhida, este arquivo conterá as rotas daquela solução.
    * `rotas_greedy.csv`, `rotas_grasp.csv`, `rotas_grasp_bl.csv`, `rotas_nn.csv`: Se o método `todos` for escolhido, um arquivo CSV será gerado para cada heurística com suas respectivas rotas.
* **Imagens**:
    * `rotas_plot.png`: Um gráfico visualizando as rotas da solução escolhida (se um único método for selecionado).
    * `comparacao_solucoes.png`: Um gráfico comparativo mostrando as rotas e distâncias de todas as heurísticas (se o método `todos` for selecionado).

## Dados de Exemplo (Conteúdo dos CSVs)

### `Units_CSV_Format.csv`
```bash
unit_id,x,y
U1,10,10
U2,90,10
U3,50,90
```

### `Emergencies_CSV_FormatRight.csv`
```bash
emergency_id,x,y
E1,20,20
E2,80,20
E3,60,80
E4,30,70
E5,70,30
E6,40,40
E7,50,50
```