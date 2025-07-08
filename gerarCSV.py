#!/usr/bin/env python3
import csv
import random
import sys

def generate_points(n, prefix, header_name, filename):
    """
    Gera n pontos com IDs prefixados (por exemplo, 'E1', 'E2', ...) em um quadrado
    de lado proporcional a n, garantindo que à medida que n cresce, a dispersão cresce.
    Salva em um arquivo CSV com colunas [header_name, x, y].
    """
    max_coord = n * 10  # quanto maior n, maior o alcance das coordenadas
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([header_name, 'x', 'y'])
        for i in range(1, n + 1):
            x = random.uniform(-max_coord, max_coord)
            y = random.uniform(-max_coord, max_coord)
            writer.writerow([f"{prefix}{i}", x, y])
    print(f"Gerado {filename}")

def main():
    if len(sys.argv) != 3:
        print("Uso: python gerarCSV.py N_emergencias N_unidades")
        sys.exit(1)

    n_emergencias = int(sys.argv[1])
    n_unidades    = int(sys.argv[2])

    # Emergências: IDs E1, E2, ..., EN
    generate_points(
        n=n_emergencias,
        prefix='E',
        header_name='emergency_id',
        #filename=f'emergencias{n_emergencias}.csv'
        filename=f'emergencias.csv'
    )

    # Unidades (ambulâncias): IDs U1, U2, ..., UN
    generate_points(
        n=n_unidades,
        prefix='U',
        header_name='unit_id',
        #filename=f'unidades{n_unidades}.csv'
        filename=f'unidades.csv'
    )

if __name__ == "__main__":
    main()
