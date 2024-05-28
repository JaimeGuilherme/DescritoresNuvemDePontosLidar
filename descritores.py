import numpy as np
import pylas
from scipy.spatial import cKDTree
import time

#Identifica se o arquivo .las contém dados RGB
def has_rgb_data(file_path):
    # Carrega o arquivo LAS
    las = pylas.read(file_path)
    
    # Obtém as dimensões do formato de ponto
    point_format_dimensions = las.point_format.dimension_names
    
    # Verifica se o arquivo tem os campos de cor (Red, Green, Blue)
    return 'red' in point_format_dimensions and 'green' in point_format_dimensions and 'blue' in point_format_dimensions

#Calcula a matriz de covariância para os dados geométricos
def compute_covariance_matrix(points):
    radius=1
    covariance_matrices = []
    num_points = points.shape[0]
    tree = cKDTree(points[:, :3])

    for i in range(num_points):
        # Extrai o ponto atual e suas coordenadas
        point_i = points[i]
        x_i, y_i, z_i = point_i[:3]

        # Extrai os pontos da vizinhança dentro do raio
        indices = tree.query_ball_point([x_i, y_i, z_i], radius)
        neighborhood_points = points[indices]

        if len(neighborhood_points) < 2:
            # Se não há pontos suficientes para calcular a matriz de covariância, use a matriz identidade
            covariance_matrix = np.identity(3)
        else:
            # Subtrai as coordenadas do ponto central das dos pontos da vizinhança
            centered_points = neighborhood_points[:, :3] - point_i[:3]

            # Computa a matriz de covariancia
            covariance_matrix = np.cov(centered_points, rowvar=False)

        # Adiciona a matriz de covariância na lista de matrizes
        covariance_matrices.append(covariance_matrix)

    return covariance_matrices

#Calcula os descritores geométricos dos pontos
def compute_geometric_features(points):
    num_points = points.shape[0]
    features = np.zeros((num_points, 7))
    covariance_matrices = compute_covariance_matrix(points)

    for i, covariance_matrix in enumerate(covariance_matrices):
        if covariance_matrix.size == 0:
            # Caso especial: matriz de covariância vazia
            features[i] = np.nan  # ou qualquer valor adequado
            continue
        # Dispõe os autovalores em ordem crescente
        eigenvalues_i = np.linalg.eigvalsh(covariance_matrix)
        sum_eigenvalues_i = np.sum(eigenvalues_i)

        if sum_eigenvalues_i == 0 or np.any(eigenvalues_i <= 0):
            # Evita valores zero ou negativos nos autovalores
            features[i] = np.nan  # ou qualquer valor adequado
            continue

        # Normalização dos autovalores
        normalized_eigenvalues_i = eigenvalues_i / sum_eigenvalues_i

        linearidade_i = 1 - (eigenvalues_i[1] / eigenvalues_i[2])
        planaridade_i = (eigenvalues_i[1] - eigenvalues_i[0]) / eigenvalues_i[2]
        esfericidade_i = eigenvalues_i[0] / eigenvalues_i[2]
        curvatura_i = eigenvalues_i[0] / sum_eigenvalues_i
        entropia_i = -np.sum((eigenvalues_i / sum_eigenvalues_i) * np.log(eigenvalues_i / sum_eigenvalues_i))
        omnivariância_i = np.cbrt(np.prod(normalized_eigenvalues_i))
        anisotropia_i = 1 - (eigenvalues_i[0] / eigenvalues_i[2])
        features[i] = [linearidade_i, planaridade_i, esfericidade_i, curvatura_i, entropia_i, omnivariância_i, anisotropia_i]

    return features

#Calcula os atributos espectrais dos pontos
def compute_spectral_features(points):
    num_points = points.shape[0]
    spectral_features = np.zeros((num_points, 4))
    
    for i in range(num_points):
        r, g, b = points[i, 3:6]
        spectral_features[i, 0] = r
        spectral_features[i, 1] = g
        spectral_features[i, 2] = b
        spectral_features[i, 3] = (r + g + b) / 3.0  # Mean

    return spectral_features

#Função para classificação dos pontos em função dos parâmetros fornecidos
def classify_points(points, method):
    num_points = points.shape[0]
    classes = []
    # Classificação diferenciada se tem ou não dados RGB
    if(method):
        geometric_features = compute_geometric_features(points)
        spectral_features = compute_spectral_features(points)
        for i in range(num_points):
            # Descritores geométricos
            linearity, planarity, sphericity, curvature, entropy, omnivariance, anisotropy = geometric_features[i]
            # Descritores espectrais
            mean_r, mean_g, mean_b, mean_intensity = spectral_features[i]
            # Condições de classificação usando apenas descritores espectrais
            if mean_intensity > 30000 or (planarity > 0.7 and omnivariance < 0.3):
                classe = 1  # Edificação (exemplo de intensidade alta)
            elif (mean_g > 32767 and mean_b < 32767 and mean_r < 32767) or entropy > 0.70:
                classe = 2  # Vegetação (exemplo espectral típico)
            else:
                classe = 0  # Pontos Específicos
            classes.append(classe)
    else:
        geometric_features = compute_geometric_features(points)
        for i in range(num_points):
            # Descritores geométricos
            linearity, planarity, sphericity, curvature, entropy, omnivariance, anisotropy = geometric_features[i]
            if planarity > 0.7 and omnivariance < 0.3:
                classe = 1  # Edificação (exemplo de altaplanaridade e baixa omnivariance)
            elif entropy > 0.70:
                classe = 2  # Vegetação (exemplo de entropia alta)
            else:
                classe = 0  # Pontos Específicos
            classes.append(classe)
    return classes

#Segmentação dos pontos pelos atributos coletados
def write_classified_points(filename, points, classes):
    colors = {
        0: [128, 128, 128],  # Pontos Específicos
        1: [255, 100, 0],    # Edificação (laranja)
        2: [0, 255, 0],      # Vegetação (verde)
    }

    # Criação do campo de cor RGB
    color_array = np.zeros((len(points), 3), dtype=np.uint16)

    for i, _ in enumerate(points):
        color_array[i] = colors[classes[i]]

    # Crie um objeto LasData com o formato de ponto que suporta cores (formato 2)
    outFile = pylas.create(point_format_id=2)

    # Defina as cores RGB
    outFile.red = color_array[:, 0]
    outFile.green = color_array[:, 1]
    outFile.blue = color_array[:, 2]

    # Escreva as coordenadas X, Y e Z
    outFile.x = points[:, 0]
    outFile.y = points[:, 1]
    outFile.z = points[:, 2]

    # Defina a intensidade como 0 (pode ser ajustada conforme necessário)
    outFile.intensity = np.zeros(len(points), dtype=np.uint16)

    # Defina a classe conforme especificado
    outFile.classification = np.array(classes, dtype=np.uint8)

    # Escreva os dados para o arquivo LAS
    outFile.write(filename)

def main():
    start_time = time.time()
    file_path = "C:/CaminhoDoArquivoDaNuvemDePontos"
    inFile = pylas.read(file_path)
    method = has_rgb_data(file_path)
    if(method):
        points = np.vstack((inFile.x, inFile.y, inFile.z, inFile.red, inFile.green, inFile.blue)).T
    else:
        points = np.vstack((inFile.x, inFile.y, inFile.z)).T
    classes = classify_points(points, method)
    write_classified_points("C:/CaminhoOndeANuvemClassificadaSeráSalva", points, classes)
    end_time = time.time()
    print("Tempo de execução:", end_time - start_time, "segundos")

if __name__ == "__main__":
    main()
