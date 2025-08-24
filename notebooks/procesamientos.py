import mlflow
from datetime import datetime
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier, BaggingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.preprocessing import StandardScaler
from feast import (
    Entity,
    FeatureStore,
    FeatureService,
    FeatureView,
    Field,
    FileSource
)
from loguru








# to_train_stresslevel = FeatureProcessor(datos = train_stresslevel, name_pipeline = 'stresslevel_pipeline')
# to_train_stresslevel.run(columnas_promedio = ('','','',''))


class FeatureProcessor:
    def __init__(self, datos: pd.DataFrame, name_pipeline: str,  columnas: tuple[str, ...], components: int):
        
        self.datos = datos
        self.name_pipeline = name_pipeline
        self.feature_table = None
        self.columnas = columnas
        self.components = components
        self.variance_ratio = None
        
    def scale(self) -> pd.DataFrame:
        
        pca = PCA(n_components = self.components)
        pipe = Pipeline(steps=[("std_scaling", StandardScaler()),("pca", pca)])
        
        self.pca = pca
        self.pipe = pipe
        
        X = self.datos.loc[:, list(self.columnas)]
        Z = pipe.fit_transform(X)
        
        self.variance_ratio = pca.explained_variance_ratio_
        print(f"Variance ratio: {self.variance_ratio}")
        
        #agregar aqui y devolver como variance_ratio en el return
        
        pipe_df = pd.DataFrame(Z, columns=[f"Pipe_feature{i+1}" for i in range(self.components)])
        
        return pipe_df, self.variance_ratio


#Una ves que se creen los pca_features agregarse al dataset final


    def run(self, columnas_promedio: tuple[str, ...]) -> pd.DataFrame:
        
        #tengo un problema con el pyproject vim/nano
        #logger.info(f"Inicializando pipeline {self.name_pipeline}")
        #numerics = self.scale()
        
        pipe_df, variance_ratio = self.scale(columnas = columnas_promedio)
        media_stress = (self.datos[list(columnas_promedio)].mean(axis=1))
        media_df = pd.DataFrame({"stress_exposure_mean": media_stress})
        modeling_dataset = pd.concat([pipe_df, media_df], axis=1)
        # Dataset Previo el pipeline
        pipe = Pipeline(
            steps=[
                ("feature_selection", VarianceThreshold()),
                ("scaling_robust", RobustScaler())
            ]
        )
        self.feature_table =  pd.DataFrame(
            pipe.fit_transform(modeling_dataset),
            columns=modeling_dataset.columns
        )
        return self.feature_table

    def write_feature_table(self, filepath: str) -> None:
        """Escribimos la feature table final para modelamiento
        """

        if not self.feature_table.empty: # -> True o False
            self.feature_table.to_parquet(f"{filepath}.parquet", index=False)
            self.feature_table.to_csv(f"{filepath}.csv", index=False)
        else:
            raise Exception("Ejecutar el comando .run()")            
        
        
        #ayuda aca 
class Metricsdeploy:
    def pcavarianza(self, variance_ratio):
        return print(f'La varianza que se explica despues del PCA:{variance_ratio}')

    def clasificacionmetrics():
        #Matriz de confusión, recall, f1 score y precisión
        return
    def clusteringmetrics(self, X_scaled, labels):
        silhouette = silhouette_score(X_scaled, labels)
        dbi = davies_bouldin_score(X_scaled, labels)
        chi = calinski_harabasz_score(X_scaled, labels)
        ari = adjusted_rand_score(y, labels) # compara clusters con etiquetas reales
        nmi = normalized_mutual_info_score(y, labels)
        
        Sil = print(f"Silhouette Score: {silhouette:.3f}")
        Dbouldin = print(f"Davies-Bouldin Index: {dbi:.3f}")
        Charab = print(f"Calinski-Harabasz Index: {chi:.3f}")
        ADI = print(f"Adjusted Rand Index {ari:.3f}")
        NM = print(f"Normalized Mutual Info: {nmi:.3f}")
        return Sil, Dbouldin, Charab, ADI, NM 
    
    
    

class GuardadoFeature:




    


class Procesoexperimento:
    def proceso_epsilon(n_muestreo: int , ):
        #Numero de muestreo = valor definido para ver donde esta el alza en la varianza
        neighbors = NearestNeighbors(n_neighbors=n_muestreo)
        neighbors_fit = neighbors.fit(X_scaled)
        distances, indices = neighbors_fit.kneighbors(X_scaled)
        
        # Ordenar las distancias al k-ésimo vecino
        distances = np.sort(distances[:, min_samples-1])
        plt.plot(distances)
        plt.ylabel("Distancia al {}-ésimo vecino".format(min_samples))
        plt.xlabel("Puntos ordenados")
        plt.show()
    
        kneighbors_eps = input('Valor aproximado para el epsilon en base al grafico:')
        dbscan = DBSCAN(eps=kneighbors_eps, min_samples=minimun_samples)
        labels = dbscan.fit_predict(X_scaled)

        return
        
    

#procesar con dbscan, 18 clusters(numero de columnas)'''

#deberia usar calses para llamar a las funciones y solo depender de ingresar los argumentos?, en ves de usar clase pense usar funciones y llamarlas directamente


def experiment_definition(X_train, X_test, y_train, y_test, model=None, input_value="mean"):
    if model is None:
        model = input(
            "Que modelo desea aplicar?\n"
            "(1) Random Forest\n"
            "(2) Bagging\n"
            "(3) Voting\n"
            "(4) XGBoost\n"
            "(5) LGBM\n"
            "(6) Catboost\n"
        )

    models = {
        "1": ("Random Forest", RandomForestClassifier()),
        "2": ("Bagging", BaggingClassifier()),
        "3": ("Voting", VotingClassifier(estimators=[
            ("rf", RandomForestClassifier()),
            ("bag", BaggingClassifier()),
        ], voting="soft")),
        "4": ("XGBoost", XGBClassifier(max_depth=5, n_estimators=100)),
        "5": ("LGBM", LGBMClassifier()),
        "6": ("CatBoost", CatBoostClassifier(verbose=0)),
    }

    if model not in models:
        print("Opción inválida.")

    run_name, algorithm = models[model]
    
    with mlflow.start_run(run_name=run_name):
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=input_value)),
            (run_name, algorithm),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        mlflow.log_params({"model": run_name, "imputer": input_value})
        mlflow.log_metrics({"accuracy": acc, "f1": f1})

        print(f"{run_name} - Accuracy: {acc:.4f} | F1: {f1:.4f}")

    return pipeline
    


























class UnsupervisedProcessor:

    def __init__(self, dataframe: pd.DataFrame, scaler, cluster_algorithm, dim_reduction_algorithm) -> None:
        self.original_data = dataframe
        self.cluster_pipeline = Pipeline(
            steps=[
                ("scaler", scaler),
                ("cluster_algorithm", cluster_algorithm)
            ]
        )
        self.dim_reduction_pipeline = Pipeline(
            steps=[
                ("scaler", scaler),
                ("dim_reduction", dim_reduction_algorithm)
            ]
        )

    def process_clustering(self, columns: list) -> pd.DataFrame:

        tmp_data_to_process = self.original_data[columns]
        self.cluster_pipeline.fit(tmp_data_to_process)
        clustering_df = pd.concat(
            [
                self.original_data[["CustomerID"]],
                self.original_data[self.original_data.columns[-3:]],
                pd.DataFrame(
                    self.cluster_pipeline.steps[1][1].labels_,
                    columns=["cluster"]
                )
            ],
            axis=1
        )

        return clustering_df

    def process_dim_reduction(self, columns: list) -> pd.DataFrame:

        tmp_data_to_process = self.original_data[columns]
        return pd.concat(
            [
                self.original_data[["CustomerID"]],
                pd.DataFrame(
                    self.dim_reduction_pipeline.fit_transform(tmp_data_to_process),
                    columns=["dim1", "dim2"]
                )
            ],
            axis=1
        )

    def run(self, columns: list) -> pd.DataFrame:

        _cluster_results = self.__process_clustering(columns=columns)
        _dim_reduction_results = self.__process_dim_reduction(columns=columns)

        return _cluster_results.merge(_dim_reduction_results,on="CustomerID")
        

    def plot_results(data: pd.DataFrame):
        sns.scatterplot(x="dim1", y="dim2", data=df_final, hue="cluster",palette="tab10")

    def calculate_clustering_metrics(data: pd.DataFrame):

    print(
        f"""
        PCA varinza explciada: {}
        Métrica Silloutte: {silhouette_score(X=data[customer_data_raw.columns[-3:]], labels=np.array(data['cluster']))}
        Métrica calinski_harabasz: {calinski_harabasz_score(X=data[customer_data_raw.columns[-3:]], labels=np.array(data['cluster']))}
        Métrica davies_bouldin: {davies_bouldin_score(X=data[customer_data_raw.columns[-3:]], labels=np.array(data['cluster']))}
        Adjusted Rand Index: { adjusted_rand_score(y, labels_pred)}
        Normalized Mutual Score : {normalized_mutual_info_score(y, labels)}
        """
        )


    plot_results(data=data)
    




























class ProcesoMLFLOW:
    def config_uri(self, uri: str):
        mlflow.set_tracking_uri(uri)
        print(f"Tracking URI configurado en: {uri}")
        return uri
    


'''Aca quiero poner la ruta de contador de experimentos en el mismo repo, ejemplo con os

def crear_experimento_mlflow(nombre_experimento: str, ruta_contador="contador_experimentos.txt"):
    if os.path.exists(ruta_contador):
        with open(ruta_contador, "r") as f:
            ultimo_numero = int(f.read().strip())
    else:
        ultimo_numero = 0

    nuevo_numero = ultimo_numero + 1
    with open(ruta_contador, "w") as f:
        f.write(str(nuevo_numero))

    fecha_actual = datetime.now().strftime("%d/%m/%y")
    if nombre_experimento is None:
        nombre_experimento = f"Experimento - {nuevo_numero}, {fecha_actual}"

    try:
        exp_id = mlflow.create_experiment(nombre_experimento)
    except Exception:
        exp_id = mlflow.get_experiment_by_name(nombre_experimento).experiment_id

    print(f"Usando experimento '{nombre_experimento}' con ID {exp_id}")
    return exp_id'''


    