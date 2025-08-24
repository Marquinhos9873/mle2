

class UnsupervisedProcessor:

    def __init__(self, dataframe: pd.DataFrame, scaler, cluster_algorithm, dim_reduction_algorithm, orderby_columna: str) -> None:
        self.original_data = dataframe
        self.orderby_columna = orderby_columna
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
                self.original_data[self.orderby_columna],
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
                self.original_data[self.orderby_columna],
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

        return _cluster_results.merge(_dim_reduction_results,on=orderby_columna)
        

    def plot_results(self, data: pd.DataFrame):
        sns.scatterplot(x="dim1", y="dim2", data=df_final, hue="cluster",palette="tab10")
    plot_results(data=data)

    def calculate_clustering_metrics(self, data: pd.DataFrame):
        
        print(
        f"""
        PCA varianza explicada: {self.variance_ratio}
        Métrica Silloutte: {silhouette_score(X=data[customer_data_raw.columns[-3:]], labels=np.array(data['cluster']))}
        Métrica calinski_harabasz: {calinski_harabasz_score(X=data[customer_data_raw.columns[-3:]], labels=np.array(data['cluster']))}
        Métrica davies_bouldin: {davies_bouldin_score(X=data[customer_data_raw.columns[-3:]], labels=np.array(data['cluster']))}
        Adjusted Rand Index: { adjusted_rand_score(y, labels_pred)}
        Normalized Mutual Score : {normalized_mutual_info_score(y, labels)}
        """
        )


    
    



