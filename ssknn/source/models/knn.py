from source.dataprep.dataprep import generate_sequential_matrix
from source.helpers.knn import truncate_similarity
from source.helpers.similarity import compute_similarity

class SKNN:
    def __init__(self, model_config=None) -> None:
        self.similarity_type = model_config['similarity']
        self.n_neighbors = model_config['n_neighbors']

    def build(self, data, data_description):
        interactions = generate_sequential_matrix(data, data_description=data_description)
        self.interactions = interactions

    def recommend(self, test_data, data_description):
        test_interactions = generate_sequential_matrix(data=test_data,
                                                       data_description = data_description,
                                                       rebase_users=True)
        full_similarity = compute_similarity(self.similarity_type,
                                             test_interactions.astype('bool').astype('int'),
                                             self.interactions.astype('bool').astype('int'))
        similarity = truncate_similarity(similarity=full_similarity, k=self.n_neighbors)
        scores = similarity.dot(self.interactions).toarray()
        return scores
    
    
class V_SKNN:
    def __init__(self, model_config=None) -> None:
        self.similarity_type = model_config['similarity']
        self.n_neighbors = model_config['n_neighbors']

    def build(self, data, data_description):
        interactions = generate_sequential_matrix(data, data_description)
        self.interactions = interactions

    def recommend(self, test_data, data_description):
        test_interactions = generate_sequential_matrix(test_data, data_description, rebase_users=True)
        full_similarity = compute_similarity(self.similarity_type, test_interactions, self.interactions)
        similarity = truncate_similarity(similarity=full_similarity, k=self.n_neighbors)
        scores = similarity.dot(self.interactions.astype('bool').astype('int')).toarray()
        return scores
    
    
class S_SKNN:
    def __init__(self, model_config=None) -> None:
        self.similarity_type = model_config['similarity']
        self.n_neighbors = model_config['n_neighbors']

    def build(self, data, data_description):
        interactions = generate_sequential_matrix(data, data_description)
        self.interactions = interactions

    def recommend(self, test_data, data_description):
        test_interactions = generate_sequential_matrix(test_data, data_description, rebase_users=True)
        full_similarity = compute_similarity(self.similarity_type, test_interactions, self.interactions)
        similarity = truncate_similarity(similarity=full_similarity, k=self.n_neighbors)
        scores = similarity.dot(self.interactions).toarray()
        return scores
