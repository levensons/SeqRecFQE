from scipy.sparse import coo_matrix
from source.dataprep.dataprep import generate_interactions_matrix

class AR:
    def __init__(self, model_config=None) -> None:
        pass

    def build(self, data, data_description):
        '''
        Builds association rules matrix.
        '''
        interactions = generate_interactions_matrix(data, data_description)

        similarity = interactions.T.dot(interactions)
        similarity.setdiag(0)
        similarity.eliminate_zeros()
        self.rules = similarity

    def recommend(self, data, data_description):
        '''
        Generate scores for given data.
        '''
        # Drop duplicates, keeping the last interaction for each user

        data_sorted = data.sort_values(by=[data_description['users'], data_description['timestamp']])
        data_last_interaction = data_sorted.drop_duplicates(subset=data_description['users'], keep='last')

        interactions = generate_interactions_matrix(data_last_interaction, data_description, rebase_users=True)

        scores = interactions.dot(self.rules).toarray()
        return scores
    
    
class SR:
    def __init__(self, model_config=None) -> None:
        pass

    def build(self, data, data_description):
        'Builds sequential rules of size two'
        rules = {}

        # get chronological interaction history for each user
        histories = (
            data
            .sort_values(
                by=data_description['timestamp']
                )
            .groupby(data_description['users'])[data_description['items']]
            .apply(list)
            )

        # count the number of pairs when item j is interacted with right after item i
        for history in histories:
            for i in range(len(history) - 1):
                if (history[i], history[i + 1]) not in rules:
                    rules[(history[i], history[i + 1])] = 0
                rules[(history[i], history[i + 1])] += 1

        # create a sparse matrix of sequential rules for easier recommendation
        items, values = zip(*rules.items())
        i1, i2 = zip(*items)
        matrix_shape = (data_description['n_items'], data_description['n_items'])
        similarity = coo_matrix((values, (list(i1), list(i2))), shape=matrix_shape).tocsr()

        self.rules = similarity

    def recommend(self, data, data_description):
        '''
        Generate scores for given data.
        '''
        data_sorted = data.sort_values(by=[data_description['users'], data_description['timestamp']])
        data_last_interaction = data_sorted.drop_duplicates(subset=data_description['users'], keep='last')

        interactions = generate_interactions_matrix(data_last_interaction, data_description, rebase_users=True)

        scores = interactions.dot(self.rules).toarray()
        return scores