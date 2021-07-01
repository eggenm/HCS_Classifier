import hcs_database as db

class Sampler:
    def __init__(self):
        self.fixed_sample_rate = 25
        self.fixed_test_rate = 5
        self.concession_test_rate = 0.2
        self.sample_rate_by_island = {
                'Sumatra':0.1,
                'Kalimantan':0.85,
                'Papua':0.15
            }
        self.sampler_train_test_dict = {
            'Sumatra': (100, self.concession_test_rate),
            'Kalimantan': (100, self.concession_test_rate),
            'Papua': (100, self.concession_test_rate),
            'supplementary_class': (self.fixed_sample_rate, self.fixed_test_rate)
        }


    def get_sample_rate_by_type(self, total_samples, sites):
        count_by_island = dict()
        for site in sites:
            type_of_class = db.data_context_dict[site]
            if(type_of_class!='supplementary_class'):
                island = type_of_class
                try:
                    count_by_island[island]+=1
                except KeyError:
                    count_by_island[island] = 1
        for island in count_by_island.keys():
            self.sampler_train_test_dict[island] = (total_samples*self.sample_rate_by_island[island]/count_by_island[island], self.concession_test_rate)
        return self.sampler_train_test_dict




