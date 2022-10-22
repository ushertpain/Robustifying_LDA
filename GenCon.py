import numpy as np

#this class generate sample for our discriminant rule
class Create:
    mu1 = [1, 1]
    mu2 = [-1, -1]
    contaminated_mu1 = [10, 10]
    contaminated_mu2 = [-10, -10]
    sigma = [[1, 0], [0, 1]]
    contaminated_sigma = [[100, 0], [0, 100]]
    sample_data = []
    def __init__(self, N):
        self.N = N

    def generate_population1(self):
        self.sample_data = np.random.multivariate_normal(self.mu1, self.sigma, self.N)

    def generate_population2(self):
        self.sample_data = np.random.multivariate_normal(self.mu2, self.sigma, self.N)

    def contaminate_p1(self, epsilon):
        number_of_contaminated_data = round(self.N * epsilon)
        to_contaminate_sample1 = np.random.multivariate_normal(self.contaminated_mu1, self.contaminated_sigma,
                                                               number_of_contaminated_data)
        self.sample_data = np.concatenate((to_contaminate_sample1,
                                           self.sample_data[number_of_contaminated_data:]), axis=0)

    def contaminate_p2(self, epsilon):
        number_of_contaminated_data = round(self.N * epsilon)
        to_contaminate_sample2 = np.random.multivariate_normal(self.contaminated_mu2, self.contaminated_sigma,
                                                               number_of_contaminated_data)
        self.sample_data = np.concatenate((to_contaminate_sample2,
                                           self.sample_data[number_of_contaminated_data:]), axis=0)

'''sample1 = Create(10)
sample2 = Create(10)
sample2.generate_population2()
print(sample2.sample_data)
print(np.mean(sample2.sample_data))
sample2.contaminate_p2(.3)
print(sample2.sample_data)'''
