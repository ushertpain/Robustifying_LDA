import GenCon
import LDAandEstimator
import numpy as np
from statistics import stdev
from statistics import variance
class Generate:
    sigma = [[1, 0], [0, 1]]
    contaminated_sigma = [[100, 0], [0, 100]]
    sample1 = []
    sample2 = []

    def __init__(self, N):
        self.N = N

    def generate(self):
        self.sample1 = np.random.multivariate_normal([1, 1], self.sigma, self.N)
        self.sample2 = np.random.multivariate_normal([-1, -1], self.sigma, self.N)
        x1 = np.ones((self.N, 1))
        x1 = np.concatenate((self.sample1, x1), axis=1)
        x2 = 2 * np.ones((self.N, 1))
        x2 = np.concatenate((self.sample2, x2), axis=1)
        return np.concatenate((x1, x2), axis=0)

    def contaminate(self, percentage):
        number_of_contaminated_data = round(self.N * percentage)
        to_contaminate_sample1 = np.random.multivariate_normal([10, 10], self.contaminated_sigma,
                                                               number_of_contaminated_data)
        to_contaminate_sample2 = np.random.multivariate_normal([-10, -10], self.contaminated_sigma,
                                                               number_of_contaminated_data)
        s1 = self.sample1[number_of_contaminated_data:]
        s2 = self.sample2[number_of_contaminated_data:]
        self.sample1 = np.concatenate((to_contaminate_sample1, s1), axis=0)
        self.sample2 = np.concatenate((to_contaminate_sample2, s2), axis=0)
        x1 = np.ones((self.N, 1))
        x1 = np.concatenate((self.sample1, x1), axis=1)
        x2 = 2 * np.ones((self.N, 1))
        x2 = np.concatenate((self.sample2, x2), axis=1)
        return np.concatenate((x1, x2), axis=0)


def tpm(data,m1,m2,z1,z2):
    n1 = np.size(data, axis=0)/2
    n2 = n1
    #sigma1 = (1/(n1+n2-2))*(((n1-1)*Z1)+((n2-1)*Z2));
    sigma = np.dot((1/(n1+n2-2)), np.add((np.dot((n1-1),z1)),(np.dot((n2-1),z2))))
    #sigma = np.divide(np.subtract(z1, z2),(n1+n2-2))
    #sigma = [[1,0],[0,1]]
    #linear = (M1-M2)*inv(sigma1);
    linear = np.dot((np.subtract(m1, m2)), np.linalg.inv(sigma))
    #print(linear)
    #constant=(1/2)*linear*(M1+M2)';
    constant = np.dot((1/2), (np.dot(linear, (np.add(m1, m2)).T)))

    #scores = linear*data(:,1:2)'-constant;
    scores = np.subtract((np.dot(linear, data[:, (0, 1)].T)), constant)

    #a = data[:, (0, 1)].T
    #b = np.dot(linear, a)
    #scores = np.subtract(b, constant)
    #scores = np.subtract(np.multiply(linear, data[:, (0, 1)].T), constant)
    scores = scores.T

    #group = (scores < 0) + 1;
    group = (scores < 0) + 1

    #miscl = mean(group ~= data(:,3));
    miscl = np.mean(group[:, 0] != data[:, 2])
    return miscl


if __name__ == '__main__':
    '''x = Generate(10).generate()
    y = Generate(10).generate()
    k = np.array([[1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    k = k.T
    print((x<0) + 1)'''

    p1 = GenCon.Create(100)
    p2 = GenCon.Create(100)
    p1.generate_population1()
    p2.generate_population2()
    p1.contaminate_p1(0.3)
    p2.contaminate_p2(0.3)
    x = p1.sample_data
    y = p2.sample_data
    m1, z1 = LDAandEstimator.s_Sn(x)
    m2, z2 = LDAandEstimator.s_Sn(y)
    l = []
    data_to_check = Generate(100)
    for i in range(10000):

        z = data_to_check.generate()
        l.append(tpm(z, m1, m2, z1, z2))
    print(np.average(l), stdev(l), variance(l))
    '''z = data_to_check.generate()
    print(tpm(z, m1, m2, z1, z2))'''




