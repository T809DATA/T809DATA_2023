import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    # Part 1.1
 nrm1 = (np.exp(-(np.power((x-mu), 2))/(2*np.power(sigma, 2)))/(np.power((2*np.pi*np.power(sigma,2)), 0.5)))
 return (nrm1)

#print(normal(0, 1, 0), normal(3, 1, 5), normal(np.array([-1,0,1]), 1, 0))



def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    # Part 1.2
    x_range = np.linspace(x_start, x_end, 500)
    #plt.clf()

    nrm2 = (np.exp(-(np.power((x_range-mu), 2))/(2*np.power(sigma, 2))) / (np.power((2*np.pi*np.power(sigma,2)), 0.5)))

    plt.plot(x_range, nrm2, label=f'Gaussian (mu={mu}, sigma={sigma})', linestyle='--')
    
    #plt.show()



def _plot_three_normals():
    # Part 1.2

    def plot_normal2(sigma: float, mu:float, x_range):
        nrm3 = (np.exp(-(np.power((x_range-mu), 2))/(2*np.power(sigma, 2))) / (np.power((2*np.pi*np.power(sigma,2)), 0.5)))
        plt.plot(x_range, nrm3, label=f'Sigma {sigma} & Mu {mu}')
    
    plt.clf()
    
    x_range = np.linspace(-5, 5, 500)

    plot_normal2(0.5, 0, x_range)
    plot_normal2(0.25, 1, x_range)
    plot_normal2(1, 1.25, x_range)
    
    plt.legend(loc='upper left')
    plt.title('Three Normal Distributions')
    plt.show()
    


def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1
    
    result = np.zeros_like(x)
    
    for i in range(len(sigmas)):
        weight = weights[i]
        sigma = sigmas[i]
        mu = mus[i]
        
        gaussian_density = weight *(np.exp(-(np.power((x-mu), 2))/(2*np.power(sigma, 2))) / (np.power((2*np.pi*np.power(sigma,2)), 0.5)))
        
        result += gaussian_density
        
    return result
""""
output = normal_mixture(np.linspace(-5, 5, 5), [0.5, 0.25, 1], [0, 1, 1.5], [1/3, 1/3, 1/3])
print(output)

output = normal_mixture(np.linspace(-2, 2, 4), [0.5], [0], [1])
print(output)
"""


def _compare_components_and_mixture():
    # Part 2.2
    x = np.linspace(-5, 5, 500)

    mu_values = [0, -0.5, 1.5]
    sigma_values = [0.5, 1.5, 0.25]
    weight_values = [1/3, 1/3, 1/3]

    plt.figure(figsize=(10, 6))

    for mu, sigma in zip(mu_values, sigma_values):
        plot_normal(sigma, mu, -5, 5)

    mixture = normal_mixture(x, sigma_values, mu_values, weight_values)
    plt.plot(x, mixture, label='Mixture')

    plt.title('Comparison of Normal Distributions and Mixture')
    plt.legend()
    plt.show()




def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.1

    sampled_data = []
    
    for _ in range(n_samples):
        component_index = np.random.multinomial(1, weights).argmax()
        sampled_value = np.random.normal(mus[component_index], sigmas[component_index])
        
        sampled_data.append(sampled_value)
    
    return np.array(sampled_data)

np.random.seed(0)
"""
result1 = sample_gaussian_mixture([0.1, 1], [-1, 1], [0.9, 0.1], 3)
print(result1)
result2 = sample_gaussian_mixture([1, 1, 1.5], [1, -1, 5], [0.1, 0.1, 0.8], 10)
print(result2)
"""


mu_values = [0, -1, 1.5]
sigma_values = [0.3, 0.5, 1]
phi_values = [0.2, 0.3, 0.5]

sample_counts = [10, 100, 500, 1000]

def _plot_mixture_and_samples():
    # Part 3.2

    plt.figure(figsize=(16, 6))

    for i, n_samples in enumerate(sample_counts, start=1):
        plt.subplot(1, len(sample_counts), i)

        samples = sample_gaussian_mixture(sigma_values, mu_values, phi_values, n_samples)

        plt.hist(samples, 50, density=True, alpha=0.6, label=f'{n_samples} samples')

        # Plot Gaussian mixture
        x_range = np.linspace(-5, 5, 500)
        mixture = normal_mixture(x_range, sigma_values, mu_values, phi_values)
        plt.plot(x_range, mixture, label='Mixture')

        plt.title(f'{n_samples} Samples')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
    #plot_normal(0.5, 0, -2, 2)
    #_plot_three_normals()
    #normal_mixture(np.linspace(-5, 5, 5), [0.5, 0.25, 1], [0, 1, 1.5], [1/3, 1/3, 1/3])
    #_compare_components_and_mixture()
    #sample_gaussian_mixture([0.1, 1], [-1, 1], [0.9, 0.1], 3)
    _plot_mixture_and_samples()