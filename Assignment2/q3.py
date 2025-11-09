import numpy as np
import matplotlib.pyplot as plt

# ----- Parameters -----
sigma_r = 0.3
sigma_x = sigma_y = 0.25
K_vals = [1, 2, 3, 4]
x_true, y_true = 0.5, -0.5  # True position

# ----- Define landmarks -----
def generate_landmarks(K):
    angles = np.linspace(0, 2*np.pi, K, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

# ----- Objective Function -----
def map_objective(x, y, r, lm, sigma_r, sigma_x, sigma_y):
    d = np.sqrt((x - lm[:,0])**2 + (y - lm[:,1])**2)
    data_term = np.sum((r - d)**2) / (2*sigma_r**2)
    prior_term = x**2 / (2*sigma_x**2) + y**2 / (2*sigma_y**2)
    return data_term + prior_term

# Plot 
xg, yg = np.meshgrid(np.linspace(-2,2,200), np.linspace(-2,2,200))

for K in K_vals:
    lm = generate_landmarks(K)
    d_true = np.sqrt((x_true - lm[:,0])**2 + (y_true - lm[:,1])**2)
    r = d_true + np.random.normal(0, sigma_r, K)
    Z = np.array([[map_objective(x,y,r,lm,sigma_r,sigma_x,sigma_y)
                   for x in np.linspace(-2,2,200)] for y in np.linspace(-2,2,200)])
    plt.figure()
    cs = plt.contour(xg, yg, Z, levels=30)
    plt.clabel(cs, inline=1, fontsize=8)
    plt.scatter(lm[:,0], lm[:,1], c='r', marker='o', label='Landmarks')
    plt.scatter(x_true, y_true, c='g', marker='+', s=100, label='True Pos.')
    plt.title(f'MAP Objective Contours, K={K}')
    plt.legend()
    plt.axis('equal')
    plt.show()
