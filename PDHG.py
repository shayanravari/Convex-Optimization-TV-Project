import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.util import random_noise

def gradient(u):
    """Compute the gradient of u using forward differences."""
    grad = np.zeros(u.shape + (2,), dtype=u.dtype)
    grad[:-1,:,0] = u[1:,:] - u[:-1,:]   # Vertical gradients (du/dx)
    grad[:,:-1,1] = u[:,1:] - u[:,:-1]   # Horizontal gradients (du/dy)
    return grad

def divergence(p):
    """Compute the divergence of p using backward differences."""
    div = np.zeros(p.shape[:2], dtype=p.dtype)
    div[1:-1,:] = p[1:-1,:,0] - p[0:-2,:,0]
    div[0,:] = p[0,:,0]
    div[-1,:] = -p[-2,:,0]
    div[:,1:-1] += p[:,1:-1,1] - p[:,0:-2,1]
    div[:,0] += p[:,0,1]
    div[:,-1] += -p[:,-2,1]
    return div

def project_onto_unit_ball(p):
    """Project each vector in p onto the unit Euclidean ball."""
    norm_p = np.maximum(1.0, np.sqrt(p[:,:,0]**2 + p[:,:,1]**2))
    p[:,:,0] /= norm_p
    p[:,:,1] /= norm_p
    return p

# Load and normalize the image
image = img_as_float(data.camera())
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image, cmap='gray')
plt.axis('off')

# Add Gaussian noise to the image
sigma = 0.1  # Noise standard deviation
noisy_image = random_noise(image, var=sigma**2)
plt.subplot(1, 3, 2)
plt.title('Noisy')
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

# Parameters
lambda_param = 13  # Regularization parameter λ
max_iterations = 100  # Number of iterations

# Initialize variables
y = noisy_image.copy()
x = np.zeros(y.shape + (2,), dtype=y.dtype)

# PDHG algorithm parameters
tau = 0.2  # Step size τ_k
theta = 0.8  # Step size θ_k

# PDHG algorithm
for k in range(max_iterations):
    # Update t_k and θ_k
    tau += 0.08
    theta = (0.5 - (5/(15+k))) / tau

    # Compute gradient of the primal variable
    grad_y = gradient(y)
    
    # Dual variable update with projection
    x += tau * lambda_param * grad_y
    x = project_onto_unit_ball(x)
    
    # Compute divergence of the dual variable
    div_x = divergence(x)
    
    # Primal variable update
    y = (1 - theta) * y + theta * (noisy_image + (1 / lambda_param) * div_x)

# Display the denoised image
plt.subplot(1, 3, 3)
plt.title('Denoised')
plt.imshow(y, cmap='gray')
plt.axis('off')
plt.show()