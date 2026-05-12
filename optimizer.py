def gradient(x: float) -> float:
    """
    Compute the gradient of f(x) = x^2 - 6x + 9 at point x.
    Gradient: f'(x) = 2x - 6
    """
    return 2 * x - 6


def cost_function(x: float) -> float:
    """
    Compute the cost f(x) = x^2 - 6x + 9.
    """
    return x**2 - 6*x + 9


def optimize(x_initial: float, learning_rate: float, max_iterations: int, tolerance: float = 1e-6):
    """
    Gradient descent optimizer for f(x) = x^2 - 6x + 9.
    
    Parameters:
    - x_initial: Starting value of x
    - learning_rate: Step size for each gradient update
    - max_iterations: Maximum number of iterations allowed
    - tolerance: Early stopping threshold (stops when |change in x| < tolerance)
    
    Returns:
    - x_opt: Final optimized x value
    - min_cost: Minimum cost achieved
    - iterations: Total number of iterations run
    - converged: Boolean indicating whether early stopping was triggered
    """
    x = x_initial
    previous_cost = cost_function(x)
    
    for iteration in range(max_iterations):
        grad = gradient(x)
        
        # Gradient update step
        x_new = x - learning_rate * grad
        
        # Check for convergence (step size)
        if abs(x_new - x) < tolerance:
            x = x_new
            final_cost = cost_function(x)
            return x, final_cost, iteration + 1, True
        
        x = x_new
        current_cost = cost_function(x)
        previous_cost = current_cost
    
    final_cost = cost_function(x)
    return x, final_cost, max_iterations, False


# Example usage
if __name__ == "__main__":
    # Test with different parameters
    result1 = optimize(x_initial=10.0, learning_rate=0.1, max_iterations=100, tolerance=1e-6)
    print(f"x_initial=10, lr=0.1: x_opt={result1[0]:.6f}, cost={result1[1]:.6f}, iterations={result1[2]}, converged={result1[3]}")
    
    result2 = optimize(x_initial=-5.0, learning_rate=0.1, max_iterations=100, tolerance=1e-6)
    print(f"x_initial=-5, lr=0.1: x_opt={result2[0]:.6f}, cost={result2[1]:.6f}, iterations={result2[2]}, converged={result2[3]}")
    
    result3 = optimize(x_initial=10.0, learning_rate=0.01, max_iterations=1000, tolerance=1e-8)
    print(f"x_initial=10, lr=0.01: x_opt={result3[0]:.6f}, cost={result3[1]:.6f}, iterations={result3[2]}, converged={result3[3]}")