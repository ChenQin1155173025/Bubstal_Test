import sys
import os

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from naive_bayes_classifier import NaiveBayesClassifier
from optimizer import gradient, cost_function, optimize
from dataset import TRAINING_DATA


# ============================================================
# HELPER FUNCTION FOR TESTS
# ============================================================
def assert_equals(actual, expected, test_name, tolerance=0.0001):
    """Custom assertion with tolerance for floats"""
    if isinstance(expected, float):
        passed = abs(actual - expected) < tolerance
    else:
        passed = (actual == expected)
    
    if passed:
        print(f"  ✓ {test_name} passed")
        return True
    else:
        print(f"  ✗ {test_name} failed: expected {expected}, got {actual}")
        return False


# ============================================================
# TEST 1: PRIOR PROBABILITIES (Naive Bayes)
# ============================================================
def test_prior_probabilities():
    """Test that prior probabilities are calculated correctly"""
    nb = NaiveBayesClassifier(smoothing_alpha=1.0)
    nb.train(TRAINING_DATA)
    
    # With 10 Happy + 10 Frustrated, both priors should be 0.5
    total_docs = sum(nb.document_counts.values())
    
    happy_prior = nb.document_counts.get("Happy", 0) / total_docs
    frustrated_prior = nb.document_counts.get("Frustrated", 0) / total_docs
    
    print(f"\n  Happy prior: {happy_prior}, Frustrated prior: {frustrated_prior}")
    
    result1 = assert_equals(happy_prior, 0.5, "Prior probability for Happy class")
    result2 = assert_equals(frustrated_prior, 0.5, "Prior probability for Frustrated class")
    
    return result1 and result2


# ============================================================
# TEST 2: PREDICTION RETURNS VALID CLASS
# ============================================================
def test_prediction_returns_valid_class():
    """Test that prediction always returns either 'Happy' or 'Frustrated'"""
    nb = NaiveBayesClassifier(smoothing_alpha=1.0)
    nb.train(TRAINING_DATA)
    
    test_texts = [
        "I love this product",
        "This is terrible",
        "Random unknown sentence with no matching words",
        ""
    ]
    
    all_valid = True
    for text in test_texts:
        pred, scores, conf = nb.predict(text)
        if pred not in ["Happy", "Frustrated"]:
            all_valid = False
            print(f"  Invalid prediction: {pred} for text: {text}")
            break
    
    assert_equals(all_valid, True, "Prediction always returns 'Happy' or 'Frustrated'")
    return all_valid


# ============================================================
# TEST 3: GRADIENT CALCULATION
# ============================================================
def test_gradient_calculation():
    """Test gradient at specific points: f'(x) = 2x - 6"""
    tests = [
        (0, -6, "gradient at x=0"),
        (3, 0, "gradient at x=3 (minimum)"),
        (5, 4, "gradient at x=5"),
        (-2, -10, "gradient at x=-2")
    ]
    
    all_passed = True
    for x, expected, name in tests:
        result = gradient(x)
        if not assert_equals(result, expected, name):
            all_passed = False
    
    return all_passed


# ============================================================
# TEST 4: GRADIENT DESCENT UPDATE STEP
# ============================================================
def test_gradient_descent_update():
    """Test that a single gradient update moves x toward minimum"""
    x = 10.0
    learning_rate = 0.1
    
    grad = gradient(x)  # 2*10 - 6 = 14
    x_new = x - learning_rate * grad  # 10 - 1.4 = 8.6
    
    # After update, should be closer to 3
    distance_before = abs(x - 3)
    distance_after = abs(x_new - 3)
    
    result1 = assert_equals(distance_after < distance_before, True, "Gradient update moves toward minimum")
    result2 = assert_equals(x_new, 8.6, "Exact gradient update value")
    
    return result1 and result2


# ============================================================
# TEST 5: OPTIMIZER CONVERGES TO MINIMUM
# ============================================================
def test_optimizer_converges_to_minimum():
    """Test that optimizer converges to x=3 (global minimum)"""
    x_opt, min_cost, iterations, converged = optimize(
        x_initial=100.0,
        learning_rate=0.1,
        max_iterations=1000,
        tolerance=1e-6
    )
    
    print(f"\n  Converged at x={x_opt:.8f}, cost={min_cost:.8f}, iterations={iterations}")
    
    result1 = assert_equals(x_opt, 3.0, "Optimizer converges to x=3 (minimum)", tolerance=1e-4)
    result2 = assert_equals(min_cost, 0.0, "Minimum cost is 0", tolerance=1e-8)
    result3 = assert_equals(converged, True, "Optimizer converges (early stopping triggered)")
    
    return result1 and result2 and result3


# ============================================================
# TEST 6: OPTIMIZER RESPECTS MAX ITERATIONS
# ============================================================
def test_optimizer_respects_max_iterations():
    """Test that optimizer stops after max_iterations even if not converged"""
    x_opt, min_cost, iterations, converged = optimize(
        x_initial=100.0,
        learning_rate=0.0001,  # Very small learning rate
        max_iterations=50,
        tolerance=1e-12  # Very strict tolerance
    )
    
    print(f"\n  Stopped at iteration {iterations} (max was 50)")
    
    return assert_equals(iterations, 50, "Optimizer respects max_iterations")


# ============================================================
# TEST 7: OPTIMIZER EARLY STOPPING
# ============================================================
def test_optimizer_early_stopping_with_loose_tolerance():
    """Test that early stopping works with loose tolerance"""
    x_opt, min_cost, iterations, converged = optimize(
        x_initial=10.0,
        learning_rate=0.1,
        max_iterations=1000,
        tolerance=0.1  # Loose tolerance = early stop
    )
    
    print(f"\n  Stopped at iteration {iterations} (early stopping)")
    print(f"  Final x={x_opt:.4f}, distance from 3: {abs(x_opt - 3):.4f}")
    
    result1 = assert_equals(iterations < 50, True, "Early stopping with loose tolerance (iterations < 50)")
    result2 = assert_equals(converged, True, "Early stopping triggered")
    
    return result1 and result2


# ============================================================
# RUN ALL TESTS
# ============================================================
def test_all():
    """Execute all tests and print summary"""
    print("\n" + "="*70)
    print("RUNNING UNIT TESTS")
    print("="*70 + "\n")
    
    # List of all tests
    tests = [
        ("Test 1: Prior Probabilities", test_prior_probabilities),
        ("Test 2: Prediction Returns Valid Class", test_prediction_returns_valid_class),
        ("Test 3: Gradient Calculation", test_gradient_calculation),
        ("Test 4: Gradient Descent Update", test_gradient_descent_update),
        ("Test 5: Optimizer Converges to Minimum", test_optimizer_converges_to_minimum),
        ("Test 6: Optimizer Respects Max Iterations", test_optimizer_respects_max_iterations),
        ("Test 7: Optimizer Early Stopping", test_optimizer_early_stopping_with_loose_tolerance),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 50)
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*70)
    
    return passed, failed


# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    test_all()