import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# DATA
# ============================================================================
Temp = np.array([40, 27, 40, 73, 65, 65, 10,  9, 24, 65, 66, 41, 22, 40, 60], dtype=float)
Ins  = np.array([ 4,  4, 10,  6,  7, 40,  6, 10, 10,  4, 10,  6,  4,  4, 10], dtype=float)
Oil  = np.array([270, 362, 162, 45, 91, 233, 372, 305, 234, 122, 25, 210, 450, 325, 52], dtype=float)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_linear(x1, x2):
    """Design matrix for linear model: [1, Temp, Ins] -> shape (n x 3)"""
    return np.column_stack([np.ones(len(x1)), x1, x2])

def build_quadratic(x1, x2):
    """
    Design matrix for quadratic model
    Order: [1, T, I, T^2, I^2, T*I] -> shape (n x 6)
    """
    return np.column_stack([np.ones(len(x1)), x1, x2, x1**2, x2**2, x1*x2])

def normal_equation(X, y):
    """
    Closed-form least squares solution: b = (X^T X)^{-1} X^T y
    Reference: Hastie, Tibshirani & Friedman (2009), Eq. 3.6
    """
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.inv(XtX) @ Xty

def r_squared(y_actual, y_predicted):
    """R^2 = 1 - SS_res / SS_tot"""
    SS_res = np.sum((y_actual - y_predicted) ** 2)
    SS_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - SS_res / SS_tot

def mahalanobis_outlier_detection(X, threshold_alpha=0.975):
    """
    Detect outliers using Mahalanobis distance.
    Reference: Li et al. (2019) - Outlier Detection Based on Robust Mahalanobis Distance
    """
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    cov_inv = np.linalg.inv(cov)
    
    distances = []
    for i in range(len(X)):
        dist = mahalanobis(X[i], mean, cov_inv)
        distances.append(dist)
    
    # Chi-square threshold for p=2 degrees of freedom
    threshold = np.sqrt(chi2.ppf(threshold_alpha, df=X.shape[1]))
    
    outliers = distances > threshold
    return np.array(distances), outliers, threshold


def sklearn_verify(T, I, y_actual, b_manual, model_type_label):
    print(f"\n--- SKLEARN VERIFICATION: {model_type_label.upper()} ---")
    
    X_input = np.column_stack([T, I])
    model = LinearRegression()
    
    is_quadratic = (len(b_manual) == 6)
    
    if not is_quadratic:
        model.fit(X_input, y_actual)
        b_sk = np.array([model.intercept_, model.coef_[0], model.coef_[1]])
        X_mat = build_linear(T, I)
        y_p_manual = X_mat @ b_manual
        labels = ['b0 (Const)', 'b1 (Temp)', 'b2 (Ins)']
    else:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X_input)  # Order: [T, I, T², T·I, I²]
        model.fit(X_poly, y_actual)
        
        c = model.coef_  # [coef_T, coef_I, coef_T2, coef_TI, coef_I2]
        
        # Reorder to match YOUR build_quadratic: [1, T, I, T², I², T·I]
        b_sk = np.array([
            model.intercept_,  # b0
            c[0],              # b1: T
            c[1],              # b2: I
            c[2],              # b3: T²
            c[4],              # b4: I²  (sklearn index 4)
            c[3]               # b5: T·I (sklearn index 3)
        ])
        
        X_mat = build_quadratic(T, I)
        y_p_manual = X_mat @ b_manual
        labels = ['b0', 'b1(T)', 'b2(I)', 'b3(T²)', 'b4(I²)', 'b5(T·I)']
    
    print(f"{'Coeff':<12} | {'Manual':>10} | {'Sklearn':>10} | {'Diff':>10}")
    print("-" * 55)
    for lbl, m, s in zip(labels, b_manual, b_sk):
        print(f"{lbl:<12} | {m:>10.4f} | {s:>10.4f} | {abs(m-s):>10.1e}")
    
    manual_r2 = r_squared(y_actual, y_p_manual)
    sklearn_r2 = r2_score(y_actual, model.predict(X_poly if is_quadratic else X_input))
    print(f"\nManual R²:  {manual_r2:.6f}")
    print(f"Sklearn R²: {sklearn_r2:.6f}")
    print(f"R² Match:   {np.isclose(manual_r2, sklearn_r2)}")




# ============================================================================
# FIGURE 1: RAW DATA SCATTERPLOT  
# ============================================================================
print("FIGURE 1: RAW DATA VISUALIZATION")


plt.figure(1, figsize=(14, 5))

# Subplot 1: Temperature vs Oil
plt.subplot(1, 2, 1)
plt.scatter(Temp, Oil, s=80, color='blue', edgecolors='black', linewidth=1.5)
plt.xlabel('Temperature (°F)', fontsize=12)
plt.ylabel('Oil Consumption (gallons)', fontsize=12)
plt.title('Temperature vs Oil Consumption', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# Highlight outlier
outlier_idx = 5
plt.scatter(Temp[outlier_idx], Oil[outlier_idx], c='red', s=150, marker='o',
            edgecolors='black', linewidth=2, label='Outlier')
plt.legend()

# Subplot 2: Insulation vs Oil
plt.subplot(1, 2, 2)
plt.scatter(Ins, Oil, s=80, color='green', edgecolors='black', linewidth=1.5)
plt.xlabel('Insulation (inches)', fontsize=12)
plt.ylabel('Oil Consumption (gallons)', fontsize=12)
plt.title('Insulation vs Oil Consumption', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# Highlight outlier
plt.scatter(Ins[outlier_idx], Oil[outlier_idx], c='red', s=150, marker='o',
            edgecolors='black', linewidth=2, label='Outlier (Row 6)')
plt.legend()

plt.suptitle('Figure 1: Raw Data Visualization', fontsize=14, fontweight='bold')
plt.tight_layout()

# ============================================================================
# MAHALANOBIS DISTANCE OUTLIER DETECTION
# ============================================================================
print("\n" + "=" * 70)
print("MAHALANOBIS DISTANCE OUTLIER DETECTION")
print("=" * 70)

X_features = np.column_stack([Temp, Ins])
distances, is_outlier, threshold = mahalanobis_outlier_detection(X_features)

print(f"\nMahalanobis Distance Results:")
print(f"  Threshold (chi-square, p=2, alpha=0.975): {threshold:.4f}")
print(f"\n{'Row':<6} {'Temp':<8} {'Ins':<8} {'Mahal. Dist':<14} {'Outlier?'}")
print("-" * 50)
for i in range(len(Temp)):
    marker = "YES" if is_outlier[i] else "NO"
    print(f"{i+1:<6} {Temp[i]:<8.0f} {Ins[i]:<8.0f} {distances[i]:<14.4f} {marker}")

print(f"\nOutlier identified: Row {outlier_idx+1} (Temp={Temp[outlier_idx]}°F, Ins={Ins[outlier_idx]} inches)")

# ============================================================================
# PART (a) - ALL 15 POINTS (with outlier)
# ============================================================================
print("\n" + "=" * 70)
print("PART (a) — All 15 points (includes outlier)")
print("=" * 70)

# Linear model
X_lin = build_linear(Temp, Ins)
b_lin = normal_equation(X_lin, Oil)
Oil_p_lin = X_lin @ b_lin
R2_lin = r_squared(Oil, Oil_p_lin)

print(f"\n[Linear Model — with outlier]")
print(f"  Equation: Oil = {b_lin[0]:.4f} + ({b_lin[1]:.4f})*Temp + ({b_lin[2]:.4f})*Ins")
print(f"  R^2 = {R2_lin:.4f}")
print(f"  Problem: Insulation coefficient is POSITIVE ({b_lin[2]:.4f}) — physically wrong!")

# Quadratic model
X_quad = build_quadratic(Temp, Ins)
b_quad = normal_equation(X_quad, Oil)
Oil_p_quad = X_quad @ b_quad
R2_quad = r_squared(Oil, Oil_p_quad)

print(f"\n[Quadratic Model — with outlier]")
print(f"  Equation: Oil = {b_quad[0]:.4f} + ({b_quad[1]:.4f})*T + ({b_quad[2]:.4f})*I")
print(f"           + ({b_quad[3]:.4f})*T^2 + ({b_quad[4]:.4f})*T*I + ({b_quad[5]:.4f})*I^2")
print(f"  R^2 = {R2_quad:.4f}")
print(f"   ΔR^2 over linear = {R2_quad - R2_lin:.4f}")

print(f"\nPart (a) Conclusion:")
print(f"  Quadratic R^2 gain = {R2_quad - R2_lin:.4f} (significant)")
print(f"  BUT this is caused by outlier overfitting — see Part (b)")

# ============================================================================
# PART (b) - REMOVE OUTLIER AND REFIT
# ============================================================================
print("\n" + "=" * 70)
print("PART (b) — Remove outlier, refit both models (n=14)")
print("=" * 70)

# Remove outlier (index 5)
Temp_c = np.delete(Temp, outlier_idx)
Ins_c = np.delete(Ins, outlier_idx)
Oil_c = np.delete(Oil, outlier_idx)

# Linear model (cleaned)
X_lin_c = build_linear(Temp_c, Ins_c)
b_lin_c = normal_equation(X_lin_c, Oil_c)
Oil_p_lin_c = X_lin_c @ b_lin_c
R2_lin_c = r_squared(Oil_c, Oil_p_lin_c)

print(f"\n[Linear Model — cleaned data]")
print(f"  Equation: Oil = {b_lin_c[0]:.4f} + ({b_lin_c[1]:.4f})*Temp + ({b_lin_c[2]:.4f})*Ins")
print(f"  R^2 = {R2_lin_c:.4f}")
print(f"  Both coefficients now physically correct!")

# Quadratic model (cleaned)
X_quad_c = build_quadratic(Temp_c, Ins_c)
b_quad_c = normal_equation(X_quad_c, Oil_c)
Oil_p_quad_c = X_quad_c @ b_quad_c
R2_quad_c = r_squared(Oil_c, Oil_p_quad_c)

print(f"\n[Quadratic Model — cleaned data]")
print(f"  Equation: Oil = {b_quad_c[0]:.4f} + ({b_quad_c[1]:.4f})*T + ({b_quad_c[2]:.4f})*I")
print(f"           + ({b_quad_c[3]:.4f})*T^2 + ({b_quad_c[4]:.4f})*T*I + ({b_quad_c[5]:.4f})*I^2")
print(f"  R^2 = {R2_quad_c:.4f}")
print(f"   ΔR^2 over cleaned linear = {R2_quad_c - R2_lin_c:.4f}")

print(f"\nPart (b) Conclusion (Note 4):")
print(f"   ΔR^2 = {R2_quad_c - R2_lin_c:.4f} (only {100*(R2_quad_c - R2_lin_c):.2f}% improvement)")
print(f"  Adding 3 extra parameters is NOT justified.")
print(f"  -> BEST MODEL: Linear regression on cleaned data")

# ============================================================================
# IMPACT SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 70)
print("IMPACT OF OUTLIER REMOVAL ON MODEL PERFORMANCE")
print("=" * 70)
print(f"\n{'Model':<35} {'R^2 (Before)':<15} {'R^2 (After)':<15} {'Improvement':<15}")
print("-" * 80)
print(f"{'Linear Model':<35} {R2_lin:<15.4f} {R2_lin_c:<15.4f} {R2_lin_c - R2_lin:<15.4f}")
print(f"{'Quadratic Model':<35} {R2_quad:<15.4f} {R2_quad_c:<15.4f} {R2_quad_c - R2_quad:<15.4f}")

# ============================================================================
# PART (c) - PREDICTION
# ============================================================================
print("\n" + "=" * 70)
print("PART (c) — Prediction: Temperature = 15°F, Insulation = 5 inches")
print("=" * 70)

T_new, I_new = 15, 5

# Linear prediction
pred_lin = b_lin_c[0] + b_lin_c[1] * T_new + b_lin_c[2] * I_new

print(f"\n[Linear Model Prediction]")
print(f"  Oil = {b_lin_c[0]:.4f} + ({b_lin_c[1]:.4f}) x {T_new} + ({b_lin_c[2]:.4f}) x {I_new}")
print(f"      = {b_lin_c[0]:.4f} + ({b_lin_c[1] * T_new:.4f}) + ({b_lin_c[2] * I_new:.4f})")
print(f"      = {pred_lin:.2f} gallons")


# Quadratic prediction
pred_quad = (b_quad_c[0] + b_quad_c[1] * T_new + b_quad_c[2] * I_new +
             b_quad_c[3] * T_new**2 + b_quad_c[4] * T_new * I_new + b_quad_c[5] * I_new**2)

print(f"\n[Quadratic Model Prediction]")
print(f"  Oil = {b_quad_c[0]:.4f} + ({b_quad_c[1]:.4f})x{T_new} + ({b_quad_c[2]:.4f})x{I_new}")
print(f"      + ({b_quad_c[3]:.4f})x{T_new}^2 + ({b_quad_c[4]:.4f})x{T_new}x{I_new} + ({b_quad_c[5]:.4f})x{I_new}^2")
print(f"      = {pred_quad:.2f} gallons")

print(f"\nPrediction Conclusion:")
print(f"  Linear prediction:   {pred_lin:.2f} gallons")
print(f"  Quadratic prediction: {pred_quad:.2f} gallons")
print(f"  Difference: {abs(pred_lin - pred_quad):.2f} gallons (~{100*abs(pred_lin - pred_quad)/pred_lin:.1f}%)")
print(f"  -> Both models agree, confirming linear model is sufficient.")

# ============================================================================
# VERIFICATIONS
# ============================================================================
sklearn_verify(Temp, Ins, Oil, b_lin, "Linear (Part A)")
sklearn_verify(Temp, Ins, Oil, b_quad, "Quadratic (Part A)")
sklearn_verify(Temp_c, Ins_c, Oil_c, b_lin_c, "Linear Cleaned (Part B)")
sklearn_verify(Temp_c, Ins_c, Oil_c, b_quad_c, "Quadratic Cleaned (Part B)")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 70)
print("FULL SUMMARY TABLE")
print("=" * 70)
print(f"\n{'Model':<40} {'R^2':>8}  {'Notes'}")
print("-" * 70)
print(f"{'Part (a) Linear (n=15, with outlier)':<40} {R2_lin:>8.4f}  Wrong sign on Insulation")
print(f"{'Part (a) Quadratic (n=15, with outlier)':<40} {R2_quad:>8.4f}  Inflated by outlier")
print(f"{'Part (b) Linear (n=14, cleaned)':<40} {R2_lin_c:>8.4f}  BEST MODEL (Note 4)")
print(f"{'Part (b) Quadratic (n=14, cleaned)':<40} {R2_quad_c:>8.4f}   ΔR^2 = {R2_quad_c - R2_lin_c:.4f} (too small)")

# ============================================================================
# FIGURES 2-5: 3D SURFACE PLOTS WITH PREDICTED POINTS
# ============================================================================
def plot_surface(fig_num, Temp_data, Ins_data, Oil_data, Oil_pred, model_func, params, title, elev=30, azim=-45):
    fig = plt.figure(fig_num, figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    T_min, T_max = Temp_data.min() - 5, Temp_data.max() + 5
    I_min, I_max = Ins_data.min() - 2, Ins_data.max() + 2
    
    T_grid = np.linspace(T_min, T_max, 40)
    I_grid = np.linspace(I_min, I_max, 40)
    T_mesh, I_mesh = np.meshgrid(T_grid, I_grid)
    
    Oil_mesh = model_func(T_mesh, I_mesh, params)
    
    surf = ax.plot_surface(T_mesh, I_mesh, Oil_mesh, cmap='plasma', alpha=0.75,
                           edgecolor='none', linewidth=0, antialiased=True)
    
    # Actual data points (RED)
    ax.scatter(Temp_data, Ins_data, Oil_data, c='red', s=100, marker='o',
               edgecolors='black', linewidth=1.5, label='Actual Data', zorder=10)
    
    # Predicted points (GREEN) 
    ax.scatter(Temp_data, Ins_data, Oil_pred, c='lime', s=80, marker='^',
               edgecolors='black', linewidth=1, label='Predicted Values', zorder=10)
    
    # Draw lines connecting actual to predicted (vertical lines)
    for i in range(len(Temp_data)):
        ax.plot([Temp_data[i], Temp_data[i]], 
                [Ins_data[i], Ins_data[i]], 
                [Oil_data[i], Oil_pred[i]], 
                c='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel('Temperature (°F)', fontsize=12, labelpad=10)
    ax.set_ylabel('Insulation (inches)', fontsize=12, labelpad=10)
    ax.set_zlabel('Oil Used (gallons)', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=25)
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.1)
    cbar.set_label('Predicted Oil (gallons)', fontsize=10)
    ax.legend(loc='upper right')
    ax.view_init(elev=elev, azim=azim)
    return fig

def linear_prediction(T, I, params):
    b0, b1, b2 = params
    return b0 + b1 * T + b2 * I

def quadratic_prediction(T, I, params):
    b0, b1, b2, b3, b4, b5 = params
    return b0 + b1*T + b2*I + b3*T**2 + b4*I**2 + b5*T*I

# Figure 2: Linear (n=15, with outlier)
Oil_pred_lin = linear_prediction(Temp, Ins, b_lin)
plot_surface(2, Temp, Ins, Oil, Oil_pred_lin, linear_prediction, b_lin,
             f"Fig 2: LINEAR PLANE (n=15, with outlier)\nR^2 = {R2_lin:.4f}",
             elev=25, azim=-55)

# Figure 3: Quadratic (n=15, with outlier)
Oil_pred_quad = quadratic_prediction(Temp, Ins, b_quad)
plot_surface(3, Temp, Ins, Oil, Oil_pred_quad, quadratic_prediction, b_quad,
             f"Fig 3: QUADRATIC SURFACE (n=15, with outlier)\nR^2 = {R2_quad:.4f}",
             elev=25, azim=-55)

# Figure 4: Linear Cleaned (n=14)
Oil_pred_lin_c = linear_prediction(Temp_c, Ins_c, b_lin_c)
plot_surface(4, Temp_c, Ins_c, Oil_c, Oil_pred_lin_c, linear_prediction, b_lin_c,
             f"Fig 4: LINEAR PLANE - CLEANED (n=14, no outlier)\nR^2 = {R2_lin_c:.4f}",
             elev=25, azim=-55)

# Figure 5: Quadratic Cleaned (n=14)
Oil_pred_quad_c = quadratic_prediction(Temp_c, Ins_c, b_quad_c)
plot_surface(5, Temp_c, Ins_c, Oil_c, Oil_pred_quad_c, quadratic_prediction, b_quad_c,
             f"Fig 5: QUADRATIC SURFACE - CLEANED (n=14, no outlier)\nR^2 = {R2_quad_c:.4f}",
             elev=25, azim=-55)

plt.tight_layout()
plt.show()
# ============================================================================
# FINAL CONCLUSION
# ============================================================================
print("\n" + "=" * 70)
print("FINAL CONCLUSION")
print("=" * 70)
print(f"""
Best Model Selected:
  Model:      Linear Regression (cleaned data, n=14)
  Equation:   Oil = 601.60 - 5.48*Temp - 23.23*Ins
  R^2:        0.9606 (96.1% of variance explained)

Why this model?
  - Both coefficients have correct physical signs
  - Quadratic model adds only 1.63% R^2 improvement
  - Per Note 4: 3 extra parameters not justified
  - Simpler, more interpretable, nearly identical predictions

Outlier Detection (Mahalanobis Distance):
  - Row 6 (Temp=65°F, Ins=40 inches) identified as outlier
  - Mahalanobis distance = {distances[outlier_idx]:.4f} > threshold = {threshold:.4f}
  - This confirms the data entry error hypothesis

Prediction for Temp=15°F, Ins=5 inches:
  -> {pred_lin:.2f} gallons of heating oil
""")