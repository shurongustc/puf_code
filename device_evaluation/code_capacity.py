import numpy as np

def calculate_puf_capacity(C, D, L, grid_resolution):
    """
    Calculate the theoretical PUF encoding capacity for a custom grid.
    :param C: Number of effective grayscale levels
    :param D: Fill density
    :param L: Subcell length
    :param grid_resolution: An integer R (representing an R x R square grid),
                            or a tuple (R_width, R_height) for a rectangular grid
    """
    # 1. Calculate the number of combinations per cell (base).
    base = C * (1 + L * (np.sqrt(1 / D) - 1))**2 + 1
    
    # 2. Calculate the total grid degrees of freedom/exponent.
    if isinstance(grid_resolution, tuple):
        exponent = grid_resolution[0] * grid_resolution[1]
        grid_desc = f"{grid_resolution[0]} × {grid_resolution[1]} (Non-square)"
    else:
        exponent = grid_resolution**2
        grid_desc = f"{grid_resolution} × {grid_resolution} (Square)"
        
    # 3. Calculate the logarithmic capacity.
    log10_cap = exponent * np.log10(base)
    log2_cap = exponent * np.log2(base)
    
    print("=" * 55)
    print(f"🔧 Custom Grid Configuration Result:")
    print(f"  * Parameters: C = {C}, D = {D}, L = {L}")
    print(f"  * Grid Resolution (R): {grid_desc}")
    print(f"  * Total Grid Cells (Exponent): {exponent:,}")
    print(f"  * Single Cell Base: {base:.4f}")
    print(f"  ---------------------------------------------------")
    print(f"  📈 Code Capacity (Base 10): 10^{log10_cap:,.0f}")
    print(f"  📈 Code Capacity (Base 2):  2^{log2_cap:,.0f}")
    print("=" * 55)
    
    return log10_cap, log2_cap

if __name__ == '__main__':
    # Example 1: Try a square grid (e.g., matching DINOv2's 518-pixel resolution).
    print("--- Scheme 1: 518 x 518 Square Grid ---")
    calculate_puf_capacity(C=141, D=0.9156, L=10, grid_resolution=518)
    
    # Example 2: Try a non-square grid that preserves the original aspect ratio (e.g., 450 wide by 300 high).
    print("\n--- Scheme 2: 450 x 300 Asymmetric Grid ---")
    calculate_puf_capacity(C=141, D=0.9156, L=10, grid_resolution=(450, 300))
    
    # Example 3: Modify and test your own ideas at any time (e.g., R = 300, L = 5).
    print("\n--- Scheme 3: Your Custom Test ---")
    calculate_puf_capacity(C=141, D=0.9156, L=5, grid_resolution=(5,30))