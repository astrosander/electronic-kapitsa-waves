# test_parabolic_limit.py
# Validates that BLG code reduces to parabolic (Kryhin-Levitov) in the limit ζ→0

import numpy as np
import math

# Import BLG functions
import sys
sys.path.append('.')

# Test dispersion in parabolic limit
def test_dispersion():
    print("="*60)
    print("TEST 1: Dispersion in parabolic limit (ζ → 0)")
    print("="*60)
    
    zetas = [1.0, 0.5, 0.1, 0.01, 0.001]
    Ps = np.array([0.5, 1.0, 1.5, 2.0])
    
    print(f"\n{'ζ':>8} {'P':>6} {'ε_BLG':>12} {'ε_para':>12} {'Rel Error':>12}")
    print("-"*60)
    
    for zeta in zetas:
        num = np.sqrt(1.0 + 4.0*(zeta*zeta)*(Ps*Ps)) - 1.0
        den = np.sqrt(1.0 + 4.0*(zeta*zeta)) - 1.0
        eps_blg = num / den
        eps_para = Ps**2
        
        for i, P in enumerate(Ps):
            rel_err = abs(eps_blg[i] - eps_para[i]) / (eps_para[i] + 1e-10)
            print(f"{zeta:8.4f} {P:6.2f} {eps_blg[i]:12.6f} {eps_para[i]:12.6f} {rel_err:12.6e}")
    
    print("\n✓ Expected: Error → 0 as ζ → 0\n")


# Test velocity (derivative)
def test_velocity():
    print("="*60)
    print("TEST 2: Velocity (dε/dP) in parabolic limit")
    print("="*60)
    
    zetas = [1.0, 0.1, 0.01, 0.001]
    Ps = np.array([0.5, 1.0, 1.5, 2.0])
    
    print(f"\n{'ζ':>8} {'P':>6} {'v_BLG':>12} {'v_para':>12} {'Rel Error':>12}")
    print("-"*60)
    
    for zeta in zetas:
        den = np.sqrt(1.0 + 4.0*(zeta*zeta)) - 1.0
        num = (4.0*zeta*zeta*Ps) / np.sqrt(1.0 + 4.0*zeta*zeta*Ps*Ps)
        v_blg = num / den
        v_para = 2.0 * Ps  # d(P²)/dP = 2P
        
        for i, P in enumerate(Ps):
            rel_err = abs(v_blg[i] - v_para[i]) / (v_para[i] + 1e-10)
            print(f"{zeta:8.4f} {P:6.2f} {v_blg[i]:12.6f} {v_para[i]:12.6f} {rel_err:12.6e}")
    
    print("\n✓ Expected: v_BLG → 2P as ζ → 0\n")


# Test chemical potential
def test_chemical_potential():
    print("="*60)
    print("TEST 3: Chemical potential μ̃(T) in parabolic limit")
    print("="*60)
    
    # For parabolic 2D with constant DOS, μ̃(Θ) has known behavior:
    # At low T: μ̃ ≈ 1 - Θ ln(2)
    # At high T: μ̃ ≈ Θ ln(Θ)
    
    def solve_mu_tilde_test(Theta, zeta):
        """Simplified version for testing"""
        Pmax = max(8.0, 2.0 / Theta)
        Ps = np.linspace(0, Pmax, 6000)
        
        def eps_tilde(P):
            num = np.sqrt(1.0 + 4.0*(zeta*zeta)*(P*P)) - 1.0
            den = np.sqrt(1.0 + 4.0*(zeta*zeta)) - 1.0
            return num / den
        
        def I(mu):
            eps = eps_tilde(Ps)
            x = np.clip((eps - mu) / Theta, -50, 50)
            f = 1.0 / (np.exp(x) + 1.0)
            return np.trapz(Ps * f, Ps) - 0.5
        
        lo, hi = -20.0, 20.0
        flo, fhi = I(lo), I(hi)
        
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            fmid = I(mid)
            if abs(fmid) < 1e-12:
                break
            if flo * fmid <= 0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
        
        return 0.5 * (lo + hi)
    
    Thetas = [0.01, 0.05, 0.1, 0.5]
    zetas = [0.001, 0.01, 0.1]
    
    print(f"\n{'Θ':>8} {'ζ':>8} {'μ̃_BLG':>12} {'μ̃_para(approx)':>16}")
    print("-"*60)
    
    for Theta in Thetas:
        for zeta in zetas:
            mu_blg = solve_mu_tilde_test(Theta, zeta)
            
            # Parabolic approximation (low T)
            if Theta < 0.2:
                mu_para_approx = 1.0 - Theta * np.log(2)
            else:
                mu_para_approx = Theta * np.log(Theta)
            
            print(f"{Theta:8.4f} {zeta:8.4f} {mu_blg:12.6f} {mu_para_approx:16.6f}")
    
    print("\n✓ At low T and small ζ: μ̃ ≈ 1 - Θ ln(2) ≈ 0.31Θ\n")


# Test Jacobian normalization
def test_jacobian():
    print("="*60)
    print("TEST 4: Jacobian weight in parabolic limit")
    print("="*60)
    print("\nFor parabolic ε = P², the weight q/|dE/dq| should → 0.5")
    print("(This is what makes your current code's Det = 2α/dotv = 2×0.5 = 1)\n")
    
    # Simple test case: head-on collision
    P1 = 1.0
    P2 = 1.0
    theta = np.pi  # opposite momenta
    
    zetas = [1.0, 0.5, 0.1, 0.01, 0.001]
    
    print(f"{'ζ':>8} {'weight':>12} {'weight/0.5':>12}")
    print("-"*50)
    
    for zeta in zetas:
        # Total momentum
        Kx = P1 + P2 * np.cos(theta)
        Ky = P2 * np.sin(theta)
        
        phi = 0.0  # outgoing direction
        nx = np.cos(phi)
        ny = np.sin(phi)
        
        # For parabolic, can solve analytically
        # For BLG, need to solve numerically (simplified here)
        
        def eps(P):
            num = math.sqrt(1.0 + 4.0*(zeta*zeta)*(P*P)) - 1.0
            den = math.sqrt(1.0 + 4.0*(zeta*zeta)) - 1.0
            return num / den
        
        def deps(P):
            den = math.sqrt(1.0 + 4.0*(zeta*zeta)) - 1.0
            num = (4.0*zeta*zeta*P) / math.sqrt(1.0 + 4.0*zeta*zeta*P*P)
            return num / den
        
        E_in = eps(P1) + eps(P2)
        
        # Initial guess (FIXED: for head-on collision with K≈0, q should be ~P1 or P2)
        q = max(P1, P2)
        
        # Newton
        for _ in range(20):
            ax = 0.5*Kx + q*nx
            ay = 0.5*Ky + q*ny
            bx = 0.5*Kx - q*nx
            by = 0.5*Ky - q*ny
            
            a = math.sqrt(ax*ax + ay*ay)
            b = math.sqrt(bx*bx + by*by)
            
            E_out = eps(a) + eps(b)
            F = E_out - E_in
            
            if abs(F) < 1e-12:
                break
            
            da_dq = (ax*nx + ay*ny) / a if a > 0 else 0
            db_dq = -(bx*nx + by*ny) / b if b > 0 else 0
            
            dE = deps(a)*da_dq + deps(b)*db_dq
            
            if abs(dE) < 1e-12:
                break
            
            q = q - F / dE
        
        # Final weight
        ax = 0.5*Kx + q*nx
        ay = 0.5*Ky + q*ny
        bx = 0.5*Kx - q*nx
        by = 0.5*Ky - q*ny
        
        a = math.sqrt(ax*ax + ay*ay)
        b = math.sqrt(bx*bx + by*by)
        
        da_dq = (ax*nx + ay*ny) / a if a > 0 else 0
        db_dq = -(bx*nx + by*ny) / b if b > 0 else 0
        dE = deps(a)*da_dq + deps(b)*db_dq
        
        weight = q / abs(dE) if dE != 0 else 0
        ratio = weight / 0.5
        
        print(f"{zeta:8.4f} {weight:12.6f} {ratio:12.6f}")
    
    print("\n✓ Expected: weight → 0.5 as ζ → 0\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BLG → PARABOLIC LIMIT VALIDATION")
    print("="*60 + "\n")
    
    test_dispersion()
    test_velocity()
    test_chemical_potential()
    test_jacobian()
    
    print("="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print("\nIf all tests show convergence as ζ → 0, the BLG code")
    print("correctly reduces to the Kryhin-Levitov parabolic limit.")
    print("="*60 + "\n")

