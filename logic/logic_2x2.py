import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import streamlit as st
import sympy as sp
from typing import Tuple


class Solve2x2:
    def __init__(self) -> None:
        pass

    def solve_system_2x2(self, A) -> Tuple:
        """
        Just a small utility that returns:
         - exp_At = e^(A)
         - stable (bool): True if real parts of all eigenvalues < 0
         - eigenvalues
         - eigenvectors
         - a random 2D vector sol_y
        """
        eigenvalues, eigenvectors = np.linalg.eig(A)
        stable = all(np.real(eigenvalues) < 0)
        exp_At = expm(A)  # e^A (not e^(A t), but at t=1)
        sol_y = np.random.rand(2, 1)
        return exp_At, stable, eigenvalues, eigenvectors, sol_y

    #
    # ----------------------------
    #  Formatting helpers
    # ----------------------------
    #
    def format_number(self, number, decimals=2) -> str:
        """
        Formats a real number with a given number of decimals, removing trailing zeros.
        """
        try:
            rounded = round(float(number), decimals)
            if abs(rounded) < 1e-14:
                return "0"
            if rounded.is_integer():
                return str(int(rounded))
            txt = f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")
            return txt
        except (TypeError, ValueError):
            return str(number)

    def format_complex(self, z: complex, decimals=2) -> str:
        """
        Formats a complex number a + bi using 'i' notation (instead of Python's 'j') and
        rounds real/imag parts to 'decimals'.
        """
        if not isinstance(z, complex):
            return self.format_number(z, decimals=decimals)

        real_part = round(z.real, decimals)
        imag_part = round(z.imag, decimals)

        if abs(real_part) < 1e-14:
            real_part = 0
        if abs(imag_part) < 1e-14:
            imag_part = 0

        if imag_part == 0:
            return self.format_number(real_part, decimals)
        elif real_part == 0:
            return f"{self.format_number(imag_part, decimals)}i"
        elif imag_part > 0:
            return f"{self.format_number(real_part, decimals)} + {self.format_number(imag_part, decimals)}i"
        else:
            return f"{self.format_number(real_part, decimals)} - {self.format_number(-imag_part, decimals)}i"

    def format_latex_complex_matrix(self, M: np.ndarray, decimals=2) -> str:
        """
        Formats a 2D array of real/complex numbers in LaTeX, using \begin{pmatrix}...\end{pmatrix}.
        """
        rows_str = []
        for row in M:
            row_items = [self.format_complex(val, decimals) for val in row]
            rows_str.append(" & ".join(row_items))
        matrix_str = "\\begin{pmatrix}" + " \\\\ ".join(rows_str) + "\\end{pmatrix}"
        return matrix_str

    def format_latex_string_matrix(self, M: list[list[str]]) -> str:
        """
        If you already have strings for each cell (like "e^{(3+2i)t} cos(...)"),
        arrange them in a LaTeX \begin{pmatrix}...\end{pmatrix}.
        """
        rows_str = []
        for row in M:
            rows_str.append(" & ".join(row))
        matrix_str = "\\begin{pmatrix}" + " \\\\ ".join(rows_str) + "\\end{pmatrix}"
        return matrix_str

    #
    # ----------------------------
    #  Main solution logic
    # ----------------------------
    #
    def get_solutions_2x2(self, eigenvalues, eigenvectors, A) -> list[str]:
        """
        Return a list of LaTeX-formatted strings describing the general solution x(t).
        In some cases, we show x_1(t), x_2(t) separately. In others, we show a single
        matrix expression x(t) = e^{A t} c.
        """
        solutions = []

        # Overall approach: for each special case, we produce "x(t) = ..." in an appropriate form.

        #
        # CASE 1: A is "effectively diagonal" with A[0][1] = 0 and A[1][1] = 0 => trivial structure
        #
        if A[0][1] == 0 and A[1][1] == 0:
            # Then e^(A t) is just diag(e^(a_{00} t), e^(a_{11} t)) if a_{01} = 0, a_{10}=?
            # The code as is, was simply building a diagonal string. We'll keep that idea.
            new_A = A
            r = []
            for i in range(len(new_A)):
                row_str = []
                for j in range(len(new_A)):
                    if i == j:
                        val_str = self.format_number(new_A[i][j], 2)
                        row_str.append(f"e^{{({val_str})t}}")
                    else:
                        row_str.append("0")
                r.append(row_str)
            # Convert to LaTeX
            mat_str = self.format_latex_string_matrix(r)
            solutions.append(
                rf"x(t) = {mat_str} \, c"
            )
            return solutions

        #
        # CASE 2: A is a real 2x2 with A[0,0] == A[1,1] and A[0,1] + A[1,0] = 0 => typical rotation form
        #
        # That is A = aI + bJ, with J = [[0, -1],[1, 0]] => x_1(t), x_2(t) have a standard form
        if (
            self.format_number(A[0][0]) == self.format_number(A[1][1])
            and self.format_number(A[0][1] + A[1][0]) == "0"
        ):
            a = A[0][0]  # real part
            b = A[1][0]  # (assuming A[0,1] = -b)
            # Show them as separate x_1(t) and x_2(t):
            a_str = self.format_number(a, 2)
            b_str = self.format_number(abs(b), 2)  # magnitude in cos/sin
            sign = "-" if b >= 0 else "+"  # because if b=+2 => x_2 uses - sin(...) for x_1
            if b < 0:
                # If b<0, we adjust the sign inside the cos/sin so that it's cos(-bt)=cos(bt)
                # but let's keep it simple: "cos(|b| t) - ..." is standard.
                pass
            # Two separate lines for x_1(t), x_2(t):
            x1 = (
                rf"x_1(t) = e^{{({a_str})t}}\bigl(c_1\cos({b_str}t) "
                + rf"{'-' if b>0 else '+'} c_2\sin({b_str}t)\bigr)"
            )
            x2 = (
                rf"x_2(t) = e^{{({a_str})t}}\bigl(c_1\sin({b_str}t) "
                + rf"{'+' if b>0 else '-'} c_2\cos({b_str}t)\bigr)"
            )

            # Combine into a single string, or keep them separate:
            solutions.append(x1)
            solutions.append(x2)
            # We could also say: x(t) = [x_1(t), x_2(t)]^T if you want:
            return solutions

        #
        # CASE 3: Repeated eigenvalues (Jordan form)
        #
        elif abs(eigenvalues[0] - eigenvalues[1]) < 1e-14:
            # We attempt the typical Jordan approach: e^(A t) = P e^(Lambda t) e^(N t) P^-1
            # For a 2x2 Jordan block, e^(A t) can have t-terms in the off-diagonal, etc.
            try:
                P_inv = np.linalg.inv(eigenvectors)
            except:
                P_inv = None

            diag_l = np.diag(eigenvalues)

            if P_inv is not None:
                # Build e^(A t) = P e^(D t) e^(N t) P^-1. If N=0, that's just diagonal.
                # For a 2x2 Jordan block with repeated λ, e^(A t) = e^(λ t)(I + (A-λI) t + ...)
                # But let's keep the code's "symbolic" approach. We'll show the final as "x(t) = P e^(D t) P^-1 (I + ... ) c"
                #
                # This portion is mostly symbolic. If you want the *actual numeric expansion*, you can do:
                #   expm(A * t) or symbolic Jordan approach. Let's keep it consistent with your original code.
                S = np.einsum("ij,jk,kl->il", eigenvectors, diag_l, P_inv)
                N = A - S
                # We'll see if N is nilpotent:
                # This loop was an attempt to see how many powers matter, but let's keep it short:
                k = 1
                while np.allclose(N, 0):
                    N = np.dot(N, N)
                    k += 1
                    if k >= 2:
                        break

                # Build the diagonal e^(D t)
                diag_end = []
                for i in range(2):
                    row_str = []
                    for j in range(2):
                        if i == j:
                            val_c = self.format_complex(diag_l[i][j], 2)
                            row_str.append(f"e^{{({val_c})t}}")
                        else:
                            row_str.append("0")
                    diag_end.append(row_str)

                # The sum for the Jordan block expansions
                # (though in practice a single Jordan 2x2 might only have 1 or 2 terms).
                # We keep your original approach:
                jordan_sum = ""
                for i_power in range(1, k + 1):
                    if i_power <= 1:
                        jordan_sum += f"+(N^{i_power} t^{i_power})"
                    else:
                        from math import factorial
                        jordan_sum += f"+(N^{i_power} t^{i_power})/{factorial(i_power)}"

                # Tidy up "N^1 t^1" => "N t"
                def fix_extras(expr: str) -> str:
                    terms = expr.split("+")
                    new_terms = []
                    for t in terms:
                        t = t.strip()
                        if not t:
                            continue
                        t = t.replace("^1", "")
                        t = t.replace("t^1", "t")
                        new_terms.append(f"\\bigl({t}\\bigr)")
                    return " + ".join(new_terms)

                # Now latex everything:
                P_str = self.format_latex_complex_matrix(eigenvectors)
                diag_str = self.format_latex_string_matrix(diag_end)
                P_inv_str = self.format_latex_complex_matrix(P_inv)
                sum_str = fix_extras(jordan_sum)

                full_expr = (
                    f"x(t) = {P_str} \\; {diag_str} \\; {P_inv_str} \\;"
                    f"(I {sum_str})\\,c"
                )
                solutions.append(full_expr)
                return solutions
            else:
                # If we can't invert P, we can't do the normal Jordan approach
                solutions.append(
                    "x(t) = P e^{(Lambda t)} P^-1 c, pero ocurrió un error al invertir P."
                )
                return solutions

        #
        # CASE 4: Distinct real eigenvalues (no imaginary part above threshold)
        #
        elif (abs(eigenvalues[0].imag) < 1e-14) and (abs(eigenvalues[1].imag) < 1e-14):
            # A is diagonalizable over R with distinct real eigenvalues => e^(A t) = P e^(D t) P^-1
            try:
                P_inv = np.linalg.inv(eigenvectors)
            except:
                P_inv = None

            if P_inv is not None:
                # Build the diagonal D
                D = np.diag(eigenvalues)
                # We'll build e^(D t):
                eDt = []
                for i in range(2):
                    row_str = []
                    for j in range(2):
                        if i == j:
                            val = self.format_complex(D[i][j], 2)
                            row_str.append(f"e^{{({val})t}}")
                        else:
                            row_str.append("0")
                    eDt.append(row_str)

                # Now the full e^(A t) = P e^(D t) P^-1
                P_str = self.format_latex_complex_matrix(eigenvectors)
                eDt_str = self.format_latex_string_matrix(eDt)
                P_inv_str = self.format_latex_complex_matrix(P_inv)

                # So the general solution is x(t) = e^(A t) c:
                # We'll show e^(A t) in diagonalized form.
                full_expr = (
                    rf"x(t) = {P_str}\,{eDt_str}\,{P_inv_str}\,c"
                )
                solutions.append(full_expr)
                return solutions
            else:
                solutions.append(
                    "x(t) = P e^(D t) P^-1 c, pero P no es invertible."
                )
                return solutions

        #
        # CASE 5: Complex eigenvalues => typical real form is e^(alpha t)*[cos(beta t), sin(beta t)], etc.
        #
        else:
            # For a 2x2 real matrix with complex eigenvalues, they come in conjugate pairs:
            # eigenvalues = alpha ± i beta. We'll parse alpha, beta from e.g. eigenvalues[0].
            lam = eigenvalues[0]  # alpha + i beta
            alpha = lam.real
            beta = abs(lam.imag)

            # The real solution for x(t) in R^2 is:
            #   x_1(t) = e^(alpha t)[ c_1 cos(beta t) - c_2 sin(beta t) ]
            #   x_2(t) = e^(alpha t)[ c_1 sin(beta t) + c_2 cos(beta t) ]
            # We'll do that format:
            alpha_str = self.format_number(alpha, 2)
            beta_str = self.format_number(beta, 2)

            # Build them:
            x1 = (
                rf"x_1(t) = e^{{({alpha_str})t}} \bigl( c_1 \cos({beta_str}t) "
                + rf"- c_2 \sin({beta_str}t) \bigr)"
            )
            x2 = (
                rf"x_2(t) = e^{{({alpha_str})t}} \bigl( c_1 \sin({beta_str}t) "
                + rf"+ c_2 \cos({beta_str}t) \bigr)"
            )
            solutions.append("x(t) = [ x_1(t), x_2(t) ]^T, con:")
            solutions.append(x1)
            solutions.append(x2)
            return solutions

    #
    # ----------------------------
    #  ODE system function & plot
    # ----------------------------
    #
    def system_2x2(self, t, Y, A):
        """
        The ODE: d/dt x(t) = A x(t).
        """
        return A @ Y

    def plot_phase_diagram_2d(self, A) -> None:
        """
        A quick phase portrait in the plane for the system x'(t) = A x(t).
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        t_span = [0, 10]
        t_eval = np.linspace(0, 10, 200)

        # Some initial conditions to see different trajectories
        Y0_points = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([0, -1]),
        ]
        colors = ["b", "g", "r", "orange"]

        for idx, y0 in enumerate(Y0_points):
            sol = solve_ivp(self.system_2x2, t_span, y0, args=(A,), t_eval=t_eval)
            ax.plot(sol.y[0], sol.y[1], color=colors[idx], alpha=0.7)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        st.pyplot(fig)
