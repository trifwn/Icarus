def xflr_eigs(filename: str) -> tuple[list[complex], list[complex]]:
    with open(filename, errors="ignore") as f:
        lines = f.readlines()

    long_eigenvalues_line = lines[23]
    long_eig_1: str = long_eigenvalues_line[19:41].replace(" ", "").replace("+-", "-").replace("i", "j")
    long_eig_2: str = long_eigenvalues_line[44:67].replace(" ", "").replace("+-", "-").replace("i", "j")
    long_eig_3: str = long_eigenvalues_line[71:93].replace(" ", "").replace("+-", "-").replace("i", "j")
    long_eig_4: str = long_eigenvalues_line[98:].replace(" ", "").replace("+-", "-").replace("i", "j")
    long_eigs: list[complex] = []
    for item in [long_eig_1, long_eig_2, long_eig_3, long_eig_4]:
        long_eigs.append(complex(item))

    late_eigenvalues_line = lines[34]
    late_eig_1 = late_eigenvalues_line[19:41].replace(" ", "").replace("+-", "-").replace("i", "j")
    late_eig_2 = late_eigenvalues_line[44:67].replace(" ", "").replace("+-", "-").replace("i", "j")
    late_eig_3 = late_eigenvalues_line[71:93].replace(" ", "").replace("+-", "-").replace("i", "j")
    late_eig_4 = late_eigenvalues_line[98:].replace(" ", "").replace("+-", "-").replace("i", "j")

    late_eigs: list[complex] = []
    for item in [late_eig_1, late_eig_2, late_eig_3, late_eig_4]:
        late_eigs.append(complex(item))

    return long_eigs, late_eigs
