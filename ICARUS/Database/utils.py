def ang2case(angle):
    if angle >= 0:
        folder = str(angle)[::-1].zfill(7)[::-1] + "_AoA"
    else:
        folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1] + "_AoA"
    return folder


def dst2case(dst):
    if dst.var == "Trim":
        folder = "Trim"
    elif dst.isPositive:
        folder = "p" + str(dst.amplitude)[::-1].zfill(6)[::-1] + f"_{dst.var}"
    else:
        folder = (
            "m" + str(dst.amplitude)[::-1].strip("-").zfill(6)[::-1] + f"_{dst.var}"
        )
    return folder
