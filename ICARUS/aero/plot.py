# from matplotlib.colors import Normalize
# from ICARUS.visualization import create_subplots
# from ICARUS.visualization import flatten_axes

# def plot_gammas(self, ax: Axes3D | None = None) -> None:
#     if ax is None:
#         fig: Figure | None = plt.figure()
#         ax_now: Axes3D = fig.add_subplot(projection="3d")  # type: ignore
#     else:
#         ax_now = ax
#         fig = ax_now.get_figure()

#     if fig is None:
#         raise ValueError("Axes must be part of a figure")
#     self.plot_panels(ax_now)

#     # Plot the gammas by coloring the panels
#     gammas = self.gammas
#     # Normalize the gammas
#     gammas = gammas / jnp.max(jnp.abs(gammas))

#     for i in np.arange(0, self.PANEL_NUM):
#         p1, p3, p4, p2 = self.panels[i, :, :]
#         xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
#         ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
#         zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))
#         # Color the area inside the wireframe
#         ax_now.plot_surface(
#             xs,
#             ys,
#             zs,
#             color=coolwarm(gammas[i]),
#             alpha=0.9,
#         )

#     # Add colorbar
#     norm = Normalize(vmin=jnp.min(gammas).item(), vmax=jnp.max(gammas).item())
#     sm = plt.cm.ScalarMappable(cmap=coolwarm, norm=norm)
#     sm.set_array([])
#     fig.colorbar(sm, ax=ax_now, label="Gamma")

#     ax_now.set_title("Grid")
#     ax_now.set_xlabel("x")
#     ax_now.set_ylabel("y")
#     ax_now.set_zlabel("z")
#     ax_now.legend()

# def plot_surface_gamma_distribution(self, axs: list[Axes] | None = None) -> None:
#     if axs is not None:
#         axs_: list[Axes] = flatten_axes(axs)
#         fig = axs_[0].get_figure()
#         if fig is None:
#             raise ValueError("Axes must be part of a figure")
#     else:
#         fig, axs_ = create_subplots(
#             nrows=len(self.surfaces),
#             ncols=1,
#             squeeze=False,
#         )

#     # Plot the gamma distribution on the wing surface
#     for surface_name, surf_dict in self.surface_dict.items():
#         surf_id: int = surf_dict["id"]
#         ax: Axes = axs_[surf_id]
#         ax.set_title(f"{surface_name} Gamma Distribution")
#         ax.set_xlabel("x")
#         ax.set_ylabel("y")
#         # Add a 2D plot of the gamma distribution
#         gammas = self.gammas
#         # Find the indices of the panels that belong to the surface
#         panel_indices = surf_dict["panel_idxs"]
#         gammas_surf = gammas[panel_indices]
#         N: int = surf_dict["N"]
#         M: int = surf_dict["M"]

#         # Reshape the gammas to the grid
#         gammas_surf = jnp.reshape(gammas_surf, (N - 1, M - 1))
#         ax.matshow(
#             gammas_surf,
#             cmap=viridis,
#         )
#         # Add colorbar
#         norm = Normalize(
#             vmin=jnp.min(gammas_surf).item(),
#             vmax=jnp.max(gammas_surf).item(),
#         )
#         sm = plt.cm.ScalarMappable(cmap=viridis, norm=norm)
#         sm.set_array([])
#         fig.colorbar(sm, ax=ax, label="Gamma")

# def plot_L_pan(self, ax: Axes3D | None = None) -> None:
#     if ax is None:
#         fig: Figure | None = plt.figure()
#         ax_now: Axes3D = fig.add_subplot(projection="3d")  # type: ignore
#     else:
#         ax_now = ax
#         fig = ax_now.get_figure()
#     if fig is None:
#         raise ValueError("Axes must be part of a figure")
#     self.plot_panels(ax_now)

#     # Plot the L_pan by coloring the panels
#     L_pan = self.L_pan

#     for i in np.arange(0, self.NM):
#         p1, p3, p4, p2 = self.panels[i, :, :]
#         xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
#         ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
#         zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))
#         ax_now.plot_surface(xs, ys, zs, color=coolwarm(L_pan[i]), alpha=0.9)

#     # Add colorbar
#     norm = Normalize(vmin=jnp.min(L_pan).item(), vmax=jnp.max(L_pan).item())
#     sm = plt.cm.ScalarMappable(cmap=coolwarm, norm=norm)
#     sm.set_array([])
#     fig.colorbar(sm, ax=ax_now, label="L_pan")

# def plot_D_pan(self, ax: Axes3D | None = None) -> None:
#     if ax is None:
#         fig: Figure | None = plt.figure()
#         ax_now: Axes3D = fig.add_subplot(projection="3d")  # type: ignore
#     else:
#         ax_now = ax
#         fig = ax_now.get_figure()

#     if fig is None:
#         raise ValueError("Axes must be part of a figure")

#     self.plot_panels(ax_now)

#     # Plot the D_pan by coloring the panels
#     D_pan = self.D_pan

#     for i in np.arange(0, self.NM):
#         p1, p3, p4, p2 = self.panels[i, :, :]
#         xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
#         ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
#         zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))
#         print(D_pan[i], i)
#         ax_now.plot_surface(xs, ys, zs, color=coolwarm(D_pan[i]), alpha=0.9)

#     # Add colorbar
#     norm = Normalize(vmin=jnp.min(D_pan).item(), vmax=jnp.max(D_pan).item())
#     sm = plt.cm.ScalarMappable(cmap=coolwarm, norm=norm)
#     sm.set_array([])
#     fig.colorbar(sm, ax=ax_now, label="D_pan")
