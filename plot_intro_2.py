import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots()

# meta-learning arrow (curved)
arrow = FancyArrowPatch((0,0), (2,-2), 
                        connectionstyle="arc3,rad=-0.2",
                        arrowstyle='-|>', 
                        linewidth=3, color="orange")
ax.add_patch(arrow)

# adapter arrows
for dx, dy in [(1,-1), (1,1)]:
    arrow = FancyArrowPatch((2,-2), (3+dx,-2+dy),
                            arrowstyle='->',
                            linestyle="--",
                            linewidth=2, color="pink")
    ax.add_patch(arrow)

# best performance stars
ax.plot([4,4],[ -1, -3],"*", markersize=20, color="gold")

ax.set_xlim(-1,5)
ax.set_ylim(-4,2)
ax.axis("off")

plt.savefig("./figures/intro2.pdf", bbox_inches='tight', pad_inches=0)