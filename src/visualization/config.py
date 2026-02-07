# =========================
# General figure settings
# =========================
FIGSIZE_SMALL = (8, 5)
FIGSIZE_MEDIUM = (12, 6)
FIGSIZE_LARGE = (18, 5)

DPI = 300
BBOX_INCHES = "tight"
CONSTRAINED_LAYOUT = True
LABEL_ROTATION = 0

# =========================
# Fonts and titles
# =========================
TITLE_FONTSIZE = 14
TITLE_WEIGHT = "bold"
AXIS_LABEL_FONTSIZE = 12
ANNOTATE_FONTSIZE = 11

# =========================
# Colors and palettes
# =========================
COLOR_R = "red"
COLOR_G = "green"
COLOR_B = "blue"

PALETTE_PASTEL = "pastel"
PALETTE_COLORBLIND = "colorblind"
PALETTE_TAB10 = "tab10"
DEFAULT_BAR_COLOR = "skyblue"

# =========================
# Annotation defaults
# =========================
ANNOTATE_XYTEXT = (0, 3)
ANNOTATE_TEXTCOORDS = "offset points"
ANNOTATE_HA = "center"
ANNOTATE_VA = "bottom"

# =========================
# Grid styling
# =========================
GRID_ALPHA = 0.3
GRID_LINESTYLE = "--"
GRID_AXIS = "y"

# =========================
# Legend styling
# =========================
LEGEND_BBOX_TO_ANCHOR = (1.05, 1)
LEGEND_LOCATION = "upper left"
LEGEND_FRAME = False

# =========================
# Scatterplot / marker
# =========================
SCATTER_MARKER_SIZE = 60
SCATTER_ALPHA = 0.7

# =========================
# Log-scale safety
# =========================
EPSILON = 1e-6  # small value to avoid log(0)
FILESIZE = 1e-3  # small value to avoid log(0)
