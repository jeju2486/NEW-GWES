#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(data.table))

# -------------------------
# Args / parsing
# -------------------------

usage_and_quit <- function() {
  cat("Usage:\n")
  cat("  Rscript plot_stage7_8_pangwes_fast.R pairs_stage7.tsv out.png \\\n")
  cat("    [y_col] [n_rows] [ld_dist] [ld_dist_alt] \\\n")
  cat("    [stage8.tsv] [q_thresh] [min_dist] [max_points] [seed] [mask_cache_rds] \\\n")
  cat("    [inblock_prefix_or_dir]\n")
  quit(status = 1)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) usage_and_quit()

infile       <- args[1]
outfile      <- args[2]
y_col        <- if (length(args) >= 3)  args[3]  else "rMI_e"
n_rows       <- if (length(args) >= 4)  as.numeric(args[4])  else 0
ld_dist      <- if (length(args) >= 5)  as.numeric(args[5])  else 0
ld_dist_alt  <- if (length(args) >= 6)  as.numeric(args[6])  else 0
stage8_path  <- if (length(args) >= 7)  args[7]  else NA
q_thresh     <- if (length(args) >= 8)  as.numeric(args[8])  else 0.05
min_dist     <- if (length(args) >= 9)  as.numeric(args[9])  else 0
max_points   <- if (length(args) >= 10) as.numeric(args[10]) else 0
seed         <- if (length(args) >= 11) as.integer(args[11]) else 1L
mask_cache   <- if (length(args) >= 12) args[12] else NA
inblock_path <- if (length(args) >= 13) args[13] else NA  # <-- OPTIONAL LAST ARG

cat("Distance vs residual epistasis plot (PAN-GWES style) — Stage7/8 (FAST)\n")
cat("Input (Stage7): ", infile, "\n", sep = "")
cat("Output: ", outfile, "\n", sep = "")
cat("y_col: ", y_col, "\n", sep = "")
cat("min_dist filter: ", min_dist, "\n", sep = "")
if (!is.na(stage8_path) && nzchar(stage8_path)) cat("Stage8: ", stage8_path, " (q<", q_thresh, ")\n", sep = "")
if (!is.na(inblock_path) && nzchar(inblock_path)) cat("True-epistasis inblock: ", inblock_path, "\n", sep = "")
if (max_points > 0) cat("max_points: ", max_points, " (downsample background only)\n", sep = "")

if (!file.exists(infile)) stop("Input file does not exist: ", infile, call. = FALSE)

t0 <- proc.time()[[3]]

# -------------------------
# Helpers
# -------------------------

coerce_int_safe <- function(x) {
  xi <- suppressWarnings(as.integer(x))
  if (all(!is.na(xi))) return(xi)
  as.character(x)
}

infer_pair_cols <- function(hdr) {
  if (("unitig_i" %in% hdr) && ("unitig_j" %in% hdr)) return(c("unitig_i", "unitig_j"))
  if (("v" %in% hdr) && ("w" %in% hdr)) return(c("v", "w"))
  stop("Could not find pair columns. Need (unitig_i,unitig_j) or (v,w).", call. = FALSE)
}

pick_y_col <- function(hdr, y_col) {
  if (y_col %in% hdr) return(y_col)
  cand <- c("srMI_e", "rMI_e", "rMI_10", "rlogOR", "delta11", "MI_obs_e", "MI_null_e")
  pick <- cand[cand %in% hdr]
  if (length(pick) == 0) stop("y_col not found and no fallback columns available.", call. = FALSE)
  cat("Requested y_col not found; using fallback y_col=", pick[1], "\n", sep = "")
  pick[1]
}

# Block file parsing
extract_block_idx <- function(fn) {
  bn <- basename(fn)
  m <- regexec("_block_([0-9]+)_", bn)
  mm <- regmatches(bn, m)
  if (length(mm) >= 1 && length(mm[[1]]) >= 2) return(as.integer(mm[[1]][2]))
  m2 <- regexec("_block_([0-9]+)", bn)
  mm2 <- regmatches(bn, m2)
  if (length(mm2) >= 1 && length(mm2[[1]]) >= 2) return(as.integer(mm2[[1]][2]))
  NA_integer_
}

build_mask_cache_path <- function(inblock_path, mask_cache) {
  if (!is.na(mask_cache) && nzchar(mask_cache)) return(mask_cache)
  if (file.exists(inblock_path) && file.info(inblock_path)$isdir) {
    return(file.path(inblock_path, "unitig_blockmask_cache.rds"))
  }
  paste0(inblock_path, "_unitig_blockmask_cache.rds")
}

load_or_build_unitig_masks <- function(inblock_path, cache_path) {
  if (file.exists(cache_path)) {
    cat("Loading cached unitig->mask map: ", cache_path, "\n", sep = "")
    return(readRDS(cache_path))
  }

  per_block_files <- character(0)
  if (file.exists(inblock_path) && file.info(inblock_path)$isdir) {
    per_block_files <- list.files(inblock_path, pattern = "_block_[0-9]+_.*\\.txt$", full.names = TRUE)
  } else {
    per_block_files <- unique(Sys.glob(paste0(inblock_path, "_block_*.txt")))
  }

  if (length(per_block_files) == 0) {
    warning("No per-block files found for inblock_path.")
    return(data.table(unitig = integer(), mask = integer()))
  }

  cat("Detected ", length(per_block_files), " per-block unitig files.\n", sep = "")

  dt_list <- vector("list", length(per_block_files))
  keep <- 0L

  for (fn in per_block_files) {
    bidx <- extract_block_idx(fn)
    if (is.na(bidx)) next
    if (bidx >= 30) warning("block_idx>=30 may overflow 32-bit bitmask in base R.")
    bit <- bitwShiftL(1L, bidx)

    u <- tryCatch(
      fread(fn, header = FALSE, sep = "\n", col.names = "unitig", showProgress = FALSE)[["unitig"]],
      error = function(e) character(0)
    )
    if (length(u) == 0) next

    ui <- suppressWarnings(as.integer(u))
    if (all(!is.na(ui))) {
      dt_list[[keep + 1L]] <- data.table(unitig = ui, mask = bit)
    } else {
      dt_list[[keep + 1L]] <- data.table(unitig = as.character(u), mask = bit)
    }
    keep <- keep + 1L
  }

  if (keep == 0L) {
    warning("No unitigs parsed from per-block files.")
    return(data.table(unitig = integer(), mask = integer()))
  }

  dt_all <- rbindlist(dt_list[seq_len(keep)], use.names = TRUE)
  dt_map <- dt_all[, .(mask = Reduce(bitwOr, unique(mask))), by = unitig]
  setkey(dt_map, unitig)

  saveRDS(dt_map, cache_path)
  cat("Wrote cached unitig->mask map: ", cache_path, "\n", sep = "")
  dt_map
}

# -------------------------
# Load Stage7
# -------------------------

hdr <- names(fread(infile, nrows = 0L, sep = "\t", showProgress = FALSE))
if (!("distance" %in% hdr)) stop("Column 'distance' not found.", call. = FALSE)

pair_cols <- infer_pair_cols(hdr)
ui_col <- pair_cols[1]
uj_col <- pair_cols[2]
has_missing <- ("missing_loci" %in% hdr)

y_col <- pick_y_col(hdr, y_col)
need_rmi_rlo <- identical(y_col, "srMI_e")

cols_needed <- c("distance", ui_col, uj_col)
if (has_missing) cols_needed <- c(cols_needed, "missing_loci")
if (need_rmi_rlo) {
  if (!("rMI_e" %in% hdr)) stop("Need column rMI_e to compute srMI_e.", call. = FALSE)
  if (!("rlogOR" %in% hdr)) stop("Need column rlogOR to compute srMI_e.", call. = FALSE)
  cols_needed <- c(cols_needed, "rMI_e", "rlogOR")
} else {
  cols_needed <- c(cols_needed, y_col)
}

n_to_read <- if (n_rows <= 0) Inf else as.integer(n_rows)

dt7 <- fread(
  infile,
  sep = "\t",
  nrows = n_to_read,
  select = cols_needed,
  showProgress = TRUE
)

cat("Read Stage7: ", nrow(dt7), " rows, ", ncol(dt7), " columns (selected).\n", sep = "")

dt7[, distance := suppressWarnings(as.numeric(distance))]
dt7 <- dt7[is.finite(distance) & distance >= min_dist]

if (has_missing) {
  dt7[, missing_loci := suppressWarnings(as.numeric(missing_loci))]
  dt7 <- dt7[is.finite(missing_loci) & missing_loci == 0]
}

if (nrow(dt7) == 0) stop("No usable rows after filtering.", call. = FALSE)

if (need_rmi_rlo) {
  dt7[, rMI_e := suppressWarnings(as.numeric(rMI_e))]
  dt7[, rlogOR := suppressWarnings(as.numeric(rlogOR))]
  dt7 <- dt7[is.finite(rMI_e) & is.finite(rlogOR)]
  dt7[, y := rMI_e * sign(rlogOR)]
} else {
  dt7[, y := suppressWarnings(as.numeric(get(y_col)))]
  dt7 <- dt7[is.finite(y)]
}

dt7[, ui := coerce_int_safe(get(ui_col))]
dt7[, uj := coerce_int_safe(get(uj_col))]
dt7[, a := pmin(ui, uj)]
dt7[, b := pmax(ui, uj)]

t1 <- proc.time()[[3]]
cat(sprintf("Time after Stage7 load+filter: %.2fs\n", t1 - t0))

# -------------------------
# Stage8 overlay (green)
# -------------------------

sig_pairs <- rep(FALSE, nrow(dt7))

if (!is.na(stage8_path) && nzchar(stage8_path)) {
  if (!file.exists(stage8_path)) stop("Stage8 file not found: ", stage8_path, call. = FALSE)

  dt8 <- fread(stage8_path, sep = "\t", select = c("v", "w", "q_primary"), showProgress = TRUE)
  dt8[, q_primary := suppressWarnings(as.numeric(q_primary))]
  dt8 <- dt8[is.finite(q_primary) & q_primary < q_thresh]

  if (nrow(dt8) == 0) {
    cat("Stage8: no pairs with q<", q_thresh, " ; overlay empty.\n", sep = "")
  } else {
    dt8[, v := coerce_int_safe(v)]
    dt8[, w := coerce_int_safe(w)]
    dt8[, a := pmin(v, w)]
    dt8[, b := pmax(v, w)]
    dt8 <- dt8[, .(q = min(q_primary)), by = .(a, b)]
    setkey(dt8, a, b)

    q7 <- dt8[dt7, on = .(a, b), q]
    sig_pairs <- is.finite(q7) & (q7 < q_thresh)
    cat("Significant pairs in plotted set (q<", q_thresh, "): ", sum(sig_pairs), "\n", sep = "")
  }
}

t2 <- proc.time()[[3]]
cat(sprintf("Time after Stage8 overlay: %.2fs\n", t2 - t1))

# -------------------------
# True-epistasis overlay (red) OPTIONAL LAST ARG
# -------------------------

highlight_pairs <- rep(FALSE, nrow(dt7))

if (!is.na(inblock_path) && nzchar(inblock_path)) {
  cache_path <- build_mask_cache_path(inblock_path, mask_cache)
  dt_map <- load_or_build_unitig_masks(inblock_path, cache_path)

  if (nrow(dt_map) > 0) {
    idx1 <- match(dt7$ui, dt_map$unitig)
    idx2 <- match(dt7$uj, dt_map$unitig)
    m1 <- dt_map$mask[idx1]
    m2 <- dt_map$mask[idx2]
    m1[is.na(m1)] <- 0L
    m2[is.na(m2)] <- 0L
    highlight_pairs <- (m1 != 0L) & (m2 != 0L) & (bitwAnd(m1, m2) == 0L)
    cat("True-epistasis highlighted pairs (different-block rule): ", sum(highlight_pairs), "\n", sep = "")
  } else {
    warning("Empty unitig->mask map; no in-block highlighting.")
  }
} else {
  cat("No inblock_path provided; true-epistasis overlay disabled.\n")
}

t3 <- proc.time()[[3]]
cat(sprintf("Time after in-block highlight: %.2fs\n", t3 - t2))

# -------------------------
# Downsample background only
# -------------------------

if (max_points > 0 && nrow(dt7) > max_points) {
  keep <- sig_pairs | highlight_pairs
  n_keep <- sum(keep)
  if (n_keep >= max_points) {
    cat("Downsampling: keep-set already >= max_points; keeping only sig/highlight.\n")
    sel <- keep
  } else {
    set.seed(seed)
    need <- max_points - n_keep
    bg_idx <- which(!keep)
    take <- sample(bg_idx, size = need, replace = FALSE)
    sel <- keep
    sel[take] <- TRUE
  }
  dt7 <- dt7[sel]
  sig_pairs <- sig_pairs[sel]
  highlight_pairs <- highlight_pairs[sel]
  cat("After downsampling: ", nrow(dt7), " points.\n", sep = "")
}

# -------------------------
# Plot (draw RED highlight last)
# -------------------------

distance <- dt7$distance
y <- dt7$y

max_distance <- max(distance, na.rm = TRUE)
exponent <- max(0, round(log10(max_distance)) - 1)
if (!is.finite(exponent)) exponent <- 0
step <- 10^exponent

y_range <- range(y, finite = TRUE)
if (!all(is.finite(y_range))) stop("Non-finite y-range for plotting.")
if (y_range[1] == y_range[2]) {
  eps <- ifelse(y_range[1] == 0, 1, abs(y_range[1]) * 0.1)
  y_range <- y_range + c(-eps, eps)
}
y_ticks <- pretty(y_range, n = 8)

col_points    <- rgb(0, 115, 190, maxColorValue = 255)  # blue
col_sig       <- "darkgreen"
col_highlight <- "red"
col_ld        <- "red"
col_ld_alt    <- "hotpink1"

pch_pt <- 19
cex_pt <- 0.2

outdir <- dirname(outfile)
if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

png(outfile, width = 1920, height = 1080, pointsize = 16)

plot(distance, y,
     col   = col_points,
     type  = "p",
     pch   = pch_pt,
     cex   = cex_pt,
     xlim  = c(0, max_distance),
     ylim  = y_range,
     xaxs  = "i",
     yaxs  = "i",
     xlab  = "",
     ylab  = "",
     xaxt  = "n",
     yaxt  = "n",
     bty   = "n")

# Draw LD lines BEFORE overlays, so red highlight points always sit on top
if (ld_dist > 0)     segments(ld_dist, y_range[1], ld_dist, y_range[2], col = col_ld, lty = 2)
if (ld_dist_alt > 0) segments(ld_dist_alt, y_range[1], ld_dist_alt, y_range[2], col = col_ld_alt, lty = 2)

# Stage8 significant overlay (green)
if (any(sig_pairs)) {
  points(distance[sig_pairs], y[sig_pairs], col = col_sig, pch = pch_pt, cex = cex_pt)
}

# True epistasis overlay (red) LAST
if (any(highlight_pairs)) {
  points(distance[highlight_pairs], y[highlight_pairs], col = col_highlight, pch = pch_pt, cex = cex_pt)
}

axis(1, at = seq(0, max_distance, step), tick = FALSE, labels = seq(0, max_distance / step), line = -0.8)
title(xlab = "Distance between unitigs (bp)", line = 1.2)
title(xlab = substitute(x10^exp, list(exp = exponent)), line = 1.4, adj = 1)

axis(2, at = y_ticks, labels = FALSE, tcl = -0.5)
axis(2, at = y_ticks, labels = y_ticks, las = 1, tcl = -0.5)
title(ylab = paste0(y_col, " (linear)"), line = 2.5)

dev.off()

t_end <- proc.time()[[3]]
cat(sprintf("Wrote: %s\n", outfile))
cat(sprintf("Time plotting: %.2fs\n", t_end - t3))
cat(sprintf("Total time: %.2fs\n", t_end - t0))